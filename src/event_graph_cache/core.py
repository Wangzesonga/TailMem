"""Minimal, query-independent Bipartite Event Graph KV scorer.

This module contains the frozen method used by the paper.  It changes only
which historical KV rows survive block-prefill eviction.  It does not modify
causal attention, generation, per-head capacity, or episode routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EventGraphConfig:
    """Frozen paper configuration.

    ``group_size`` is the number of consecutive original positions in a micro
    group.  ``landmarks`` and ``semantic_neighbors`` bound the bipartite graph
    to O(number_of_groups * landmarks) similarity work per head.
    """

    group_size: int = 4
    landmarks: int = 96
    semantic_neighbors: int = 8
    min_similarity: float = 0.12
    alpha: float = 1.0
    self_completion: float = 0.70
    neighbor_completion: float = 0.55
    event_decay: float = 0.65

    def validate(self) -> "EventGraphConfig":
        if self.group_size != 4:
            raise ValueError("the released sync-free method fixes group_size=4")
        if self.landmarks < 1 or self.semantic_neighbors < 0:
            raise ValueError("invalid graph width")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        for name in ("alpha", "self_completion", "neighbor_completion", "event_decay"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return self


@dataclass
class HeadGraph:
    """Transient graph for one layer and one KV head."""

    eligible: torch.Tensor
    inverse: torch.Tensor
    group_ids: torch.Tensor
    token_salience: torch.Tensor
    roles: torch.Tensor
    group_salience: torch.Tensor
    node_keys: torch.Tensor
    node_values: torch.Tensor
    neighbors: Optional[torch.Tensor] = None
    semantic_width: int = 0
    local_gate: Optional[torch.Tensor] = None

    @property
    def groups(self) -> int:
        return int(self.group_ids.numel())


@dataclass(frozen=True)
class SelectionResult:
    """Masks selected under the native cross-head quota allocation."""

    masks: tuple[torch.Tensor, ...]
    native_masks: tuple[torch.Tensor, ...]
    head_quotas: tuple[int, ...]


def tie_aware_percentile(x: torch.Tensor) -> torch.Tensor:
    """Ascending tie-aware midrank on the final axis.

    Sort + two fixed-shape ``searchsorted`` calls exactly reproduce the
    research implementation's tie semantics without ``unique_consecutive``.
    Constant rows map to zero.
    """

    width = x.shape[-1]
    if width < 2:
        return torch.zeros_like(x, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("percentile input contains non-finite values")
    ordered = torch.sort(x, dim=-1).values
    left = torch.searchsorted(ordered, x, right=False)
    right = torch.searchsorted(ordered, x, right=True)
    middle = (left.float() + right.float() - 1.0) * 0.5
    first_right = torch.searchsorted(
        ordered, ordered[..., :1].contiguous(), right=True
    ).float()
    last_left = torch.searchsorted(
        ordered, ordered[..., -1:].contiguous(), right=False
    ).float()
    lower = (first_right - 1.0) * 0.5
    upper = (last_left + float(width - 1)) * 0.5
    return (middle - lower) / (upper - lower).clamp_min(1.0)


def undo_rotary(
    keys: torch.Tensor,
    positions: torch.Tensor,
    metadata: Mapping[str, object],
) -> torch.Tensor:
    """Recover content keys from fully RoPE-rotated cached keys.

    Supported metadata keys are ``inv_freq`` and ``attention_scaling``.  This
    inversion requires full rotary dimensions and a fixed-frequency RoPE
    variant (``default`` or ``llama3``).
    """

    kind = str(metadata.get("rope_type", "default"))
    if kind not in ("default", "llama3"):
        raise ValueError(f"unsupported RoPE type: {kind}")
    if keys.ndim != 2 or positions.shape != (len(keys),) or keys.shape[-1] % 2:
        raise ValueError("invalid key or position shape")
    inv_freq = torch.as_tensor(metadata["inv_freq"], device=keys.device, dtype=torch.float32)
    if inv_freq.numel() * 2 != keys.shape[-1]:
        raise ValueError("partial rotary dimensions are unsupported")
    phase = positions.float()[:, None] * inv_freq[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    scale = float(metadata.get("attention_scaling", 1.0))
    cos = (phase.cos() * scale).to(keys.dtype).float()
    sin = (phase.sin() * scale).to(keys.dtype).float()
    x = keys.float()
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[:, half:], x[:, :half]), dim=-1)
    return (x * cos - rotated * sin) / (cos.square() + sin.square()).clamp_min(1e-20)


def _micro_groups(
    content_keys: torch.Tensor,
    positions: torch.Tensor,
    system_tokens: int,
    config: EventGraphConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if content_keys.ndim != 2 or positions.shape != (len(content_keys),):
        raise ValueError("keys must be [tokens, dim] and positions [tokens]")
    if len(positions) > 1 and not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError("positions must be strictly increasing")
    if not torch.isfinite(content_keys).all():
        raise ValueError("keys contain non-finite values")
    eligible = torch.nonzero(positions >= system_tokens).flatten()
    group_ids, inverse = torch.unique(
        (positions[eligible] - system_tokens) // config.group_size,
        sorted=True,
        return_inverse=True,
    )
    return eligible, inverse, group_ids


def _token_salience(
    content_keys: torch.Tensor,
    content_values: Optional[torch.Tensor],
    eligible: torch.Tensor,
    inverse: torch.Tensor,
    group_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    result = content_keys.new_zeros(len(content_keys), dtype=torch.float32)
    count = int(eligible.numel())
    if count < 2:
        if count == 1:
            result[eligible] = 1.0
        return result

    key = F.normalize(content_keys[eligible].float(), dim=-1, eps=1e-12)
    key_mean = F.normalize(key.mean(0, keepdim=True), dim=-1, eps=1e-12)
    raw_global = -(key * key_mean).sum(-1)

    # Four micro-groups form the frozen 16-token local contrast window.
    group_window = group_ids // 4
    boundary = torch.ones_like(group_window, dtype=torch.long)
    if len(group_window) > 1:
        boundary[1:] = (group_window[1:] != group_window[:-1]).long()
    group_to_window = boundary.cumsum(0) - 1
    window_inverse = group_to_window[inverse]
    capacity = len(group_ids)
    sums = key.new_zeros((capacity, key.shape[-1]))
    sums.index_add_(0, window_inverse, key)
    sizes = torch.bincount(window_inverse, minlength=capacity).to(key.dtype)
    local_mean = F.normalize(sums / sizes[:, None].clamp_min(1), dim=-1, eps=1e-12)
    raw_local = 1.0 - (key * local_mean[window_inverse]).sum(-1)

    if content_values is not None:
        value_raw = content_values[eligible].float()
        value_norm = value_raw.norm(p=2, dim=-1)
        value = F.normalize(value_raw, dim=-1, eps=1e-12)
        value_mean = F.normalize(value.mean(0, keepdim=True), dim=-1, eps=1e-12)
        raw_value_isolation = -(value * value_mean).sum(-1)
        raw_value_energy = torch.log1p(
            value_norm / value_norm.mean().clamp_min(1e-12)
        )
        ranked = tie_aware_percentile(
            torch.stack((raw_global, raw_local, raw_value_isolation, raw_value_energy))
        )
        value_potential = tie_aware_percentile(0.6 * ranked[2] + 0.4 * ranked[3])
        combined = 0.50 * ranked[0] + 0.25 * ranked[1] + 0.25 * value_potential
    else:
        ranked = tie_aware_percentile(torch.stack((raw_global, raw_local)))
        combined = 0.65 * ranked[0] + 0.35 * ranked[1]

    if count > 2:
        adjacent = positions[eligible][1:] - positions[eligible][:-1] <= 2
        left = torch.zeros_like(combined)
        right = torch.zeros_like(combined)
        left[1:] = torch.where(adjacent, combined[:-1], left[1:])
        right[:-1] = torch.where(adjacent, combined[1:], right[:-1])
        combined = combined + 0.15 * (torch.maximum(left, right) - combined).clamp_min(0)
    result[eligible] = tie_aware_percentile(combined)
    return result


def _pool_nodes(
    content_keys: torch.Tensor,
    content_values: Optional[torch.Tensor],
    eligible: torch.Tensor,
    inverse: torch.Tensor,
    groups: int,
    salience: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = content_keys.shape[-1]
    if groups == 0:
        empty = content_keys.new_zeros((0, dim), dtype=torch.float32)
        return empty, empty, content_keys.new_zeros(0, dtype=torch.float32)
    keys = content_keys[eligible].float()
    if len(keys) > 1:
        keys = keys - keys.mean(0, keepdim=True)
    keys = F.normalize(keys, dim=-1, eps=1e-12)
    local_salience = salience[eligible].float()
    peak = local_salience.new_zeros(groups)
    peak.scatter_reduce_(0, inverse, local_salience, reduce="amax", include_self=True)
    logits = ((local_salience - peak[inverse]) * 3.0).exp()
    denom = local_salience.new_zeros(groups)
    denom.index_add_(0, inverse, logits)
    weights = (logits / denom[inverse].clamp_min(1e-12))[:, None]
    node_keys = keys.new_zeros((groups, dim))
    node_keys.index_add_(0, inverse, keys * weights)
    node_keys = F.normalize(node_keys, dim=-1, eps=1e-12)
    if content_values is None:
        node_values = node_keys
    else:
        values = F.normalize(content_values[eligible].float(), dim=-1, eps=1e-12)
        node_values = values.new_zeros((groups, dim))
        node_values.index_add_(0, inverse, values * weights)
        node_values = F.normalize(node_values, dim=-1, eps=1e-12)
    return node_keys, node_values, peak


def _roles(salience: torch.Tensor, eligible: torch.Tensor, inverse: torch.Tensor, groups: int) -> torch.Tensor:
    roles = torch.zeros_like(salience, dtype=torch.float32)
    if groups == 0 or eligible.numel() == 0:
        return roles
    local = salience[eligible].float()
    peak = local.new_zeros(groups)
    floor = local.new_full((groups,), float("inf"))
    peak.scatter_reduce_(0, inverse, local, reduce="amax", include_self=True)
    floor.scatter_reduce_(0, inverse, local, reduce="amin", include_self=True)
    relative = (local - floor[inverse]) / (peak[inverse] - floor[inverse]).clamp_min(1e-6)
    roles[eligible] = (0.55 * relative + 0.45 * local).clamp(0, 1)
    return roles


def build_head_graph(
    content_keys: torch.Tensor,
    values: Optional[torch.Tensor],
    positions: torch.Tensor,
    system_tokens: int,
    config: EventGraphConfig = EventGraphConfig(),
) -> HeadGraph:
    """Build query-independent micro-groups and their K/V representations."""

    config.validate()
    if values is not None and values.shape != content_keys.shape:
        raise ValueError("values must match keys")
    eligible, inverse, group_ids = _micro_groups(
        content_keys, positions, system_tokens, config
    )
    salience = _token_salience(
        content_keys, values, eligible, inverse, group_ids, positions
    )
    node_keys, node_values, group_salience = _pool_nodes(
        content_keys, values, eligible, inverse, len(group_ids), salience
    )
    return HeadGraph(
        eligible=eligible,
        inverse=inverse,
        group_ids=group_ids,
        token_salience=salience,
        roles=_roles(salience, eligible, inverse, len(group_ids)),
        group_salience=group_salience,
        node_keys=node_keys,
        node_values=node_values,
    )


def _pool_max(values: torch.Tensor, graph: HeadGraph) -> torch.Tensor:
    out = values.new_zeros(graph.groups)
    if graph.groups:
        out.scatter_reduce_(
            0,
            graph.inverse,
            values[graph.eligible],
            reduce="amax",
            include_self=True,
        )
    return out


def _complete_bipartite_graph(
    base_scores: torch.Tensor,
    graph: HeadGraph,
    config: EventGraphConfig,
) -> None:
    groups = graph.groups
    if groups == 0:
        graph.neighbors = graph.group_ids.new_empty((0, 0))
        return
    node_base = _pool_max(base_scores.float(), graph)
    count = min(config.landmarks, groups)
    priority = (
        0.85 * node_base / node_base.max().clamp_min(1e-30)
        + 0.15 * graph.group_salience
    )
    landmark_ids = torch.topk(priority, count, sorted=True).indices
    similarity = (graph.node_keys @ graph.node_keys[landmark_ids].T).clamp_min_(0)
    distance = (graph.group_ids[:, None] - graph.group_ids[landmark_ids][None, :]).abs()
    similarity.masked_fill_(distance <= 1, 0.0)
    if graph.node_values is not None:
        value_affinity = (graph.node_values @ graph.node_values[landmark_ids].T)
        similarity.mul_((0.5 + 0.5 * value_affinity).clamp(0.2, 1.0))

    width = min(config.semantic_neighbors, max(0, count - 1))
    if width:
        edge_values, relative = similarity.topk(width, dim=-1)
        candidates = landmark_ids[relative]
        semantic = torch.where(edge_values >= config.min_similarity, candidates, -1)
    else:
        semantic = graph.group_ids.new_empty((groups, 0))

    local_columns = []
    for offset in (-1, 1):
        target = graph.group_ids + offset
        index = torch.searchsorted(graph.group_ids, target).clamp_max(groups - 1)
        local_columns.append(torch.where(graph.group_ids[index] == target, index, -1))
    local = torch.stack(local_columns, dim=1)
    graph.neighbors = torch.cat((semantic, local), dim=1)
    graph.semantic_width = width
    graph.local_gate = (
        0.5
        + 0.5
        * (graph.node_values[1:] * graph.node_values[:-1]).sum(-1).clamp(0, 1)
        if groups > 1
        else None
    )


def score_head(
    base_scores: torch.Tensor,
    graph: HeadGraph,
    config: EventGraphConfig = EventGraphConfig(),
) -> torch.Tensor:
    """Return graph-adjusted retention scores for one layer/head.

    ``base_scores`` are EpiCache's native center-patch attention scores.  System
    rows retain their original score.  This function performs no Top-K itself.
    """

    config.validate()
    if base_scores.ndim != 1 or len(base_scores) != len(graph.token_salience):
        raise ValueError("base_scores must be a vector aligned with graph tokens")
    if not torch.isfinite(base_scores).all() or bool((base_scores < 0).any()):
        raise ValueError("base_scores must be finite and non-negative")
    if graph.groups == 0:
        return base_scores.float().clone()
    _complete_bipartite_graph(base_scores, graph, config)
    node = _pool_max(base_scores.float(), graph)
    width = graph.semantic_width
    semantic_ids = graph.neighbors[:, :width]
    semantic = (
        (node[semantic_ids.clamp_min(0)] * (semantic_ids >= 0)).amax(-1)
        if width
        else torch.zeros_like(node)
    )
    chain = node + 0.5 * semantic
    adjacent = (graph.group_ids[1:] - graph.group_ids[:-1] == 1).to(node.dtype)
    if graph.local_gate is not None:
        adjacent = adjacent * graph.local_gate
    left = torch.zeros_like(node)
    right = torch.zeros_like(node)
    left[1:] = chain[:-1] * adjacent
    right[:-1] = chain[1:] * adjacent
    event = config.event_decay * torch.maximum(left, right)
    support = torch.maximum(semantic, event)

    output = base_scores.float().clone()
    source = output[graph.eligible]
    own = node[graph.inverse]
    supported = support[graph.inverse]
    native_share = torch.where(
        own > 0, source / own.clamp_min(1e-30), torch.zeros_like(source)
    )
    role = graph.roles[graph.eligible]
    own_share = torch.maximum(native_share, config.self_completion * role)
    neighbor_share = (
        (1.0 - config.neighbor_completion) * native_share
        + config.neighbor_completion * role
    )
    output[graph.eligible] = own * own_share + config.alpha * supported * neighbor_share
    return output


def native_mask(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Stable native Top-K mask used to derive the original head quotas."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    mask = torch.ones_like(scores, dtype=torch.bool)
    if len(scores) > budget:
        mask.zero_()
        if budget:
            mask[torch.topk(scores, budget).indices] = True
    return mask


def select_layer(
    base_heads: Sequence[torch.Tensor],
    proposal_heads: Sequence[torch.Tensor],
    total_budget: int,
) -> SelectionResult:
    """Select rows while preserving EpiCache's native non-uniform head quotas."""

    if len(base_heads) != len(proposal_heads) or not base_heads:
        raise ValueError("base/proposal head lists must be non-empty and aligned")
    lengths = [len(x) for x in base_heads]
    if any(len(a) != len(b) for a, b in zip(base_heads, proposal_heads)):
        raise ValueError("base/proposal lengths differ")
    base_flat = torch.cat(tuple(base_heads))
    native_flat = native_mask(base_flat, min(total_budget, len(base_flat)))
    native_masks = tuple(native_flat.split(lengths))
    selected = []
    quotas = []
    for proposal, native in zip(proposal_heads, native_masks):
        if not torch.isfinite(proposal).all():
            raise ValueError("proposal contains non-finite values")
        quota = int(native.sum())
        mask = torch.zeros_like(native)
        if quota:
            order = torch.argsort(proposal, descending=True, stable=True)
            mask[order[:quota]] = True
        if int(mask.sum()) != quota:
            raise RuntimeError("head quota changed")
        selected.append(mask)
        quotas.append(quota)
    return SelectionResult(tuple(selected), native_masks, tuple(quotas))


class BipartiteEventGraphScorer:
    """Small public API for the frozen paper method."""

    def __init__(self, config: EventGraphConfig = EventGraphConfig()):
        self.config = config.validate()

    @torch.no_grad()
    def propose(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        base_scores: torch.Tensor,
        system_tokens: int,
        *,
        rotary_metadata: Optional[Mapping[str, object]] = None,
        keys_are_content: bool = False,
    ) -> torch.Tensor:
        """Build a transient graph and return retention scores.

        Pass ``keys_are_content=True`` when K is already pre-RoPE.  Otherwise
        supply ``rotary_metadata`` so cached keys can be inverted exactly.
        No question, answer, or future-turn input is accepted by this API.
        """

        if keys_are_content:
            content_keys = keys.float()
        else:
            if rotary_metadata is None:
                raise ValueError("cached RoPE keys require rotary_metadata")
            content_keys = undo_rotary(keys, positions, rotary_metadata)
        graph = build_head_graph(
            content_keys, values.float(), positions, system_tokens, self.config
        )
        return score_head(base_scores, graph, self.config)
