"""Synthetic end-to-end example; no model weights or dataset required."""
import torch

from event_graph_cache import BipartiteEventGraphScorer, select_layer


def make_head(seed: int, tokens: int = 256, dim: int = 64):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(tokens, dim, generator=generator)
    values = torch.randn(tokens, dim, generator=generator)
    positions = torch.arange(tokens)
    base = torch.rand(tokens, generator=generator)
    return keys, values, positions, base


def main():
    scorer = BipartiteEventGraphScorer()
    base_heads, proposals = [], []
    for head in range(4):
        keys, values, positions, base = make_head(100 + head)
        proposal = scorer.propose(
            keys,
            values,
            positions,
            base,
            system_tokens=8,
            keys_are_content=True,
        )
        base_heads.append(base)
        proposals.append(proposal)

    result = select_layer(base_heads, proposals, total_budget=256)
    print("native head quotas:", result.head_quotas)
    print("retained rows:", sum(int(mask.sum()) for mask in result.masks))
    print(
        "changed identities:",
        sum(int((a != b).sum()) for a, b in zip(result.masks, result.native_masks)),
    )


if __name__ == "__main__":
    main()
