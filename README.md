# Anonymous Review Code: TailMem

This anonymous supplement contains the minimal implementation of **Ours** used in the submitted paper.
It implements the query-independent KV-retention scorer used during EpiCache
block prefill.  The release deliberately excludes exploratory V1--V16 variants,
large result files, model weights, generated caches, and paper-only analysis
scripts.

The implementation accepts only cached K/V states, original token positions,
and the native center-patch retention score.  Its API has no question or answer
argument.  It therefore preserves the paper setting in which future user queries
are unknown when historical KV caches are compressed.

## What is included

```text
paper_release/
├── src/event_graph_cache/
│   ├── __init__.py
│   └── core.py                 # complete method and native-quota selection
├── examples/minimal_example.py # synthetic example, no model download
├── tests/test_core.py           # deterministic CPU tests
├── tools/check_research_equivalence.py
│                                # bit-exact check against frozen research code
├── scripts/smoke_test.sh        # tests plus minimal example
├── scripts/run_frozen_eval.sh   # full runner entry inside modified EpiCache
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

The central public class is:

```python
from event_graph_cache import BipartiteEventGraphScorer
```

## Method in one page

For each Transformer layer and KV head, the method performs the following steps.

1. Exclude system-prompt rows and group surviving historical positions into
   contiguous four-token micro-groups.
2. Compute a query-independent token salience signal from global K isolation,
   local K contrast, V isolation, and V energy.  Tie-aware ranks use fixed-shape
   `sort + searchsorted` operations.
3. Pool each micro-group into K-side and V-side node representations using the
   salience weights.
4. Select at most 96 patch-guided landmarks.  A landmark priority combines the
   native center-patch score and query-independent group salience:

   ```text
   priority(g) = 0.85 * normalized_native_score(g)
               + 0.15 * structural_salience(g)
   ```

5. Connect every micro-group to at most eight non-local landmarks using positive
   K similarity gated by V affinity.  Add the two valid event-chain neighbors
   at original group positions `g-1` and `g+1`.
6. Propagate the strongest semantic or local event support and distribute it
   within each micro-group using token roles.  This changes retention scores,
   not attention logits.
7. Apply Top-K within each KV head while preserving the exact non-uniform head
   quotas produced by native EpiCache.

The temporary graph is discarded after the retention mask is constructed.  The
final cache capacity, causal attention, decoding path, and four-episode routing
remain unchanged.

## Installation

The minimal scorer requires Python 3.10+ and PyTorch 2.3+.

```bash
cd paper_release
python -m pip install -e .
```

For the full EpiCache experiments, use the environment documented by the parent
repository: CUDA 12.1, Python 3.10, PyTorch 2.3, Transformers 4.51, and
FlashAttention 2.7.4.post1.

## Quick start

No GPU, dataset, model, or network access is needed for the smoke test:

```bash
bash scripts/smoke_test.sh
```

A direct call for one layer and one KV head is:

```python
import torch
from event_graph_cache import BipartiteEventGraphScorer

T, D = 2048, 128
keys = torch.randn(T, D)          # pre-RoPE content keys in this example
values = torch.randn(T, D)
positions = torch.arange(T)
base_scores = torch.rand(T)       # native center-patch retention scores

scorer = BipartiteEventGraphScorer()
proposal = scorer.propose(
    keys,
    values,
    positions,
    base_scores,
    system_tokens=32,
    keys_are_content=True,
)
```

Real cached keys are normally RoPE-rotated.  In that case, leave
`keys_are_content=False` and pass the model's fixed rotary metadata:

```python
proposal = scorer.propose(
    cached_keys,
    cached_values,
    absolute_positions,
    base_scores,
    system_tokens,
    rotary_metadata={
        "inv_freq": model.model.rotary_emb.inv_freq,
        "attention_scaling": model.model.rotary_emb.attention_scaling,
        "rope_type": model.model.rotary_emb.rope_type,
    },
)
```

The released inversion supports full-dimension `default` and `llama3` RoPE.
Models with partial or position-dependent rotary frequencies require an
architecture-specific content-key hook.

## Preserving native head allocation

EpiCache allocates a layer budget non-uniformly across KV heads.  Graph scoring
must change token identities without silently changing those head quotas:

```python
from event_graph_cache import select_layer

result = select_layer(
    base_heads=[base_h0, base_h1, ...],
    proposal_heads=[proposal_h0, proposal_h1, ...],
    total_budget=layer_budget,
)
# result.masks: graph-selected masks
# result.native_masks: native EpiCache masks
# result.head_quotas: identical per-head capacities for both methods
```

Use each Boolean mask to prune K and V at the same original positions.

## Block-prefill integration

The scorer is called after every historical block has produced its native patch
attention scores and before physical KV eviction:

```text
historical block prefill
  -> native center-patch attention score for every candidate KV row
  -> recover content K and build a transient bipartite event graph
  -> compute graph-adjusted retention proposal per layer/head
  -> preserve native cross-head quotas and select rows
  -> prune K and V with the same Boolean mask
  -> continue block prefill
```

Do not feed the final user question, reference answer, evidence annotation, or
precomputed query score into compression.  The graph must be rebuilt from only
the history-side tensors available at that block.

## Frozen implementation

The submitted implementation is frozen for review.  The standalone tests cover
tie-aware salience, original-position grouping, query-independent scoring, and
exact preservation of native non-uniform head quotas.  Exploratory variants,
internal experiment logs, development reports, and repository history are not
part of the anonymous supplement.

## Reproducing full benchmark runs

Dataset preprocessing and episode manifests follow the parent EpiCache code.
Once a frozen manifest exists, run the paper method with:

```bash
cd paper_release
EPICACHE_ROOT=/path/to/ml-epicache \
GPU=0 \
./scripts/run_frozen_eval.sh \
  llama \
  locomo \
  '/path/to/manifests/history_*.json' \
  /path/to/results/llama_locomo_ours.json
```

Supported dataset labels in the frozen runner are `locomo`, `realtalk`, and
`longmemeval`.  Supported model labels are `qwen`, `qwen25_3b`, `qwen3`, and
`llama`.  Use the same manifests for EpiCache and Ours; change only the method
from `center` to the frozen registry name above.  LoCoMo and LongMemEval use the
repository's token F1/EM evaluation.  RealTalk generation can be produced
locally, but the submission score should be computed with the official
GPT-based judge.

## Reproducibility rules

- Freeze model revision, tokenizer, episode patches, routing decisions, layer
  profile, KV budget, and generation configuration across methods.
- Use greedy generation and reset the cache exactly between questions.
- Report 2K/4K/8K as target budgets and retain the actual per-layer/per-head
  capacities in run metadata.
- Do not compare the reconstructed LongMemEval-20K subset with results from the
  official 100K/500-question protocol without an explicit protocol label.
- Keep diagnostic token F1 on RealTalk separate from the official GPT-judge
  score.

## Scope

This release isolates the paper's compression contribution.  It does not ship
model weights, benchmark data, BookSum profiles, generated answers, external
judge credentials, or the exploratory method sweep.  Those artifacts are not
required to inspect or test the scoring algorithm.

## License and acknowledgement

The code is released under the MIT License.  The full experimental system is
built on EpiCache and KVzip; retain their copyright and citation notices when
redistributing the integrated runner.
