import unittest

import torch

from event_graph_cache import (
    BipartiteEventGraphScorer,
    build_head_graph,
    score_head,
    select_layer,
    tie_aware_percentile,
)


class EventGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261021)
        torch.set_num_threads(1)

    def fixture(self, tokens=129, dim=16):
        keys = torch.randn(tokens, dim)
        values = torch.randn(tokens, dim)
        positions = torch.arange(tokens)
        base = torch.rand(tokens)
        return keys, values, positions, base

    def test_tie_aware_percentile(self):
        values = torch.tensor([
            [1.0, 1.0, 2.0, 2.0, 9.0, 9.0, 9.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        ])
        got = tie_aware_percentile(values)
        expected = torch.tensor([
            [0.0, 0.0, 4.0 / 9.0, 4.0 / 9.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_score_is_query_independent_and_preserves_system_rows(self):
        keys, values, positions, base = self.fixture()
        scorer = BipartiteEventGraphScorer()
        first = scorer.propose(
            keys, values, positions, base, system_tokens=7, keys_are_content=True
        )
        second = scorer.propose(
            keys, values, positions, base, system_tokens=7, keys_are_content=True
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[:7], base[:7], rtol=0, atol=0)
        self.assertEqual(first.shape, base.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue((first >= 0).all())

    def test_graph_uses_original_position_groups(self):
        keys, values, _, base = self.fixture(tokens=12)
        positions = torch.tensor([0, 1, 2, 3, 4, 7, 8, 18, 19, 20, 35, 39])
        graph = build_head_graph(keys, values, positions, system_tokens=3)
        expected = torch.unique((positions[positions >= 3] - 3) // 4)
        torch.testing.assert_close(graph.group_ids, expected)
        proposal = score_head(base, graph)
        self.assertEqual(proposal.shape, base.shape)

    def test_native_nonuniform_head_quotas_are_preserved(self):
        bases = [torch.rand(n) for n in (31, 47, 59, 73)]
        proposals = [torch.rand_like(x) for x in bases]
        result = select_layer(bases, proposals, total_budget=91)
        self.assertEqual(sum(result.head_quotas), 91)
        self.assertEqual(
            tuple(int(mask.sum()) for mask in result.masks), result.head_quotas
        )
        self.assertEqual(
            tuple(int(mask.sum()) for mask in result.native_masks), result.head_quotas
        )

    def test_cached_keys_require_rotary_metadata(self):
        keys, values, positions, base = self.fixture()
        with self.assertRaisesRegex(ValueError, "rotary_metadata"):
            BipartiteEventGraphScorer().propose(
                keys, values, positions, base, system_tokens=7
            )


if __name__ == "__main__":
    unittest.main()
