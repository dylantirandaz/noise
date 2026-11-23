"""CPU regression tests for the Track 3 continuation cache.

Run with python -m unittest discover -s train/arch -p test_cache.py
"""
import unittest
from unittest.mock import patch

import torch

import model
from utils import eval as evaluator


def prefix_distribution(prefix, vocab_size):
    """Enumerate continuations using only the visible prefix, independently of the batched code."""
    for order in range(min(4, len(prefix)), 0, -1):
        suffix = prefix[-order:]
        tokens = [prefix[i] for i in range(order, len(prefix)) if prefix[i - order:i] == suffix]
        if tokens:
            support_bin = int(len(tokens) >= 2) + int(len(tokens) >= 4)
            bin_id = 1 + 3 * (order - 1) + support_bin
            break
    else:
        tokens, bin_id = prefix, 0
    counts = torch.bincount(torch.tensor(tokens), minlength=vocab_size)
    return counts / len(tokens), bin_id


class ContinuationCacheTest(unittest.TestCase):
    def test_matches_independent_prefix_enumeration(self):
        generator = torch.Generator().manual_seed(31)
        tokens = torch.randint(0, 5, (3, 147), generator=generator)
        tokens[0] = torch.arange(tokens.size(1)) % 4
        tokens[1, 80:110] = tokens[1, 14:44]
        seen = set()
        for query_slice, matches, support, bins in model.cache_matches(tokens):
            for row, token_row in enumerate(tokens.tolist()):
                for column, q in enumerate(range(query_slice.start, query_slice.stop)):
                    actual = torch.bincount(tokens[row, matches[row, column]], minlength=5)
                    expected, bin_id = prefix_distribution(token_row[:q + 1], 5)
                    seen.add(bin_id)
                    self.assertEqual(actual.sum(), support[row, column])
                    self.assertEqual(bin_id, bins[row, column].item())
                    torch.testing.assert_close(actual / support[row, column], expected)
        self.assertEqual(seen, set(range(model.CACHE_BINS)))

    @torch.no_grad()
    def test_predictions_loss_causality_and_reload(self):
        config = model.GPTConfig(sequence_len=16, vocab_size=8, n_layer=2, n_head=2,
                                 n_kv_head=2, n_embd=32, ngram_rows=32, ngram_dim=8)
        torch.manual_seed(7)
        net = model.GPT(config).eval()
        net.init_weights()
        ids = torch.tensor([[2, 1, 2, 4, 2, 1, 2, 3], [7, 5, 5, 7, 5, 5, 7, 0]])
        labels = torch.tensor([[1, 2, 4, 2, 1, 2, 3, -1], [5, 5, 7, -1, 5, 7, 0, 3]])
        # Exercise the actual model on CPU without requiring a local Inductor compiler.
        with patch.object(model, 'flex', model.flex_attention):
            neural = net.neural_logits(ids).softmax(-1)
            torch.testing.assert_close(net(ids).softmax(-1), neural)
            net.cache_weights.copy_(torch.linspace(.03, .91, model.CACHE_BINS))
            expected = torch.empty_like(neural)
            for row, token_row in enumerate(ids.tolist()):
                for q in range(len(token_row)):
                    cache, bin_id = prefix_distribution(token_row[:q + 1], config.vocab_size)
                    w = net.cache_weights[bin_id]
                    expected[row, q] = (1 - w) * neural[row, q] + w * cache
            log_probs = net(ids)
            torch.testing.assert_close(log_probs.exp(), expected)
            torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones_like(ids, dtype=torch.float))
            nll = -expected.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1).log()
            nll = nll.masked_fill(labels == -1, 0)
            torch.testing.assert_close(net(ids, labels, 'none'), nll.flatten())
            torch.testing.assert_close(net(ids, labels, 'sum'), nll.sum())
            torch.testing.assert_close(net(ids, labels, 'mean'), nll.sum() / (labels != -1).sum())
            changed = ids.clone()
            changed[0, 5:] = 6
            changed[1] = 3
            torch.testing.assert_close(net(changed)[0, :5], log_probs[0, :5], atol=0, rtol=0)
            torch.testing.assert_close(net(ids[:1, :1])[0, 0], log_probs[0, 0])
            # Run the evaluator's meta-device construction, initialization and strict state-dict load.
            with patch.object(evaluator, 'model_classes', return_value=(model.GPT, lambda: config)):
                restored = evaluator.load_model(net.state_dict(), device='cpu')
            torch.testing.assert_close(restored(ids), log_probs, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
