"""Measure the Track 3 cache on a four-gram control using only the fixed training split.

Run after utils/prepare_data.py with python train/arch/cache_proxy.py --training data/train.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import CACHE_BINS, cache_matches
from run import CACHE_GRID
from utils.prepare_data import TRAIN_ROWS, load_rows


VOCAB = 4096


def pack(rows, order):
    """Row-local full n-gram IDs in exactly 12 bits per token."""
    length = rows.shape[1] - order + 1
    keys = rows[:, :length].astype(np.int64)
    for offset in range(1, order):
        keys <<= 12
        keys |= rows[:, offset:offset + length]
    return keys


def lookup(keys, sorted_ids, values):
    ix = np.minimum(np.searchsorted(sorted_ids, keys), len(sorted_ids) - 1)
    return np.where(sorted_ids[ix] == keys, values[ix], 0)


class CountLM:
    """Interpolated Witten-Bell, orders 2 through 4 over an add-half unigram."""

    def __init__(self, rows):
        unigram_counts = np.bincount(rows[:, 1:].ravel(), minlength=VOCAB)
        self.unigram = (unigram_counts + .5) / (unigram_counts.sum() + VOCAB * .5)
        self.tables = []
        for order in (2, 3, 4):
            events, freqs = np.unique(pack(rows, order).ravel(), return_counts=True)
            contexts, first, distinct = np.unique(events >> 12, return_index=True, return_counts=True)
            totals = np.add.reduceat(freqs, first)
            self.tables.append((events, freqs, contexts, distinct, totals))

    def probabilities(self, rows):
        probabilities = self.unigram[rows[:, 1:]].copy()
        for order, (events, freqs, contexts, distinct, totals) in enumerate(self.tables, 2):
            event = pack(rows, order)
            hits = lookup(event, events, freqs)
            d = lookup(event >> 12, contexts, distinct)
            n = lookup(event >> 12, contexts, totals)
            prior = probabilities[:, order - 2:]
            prior[:] = np.where(n > 0, (hits + d * prior) / np.maximum(n + d, 1), prior)
        return probabilities


def observations(rows, lm):
    base, cache, bins = [], [], []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        idx = torch.tensor(batch[:, :-1], dtype=torch.long)
        targets = torch.tensor(batch[:, 1:], dtype=torch.long)
        cache_p = torch.empty_like(targets, dtype=torch.float64)
        bin_ids = torch.empty_like(targets)
        for sl, matches, support, bin_id in cache_matches(idx):
            hits = (matches & (idx[:, None, :] == targets[:, sl, None])).sum(-1)
            assert torch.all(support >= 1)
            cache_p[:, sl] = hits.double() / support
            bin_ids[:, sl] = bin_id
        base.append(lm.probabilities(batch))
        cache.append(cache_p.numpy())
        bins.append(bin_ids.numpy())
    return np.concatenate(base), np.concatenate(cache), np.concatenate(bins)


def fit(base, cache, bins):
    scores = []
    for w in CACHE_GRID:
        nll = -np.log((1 - w) * base + w * cache)
        scores.append(np.bincount(bins.ravel(), weights=nll.ravel(), minlength=CACHE_BINS))
    return np.asarray(CACHE_GRID)[np.argmin(scores, axis=0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("data/train.pt"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    rows = load_rows(args.training, TRAIN_ROWS)[:4880].numpy()
    # Leave eight complete rows between each split in corpus order.
    fit_rows, cal_rows, test_rows = rows[:4544], rows[4552:4616], rows[4624:4880]
    lm = CountLM(fit_rows)
    w = fit(*observations(cal_rows, lm))
    base, cache, bins = observations(test_rows, lm)
    base_nll = -np.log(base)
    mixture_nll = -np.log((1 - w[bins]) * base + w[bins] * cache)
    paired = mixture_nll - base_nll
    # 32 adjacent blocks of eight sequences, preserving within-block dependence.
    paired_blocks = paired.reshape(32, -1).mean(axis=1)
    rng = np.random.default_rng(0)
    draws = rng.choice(paired_blocks, size=(10000, 32), replace=True).mean(axis=1)
    print(json.dumps({
        "rows_fit": len(fit_rows), "rows_calibration": len(cal_rows),
        "rows_proxy_test": len(test_rows), "tokens_proxy_test": test_rows[:, 1:].size,
        "baseline": base_nll.mean(), "cache_mixture": mixture_nll.mean(),
        "paired_delta": paired.mean(), "paired_block_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
        "weights": w.tolist(), "counts_by_bin": np.bincount(bins.ravel(), minlength=CACHE_BINS).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
