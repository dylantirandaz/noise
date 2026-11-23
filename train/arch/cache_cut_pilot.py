"""Small CPU Cut experiment on disjoint rows of the fixed training split.

Run with python train/arch/cache_cut_pilot.py --training data/train.pt
"""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

# Use the reference PyTorch kernels for this single-threaded CPU experiment.
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

import run
from utils.prepare_data import EXPECTED, TRAIN_ROWS, load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("data/train.pt"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    config = run.GPTConfig(n_embd=32, n_layer=2, n_head=2, n_kv_head=2,
                           sequence_len=64, ngram_rows=16384, ngram_dim=8)
    settings = run.Settings(model=config, cut_batch=1, boost_batch=1, query_draws=16,
                            nrep=8, route_k=32, query_keys=16, chain_tokens=16,
                            chain_precision="fp32", chain_act="fp32", key_path_draws=32,
                            moment_kernel=False, fused_query=False, fused_moves=False, chain_fast=False)
    rows = load_rows(args.training, TRAIN_ROWS)[:432, :65].clone()
    backbone, calibration, audit = rows[:256], rows[264:296], rows[304:432]
    model = run.make_model(torch.device("cpu"), 0, config, settings.projection_std, settings.head_std)
    estimator = run.Cut(model, settings, 0, 0, 1)
    optimizer = run.make_optimizer(model, settings, len(backbone))
    try:
        for step, row in enumerate(backbone, start=1):
            run.set_rate(optimizer, step, len(backbone), settings.warmdown, settings.final)
            estimator.step(row[None, :-1], row[None, 1:], optimizer)
    finally:
        estimator.close()

    with torch.no_grad():
        grid = torch.tensor(run.CACHE_GRID, dtype=torch.float64)
        losses, counts = run.cache_losses(model, calibration, grid.expand(len(model.cache_weights), -1), 0, 1)
        model.cache_weights.copy_(grid[losses.argmin(dim=1)])
        weights = model.cache_weights.double()
        audit_weights = torch.stack((torch.zeros_like(weights), weights), dim=1)
        losses, audit_counts = run.cache_losses(model, audit, audit_weights, 0, 1)
        base, mixed = losses.sum(0) / audit_counts.sum()
        row_losses = []
        for row in audit:
            x, y = row[None, :-1], row[None, 1:]
            row_losses.append(torch.stack((run.neural_loss(model, x, y, "mean"), model(x, y))))
        direct = torch.stack(row_losses).double()
        torch.testing.assert_close(direct.mean(0), torch.stack((base, mixed)), atol=1e-5, rtol=0)
    print(json.dumps(dict(torch_version=str(torch.__version__), train_sha256=EXPECTED["train.pt"],
                          settings=asdict(settings), backbone_updates=len(backbone), seed=0,
                          calibration_tokens=counts.sum().item(), audit_tokens=audit_counts.sum().item(),
                          neural_loss=base.item(), mixed_loss=mixed.item(), gain=(base - mixed).item(),
                          improved_rows=(direct[:, 1] < direct[:, 0]).sum().item(), audit_rows=len(audit),
                          weights=weights.tolist()), indent=2))


if __name__ == "__main__":
    main()
