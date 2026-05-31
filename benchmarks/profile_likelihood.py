"""Profile one simulated likelihood evaluation on synthetic data."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from rpopit.draws import generate_draws
from rpopit.likelihood import simulated_log_likelihood
from rpopit.simulation import simulate_ordered_probit_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=10_000)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    return parser


def run_once(args: argparse.Namespace) -> float:
    n_groups = max(args.observations // args.group_size, 1)
    data, _ = simulate_ordered_probit_data(
        n_groups=n_groups,
        observations_per_group=args.group_size,
        fixed_betas={"x": 0.6},
        random_means={"z": -0.5},
        random_sds={"z": 0.35},
        thresholds=(-0.4, 0.6),
        seed=123,
    )
    order = np.arange(len(data), dtype=int)
    starts = np.arange(0, len(data), args.group_size, dtype=int)
    counts = np.full(n_groups, args.group_size, dtype=int)
    draws = generate_draws(n_groups, args.draws, 1, "halton", seed=99)

    start = time.perf_counter()
    value = simulated_log_likelihood(
        beta_fixed=np.array([0.6]),
        random_means=np.array([-0.5]),
        random_sds=np.array([0.35]),
        thresholds=np.array([-0.4, 0.6]),
        x_fixed=data[["x"]].to_numpy()[order],
        x_random=data[["z"]].to_numpy()[order],
        y_codes=data["severity"].to_numpy()[order],
        group_indices=[],
        group_starts=starts,
        group_counts=counts,
        draws=draws,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - start
    print(
        f"observations={len(data)} groups={n_groups} draws={args.draws} "
        f"chunk_size={args.chunk_size} workers={args.workers} "
        f"log_likelihood={value:.6f} seconds={elapsed:.6f}"
    )
    return elapsed


def main() -> int:
    args = build_parser().parse_args()
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_once(args)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(15)
    else:
        run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
