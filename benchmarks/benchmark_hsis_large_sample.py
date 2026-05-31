"""Benchmark likelihood evaluation on a real HSIS-style CSV or sampled subset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rpopit.config import load_model_spec
from rpopit.data_validation import load_schema, validate_dataframe
from rpopit.draws import generate_draws
from rpopit.likelihood import simulated_log_likelihood
from rpopit.model import RandomParametersOrderedProbit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to real crash severity CSV.")
    parser.add_argument("--spec", required=True, help="Path to rpopit YAML model spec.")
    parser.add_argument("--schema", default=None, help="Optional YAML or CSV schema.")
    parser.add_argument("--out", required=True, help="Output CSV for benchmark results.")
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[100_000, 500_000, 1_000_000],
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = load_model_spec(args.spec)
    data = pd.read_csv(args.data)
    if args.schema:
        validation = validate_dataframe(data, load_schema(args.schema), missing=spec.missing)
        if not validation.valid:
            raise SystemExit("Data validation failed; run validation_suite/validate_hsis_data.py.")
        data = validation.cleaned_data

    rows = []
    for sample_size in args.sample_sizes:
        sample = _sample_data(data, sample_size, args.seed)
        seconds, ll = _time_one_likelihood(sample, spec)
        rows.append(
            {
                "observations": len(sample),
                "draws": spec.draws,
                "random_parameters": len(spec.random),
                "chunk_size": spec.chunk_size,
                "workers": spec.workers,
                "log_likelihood": ll,
                "one_likelihood_eval_seconds": seconds,
                "estimated_100_eval_seconds": seconds * 100.0,
            }
        )
        print(rows[-1])

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print("Benchmark written to:", output)
    return 0


def _sample_data(data: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    replace = n_rows > len(data)
    return data.sample(n=n_rows, replace=replace, random_state=seed).reset_index(drop=True)


def _time_one_likelihood(data: pd.DataFrame, spec) -> tuple[float, float]:
    model = RandomParametersOrderedProbit.from_spec(spec)
    logger = _NullLogger()
    work = model._prepare_frame(data, logger)
    y_codes, _ = model._encode_dependent(work[model.dependent])
    x_fixed = work.loc[:, model.fixed].astype(float).to_numpy() if model.fixed else np.zeros((len(work), 0))
    x_random = work.loc[:, model.random].astype(float).to_numpy() if model.random else np.zeros((len(work), 0))
    group_codes, group_labels, group_indices, order, starts, counts = model._group_indices(
        work, len(model.random) > 0
    )
    del group_codes, group_labels
    draws = generate_draws(len(group_indices), model.draws, len(model.random), model.draw_type, model.seed)
    theta = model._start_params(y_codes, len(model.categories or sorted(work[model.dependent].unique())))
    state = model._unpack_params(theta, len(model.categories or sorted(work[model.dependent].unique())) - 1)

    start = time.perf_counter()
    ll = simulated_log_likelihood(
        beta_fixed=state.fixed,
        random_means=state.random_means,
        random_sds=state.random_sds,
        cholesky=state.cholesky,
        thresholds=state.thresholds,
        x_fixed=x_fixed[order],
        x_random=x_random[order],
        y_codes=y_codes[order],
        group_indices=group_indices,
        group_starts=starts,
        group_counts=counts,
        draws=draws,
        chunk_size=model.chunk_size,
        workers=model.workers,
    )
    return time.perf_counter() - start, ll


class _NullLogger:
    def info(self, *args, **kwargs):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
