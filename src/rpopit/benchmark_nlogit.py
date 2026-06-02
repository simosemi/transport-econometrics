"""Create a simulated rpopit versus NLOGIT benchmark package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rpopit.config import parse_model_spec
from rpopit.model import RandomParametersOrderedProbit
from rpopit.nlogit_comparison import compare_with_nlogit, write_nlogit_template
from rpopit.simulation import simulate_ordered_probit_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Benchmark output directory.")
    parser.add_argument(
        "--nlogit-results",
        default=None,
        help="Optional NLOGIT CSV in component,variable,estimate format.",
    )
    parser.add_argument("--groups", type=int, default=200)
    parser.add_argument("--observations-per-group", type=int, default=3)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202406)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--categories",
        type=int,
        default=3,
        help="Number of ordered severity categories to simulate. Use 4 for [0,1,2,3].",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds_for_categories(args.categories)
    categories = list(range(args.categories))

    data, truth = simulate_ordered_probit_data(
        n_groups=args.groups,
        observations_per_group=args.observations_per_group,
        fixed_betas={"x": 0.7},
        random_means={"z": -0.8},
        random_sds={"z": 0.45},
        thresholds=thresholds,
        seed=args.seed,
    )
    data_path = out_dir / "simulated_nlogit_benchmark_data.csv"
    data.to_csv(data_path, index=False)
    _write_truth(truth, out_dir / "simulated_truth.csv")

    spec_dict = _benchmark_spec(args.draws, out_dir, categories)
    spec_path = out_dir / "rpopit_benchmark_model.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec_dict, handle, sort_keys=False)

    model = RandomParametersOrderedProbit.from_spec(parse_model_spec(spec_dict))
    results = model.fit(data, save_run=True, output_dir=out_dir / "rpopit_runs", export=True)

    template_path = write_nlogit_template(
        out_dir / "nlogit_results_template.csv",
        fixed_variables=["x"],
        random_variables=["z"],
        n_thresholds=args.categories - 1,
    )
    instructions_path = _write_nlogit_instructions(
        out_dir,
        data_path,
        spec_path,
        template_path,
        categories,
        args.categories - 1,
    )

    print("Simulated benchmark data:", data_path)
    print("rpopit model spec:", spec_path)
    print("rpopit run directory:", results.run_dir)
    print("NLOGIT results template:", template_path)
    print("NLOGIT instructions:", instructions_path)

    if args.nlogit_results:
        report = compare_with_nlogit(
            results,
            args.nlogit_results,
            tolerance=args.tolerance,
            metadata={
                "dataset": str(data_path),
                "rpopit_run_dir": str(results.run_dir),
                "nlogit_results": str(args.nlogit_results),
                "draws": args.draws,
                "seed": args.seed,
                "categories": args.categories,
                "thresholds": args.categories - 1,
            },
        )
        paths = report.export(out_dir / "comparison_report")
        print("Comparison report:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print(
            "No NLOGIT results were supplied. Fill nlogit_results_template.csv "
            "from the NLOGIT run, then rerun with --nlogit-results."
        )
    return 0


def _benchmark_spec(draws: int, out_dir: Path, categories: list[int]) -> dict[str, object]:
    return {
        "model": {
            "dependent": "severity",
            "fixed": ["x"],
            "random": {
                "z": {
                    "distribution": "normal",
                    "start_mean": -0.2,
                    "start_sd": 0.3,
                }
            },
            "group_id": "group",
            "categories": categories,
            "correlated_random_parameters": False,
            "missing": "drop",
        },
        "simulation": {
            "draws": draws,
            "draw_type": "halton",
            "chunk_size": 10000,
            "workers": 1,
            "seed": 99,
        },
        "estimation": {
            "maxiter": 1000,
            "tolerance": 0.0001,
            "covariance": "bfgs",
            "checkpoint_interval": 10,
        },
        "output": {"directory": str(out_dir / "rpopit_runs")},
    }


def _write_truth(truth: dict[str, object], path: Path) -> None:
    rows = []
    for name, value in truth["fixed_betas"].items():
        rows.append({"component": "coefficient", "variable": name, "truth": value})
    for name, value in truth["random_means"].items():
        rows.append({"component": "random_mean", "variable": name, "truth": value})
    for name, value in truth["random_sds"].items():
        rows.append({"component": "random_sd", "variable": name, "truth": value})
    for index, value in enumerate(truth["thresholds"], start=1):
        rows.append(
            {
                "component": "threshold",
                "variable": f"threshold[{index}]",
                "truth": value,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_nlogit_instructions(
    out_dir: Path,
    data_path: Path,
    spec_path: Path,
    template_path: Path,
    categories: list[int],
    n_thresholds: int,
) -> Path:
    path = out_dir / "nlogit_run_instructions.md"
    path.write_text(
        "\n".join(
            [
                "# NLOGIT Benchmark Instructions",
                "",
                "Use the same simulated CSV created by this benchmark:",
                "",
                f"`{data_path}`",
                "",
                "Fit an ordered probit model with one fixed coefficient `x`, one normally",
                "distributed random coefficient `z`, grouped by `group`, and two thresholds",
                f"for severity categories `{categories}`.",
                "",
                "Use the same simulation draw count and Halton draws specified in:",
                "",
                f"`{spec_path}`",
                "",
                "After the NLOGIT run, fill this CSV template with NLOGIT estimates:",
                "",
                f"`{template_path}`",
                "",
                "Expected columns:",
                "",
                "```text",
                "component,variable,estimate,std_error",
                "coefficient,x,...,...",
                "random_mean,z,...,...",
                "random_sd,z,...,...",
                *[
                    f"threshold,threshold[{threshold_number}],...,..."
                    for threshold_number in range(1, n_thresholds + 1)
                ],
                "log_likelihood,LL,...,",
                "```",
                "",
                "Then rerun:",
                "",
                "```bash",
                "python -m rpopit.benchmark_nlogit "
                f"--out {out_dir} --nlogit-results {template_path}",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _thresholds_for_categories(categories: int) -> tuple[float, ...]:
    if categories < 2:
        raise ValueError("--categories must be at least 2.")
    if categories == 3:
        return (-0.35, 0.85)
    if categories == 4:
        return (-0.75, 0.15, 1.05)
    return tuple(float(value) for value in np.linspace(-1.0, 1.0, categories - 1))


if __name__ == "__main__":
    raise SystemExit(main())
