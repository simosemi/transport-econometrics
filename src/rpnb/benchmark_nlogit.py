"""Create a simulated RPNB versus NLOGIT benchmark package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rpnb.config import parse_model_spec
from rpnb.model import RandomParametersNegativeBinomial
from rpnb.nlogit_comparison import (
    compare_with_nlogit,
    write_nlogit_prediction_template,
    write_nlogit_template,
)
from rpnb.simulation import simulate_negative_binomial_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Benchmark output directory.")
    parser.add_argument(
        "--nlogit-results",
        default=None,
        help="Optional NLOGIT parameter CSV in component,variable,estimate format.",
    )
    parser.add_argument(
        "--nlogit-predictions",
        default=None,
        help="Optional NLOGIT predicted means CSV with row_index,predicted_mean.",
    )
    parser.add_argument("--groups", type=int, default=200)
    parser.add_argument("--observations-per-group", type=int, default=3)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=202406)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, truth = simulate_negative_binomial_data(
        n_groups=args.groups,
        observations_per_group=args.observations_per_group,
        intercept=-0.45,
        fixed_betas={"x1": 0.40},
        random_means={"z1": -0.55},
        random_sds={"z1": 0.35},
        alpha=0.60,
        seed=args.seed,
    )
    data_path = out_dir / "simulated_rpnb_nlogit_benchmark_data.csv"
    data.to_csv(data_path, index=False)
    _write_truth(truth, out_dir / "simulated_truth.csv")

    spec_dict = _benchmark_spec(args.draws, out_dir)
    spec_path = out_dir / "rpnb_benchmark_model.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec_dict, handle, sort_keys=False)

    model = RandomParametersNegativeBinomial.from_spec(parse_model_spec(spec_dict))
    results = model.fit(data, save_run=True, output_dir=out_dir / "rpnb_runs", export=True)

    template_path = write_nlogit_template(
        out_dir / "nlogit_results_template.csv",
        fixed_variables=["x1"],
        random_variables=["z1"],
        offset_variable="log_exposure",
        include_intercept=True,
    )
    prediction_template_path = write_nlogit_prediction_template(
        out_dir / "nlogit_predicted_means_template.csv",
        results.predictions["row_index"],
    )
    instructions_path = _write_nlogit_instructions(
        out_dir,
        data_path,
        spec_path,
        template_path,
        prediction_template_path,
        args.draws,
    )

    print("Simulated benchmark data:", data_path)
    print("RPNB model spec:", spec_path)
    print("RPNB run directory:", results.run_dir)
    print("NLOGIT parameter template:", template_path)
    print("NLOGIT predicted means template:", prediction_template_path)
    print("NLOGIT instructions:", instructions_path)

    if args.nlogit_results:
        report = compare_with_nlogit(
            results,
            args.nlogit_results,
            nlogit_predictions_path=args.nlogit_predictions,
            tolerance=args.tolerance,
            prediction_tolerance=args.prediction_tolerance,
            metadata={
                "dataset": str(data_path),
                "rpnb_run_dir": str(results.run_dir),
                "nlogit_results": str(args.nlogit_results),
                "nlogit_predictions": args.nlogit_predictions,
                "draws": args.draws,
                "seed": args.seed,
                "offset_variable": "log_exposure",
                "offset_coefficient": 1.0,
            },
        )
        paths = report.export(out_dir / "comparison_report")
        print("Comparison report:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print(
            "No NLOGIT results were supplied. Fill nlogit_results_template.csv "
            "and optionally nlogit_predicted_means_template.csv from the NLOGIT "
            "run, then rerun with --nlogit-results and --nlogit-predictions."
        )
    return 0


def _benchmark_spec(draws: int, out_dir: Path) -> dict[str, object]:
    return {
        "model": {
            "dependent": "crashes",
            "offset": "log_exposure",
            "fixed": ["x1"],
            "random": {
                "z1": {
                    "distribution": "normal",
                    "start_mean": -0.2,
                    "start_sd": 0.25,
                }
            },
            "group_id": "group",
            "intercept": True,
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
            "start_alpha": 0.5,
        },
        "output": {"directory": str(out_dir / "rpnb_runs")},
    }


def _write_truth(truth: dict[str, object], path: Path) -> None:
    rows = [{"component": "coefficient", "variable": "Intercept", "truth": truth["intercept"]}]
    for name, value in truth["fixed_betas"].items():
        rows.append({"component": "coefficient", "variable": name, "truth": value})
    for name, value in truth["random_means"].items():
        rows.append({"component": "random_mean", "variable": name, "truth": value})
    for name, value in truth["random_sds"].items():
        rows.append({"component": "random_sd", "variable": name, "truth": value})
    rows.append({"component": "alpha", "variable": "alpha", "truth": truth["alpha"]})
    rows.append(
        {
            "component": "offset_coefficient",
            "variable": "log_exposure",
            "truth": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_nlogit_instructions(
    out_dir: Path,
    data_path: Path,
    spec_path: Path,
    template_path: Path,
    prediction_template_path: Path,
    draws: int,
) -> Path:
    path = out_dir / "nlogit_run_instructions.md"
    path.write_text(
        "\n".join(
            [
                "# NLOGIT RPNB Benchmark Instructions",
                "",
                "Use the same simulated CSV created by this benchmark:",
                "",
                f"`{data_path}`",
                "",
                "Benchmark model:",
                "",
                "- Dependent count variable: `crashes`.",
                "- Log-offset variable: `log_exposure`.",
                "- Fixed variables: `Intercept`, `x1`.",
                "- Normally distributed random parameter: `z1`.",
                "- Panel/group ID: `group`.",
                "- NB2 dispersion: report `alpha` where `Var(y)=mu+alpha*mu^2`.",
                f"- Simulation draws: `{draws}` Halton draws.",
                "",
                "RPNB model specification:",
                "",
                f"`{spec_path}`",
                "",
                "Critical offset requirement:",
                "",
                "`log_exposure` must enter the NLOGIT log-mean with coefficient fixed",
                "at 1. Do not estimate `log_exposure` as an ordinary RHS variable.",
                "",
                "NLOGIT command sketch to adapt to your local NLOGIT syntax/version:",
                "",
                "```text",
                "READ; FILE = simulated_rpnb_nlogit_benchmark_data.csv$",
                "",
                "NEGBIN ; Lhs = crashes",
                "       ; Rhs = ONE, x1, z1",
                "       ; RPL = z1",
                "       ; PDS = group",
                "       ; Halton",
                f"       ; Pts = {draws}",
                "       ; Offset = log_exposure",
                "       ; Parameters$",
                "",
                "CREATE ; nlogit_mu = fitted means from the final model$",
                "```",
                "",
                "If your NLOGIT build uses a different keyword for count-model offsets",
                "or exposure, use the equivalent option that fixes the coefficient on",
                "`log_exposure` to exactly 1 on the log-link scale.",
                "",
                "After the NLOGIT run, fill this parameter template:",
                "",
                f"`{template_path}`",
                "",
                "Expected parameter rows:",
                "",
                "```csv",
                "component,variable,estimate,std_error",
                "coefficient,Intercept,...,...",
                "coefficient,x1,...,...",
                "random_mean,z1,...,...",
                "random_sd,z1,...,...",
                "alpha,alpha,...,...",
                "offset_coefficient,log_exposure,1,",
                "log_likelihood,LL,...,",
                "```",
                "",
                "Fill this predicted means template with NLOGIT fitted means:",
                "",
                f"`{prediction_template_path}`",
                "",
                "Then rerun:",
                "",
                "```bash",
                "python -m rpnb.benchmark_nlogit "
                f"--out {out_dir} "
                f"--nlogit-results {template_path} "
                f"--nlogit-predictions {prediction_template_path}",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
