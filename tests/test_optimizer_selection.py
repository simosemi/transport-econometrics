import numpy as np
import pandas as pd
import pytest

from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data
from rpnb.compare_runs import compare_runs, main as compare_runs_main
from rpnb.audit_nlogit import main as audit_nlogit_main
from rpnb.model import RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data


def test_rpnb_bfgs_and_lbfgsb_produce_similar_estimates():
    data, _ = simulate_negative_binomial_data(
        n_groups=80,
        observations_per_group=2,
        intercept=-0.6,
        fixed_betas={"x1": 0.35},
        random_means={},
        alpha=0.45,
        seed=202610,
    )

    bfgs = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        optimizer="bfgs",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)
    lbfgsb = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        optimizer="lbfgsb",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)

    assert bfgs.convergence["optimizer_method"] == "bfgs"
    assert lbfgsb.convergence["optimizer_method"] == "lbfgsb"
    _assert_common_estimates_close(bfgs.parameter_table, lbfgsb.parameter_table)


def test_rpopit_bfgs_and_lbfgsb_produce_similar_estimates():
    data, _ = simulate_ordered_probit_data(
        n_groups=80,
        observations_per_group=2,
        fixed_betas={"x": 0.65},
        random_means={},
        random_sds={},
        thresholds=(-0.35, 0.75),
        seed=202611,
    )

    bfgs = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=[],
        categories=[0, 1, 2],
        optimizer="bfgs",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)
    lbfgsb = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=[],
        categories=[0, 1, 2],
        optimizer="lbfgsb",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)

    assert bfgs.convergence["optimizer_method"] == "bfgs"
    assert lbfgsb.convergence["optimizer_method"] == "lbfgsb"
    _assert_common_estimates_close(bfgs.parameter_table, lbfgsb.parameter_table)


def test_unsupported_optimizer_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        RandomParametersNegativeBinomial(
            dependent="crashes",
            offset="log_exposure",
            optimizer="newton-cg",
        )


def test_rpnb_multistart_exports_summary_and_local_solutions(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=48,
        observations_per_group=2,
        fixed_betas={"x1": 0.25},
        random_means={},
        alpha=0.5,
        seed=202612,
    )
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        multistart=3,
        multistart_random_seed=123,
        multistart_scale=0.15,
        maxiter=60,
        tolerance=1e-5,
        checkpoint_interval=0,
    )

    results = model.fit(data, save_run=True, output_dir=tmp_path, export=True)

    _assert_multistart_outputs(results, parameter_name="alpha")
    assert (results.run_dir / "multistart_summary.csv").exists()
    assert (results.run_dir / "multistart_summary.xlsx").exists()
    assert (results.run_dir / "multistart_summary.html").exists()
    assert (results.run_dir / "multistart_local_solutions.csv").exists()
    assert (results.run_dir / "random_parameter_tests.csv").exists()
    assert (results.run_dir / "random_parameter_tests.xlsx").exists()
    assert (results.run_dir / "random_parameter_tests.html").exists()
    assert (results.run_dir / "random_parameter_screening.csv").exists()
    assert (results.run_dir / "nlogit_style_report.txt").exists()

    exported_summary = pd.read_csv(results.run_dir / "multistart_summary.csv")
    exported_summary_xlsx = pd.read_excel(results.run_dir / "multistart_summary.xlsx")
    exported_solutions = pd.read_csv(results.run_dir / "multistart_local_solutions.csv")
    assert len(exported_summary) == 3
    assert len(exported_summary_xlsx) == 3
    assert set(exported_solutions["start_id"]) == {1, 2, 3}
    assert "multiple_local_optima_found" in results.fit_statistics
    assert "n_local_optima" in results.fit_statistics
    assert "random_parameter_lr_tests_executed" in results.fit_statistics
    report_text = (results.run_dir / "nlogit_style_report.txt").read_text(encoding="utf-8")
    assert "RPNB NLOGIT-STYLE REPORT" in report_text
    assert "RANDOM PARAMETER MEANS" in report_text
    assert "DISPERSION PARAMETER" in report_text


def test_rpopit_multistart_exports_summary_and_local_solutions(tmp_path):
    data, _ = simulate_ordered_probit_data(
        n_groups=48,
        observations_per_group=2,
        fixed_betas={"x": 0.5},
        random_means={},
        random_sds={},
        thresholds=(-0.3, 0.7),
        seed=202613,
    )
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=[],
        categories=[0, 1, 2],
        multistart=3,
        multistart_random_seed=123,
        multistart_scale=0.15,
        maxiter=60,
        tolerance=1e-5,
        checkpoint_interval=0,
    )

    results = model.fit(data, save_run=True, output_dir=tmp_path, export=True)

    _assert_multistart_outputs(results, parameter_name="threshold[1]")
    assert (results.run_dir / "multistart_summary.csv").exists()
    assert (results.run_dir / "multistart_summary.xlsx").exists()
    assert (results.run_dir / "multistart_summary.html").exists()
    assert (results.run_dir / "multistart_local_solutions.csv").exists()
    assert (results.run_dir / "nlogit_style_report.txt").exists()

    exported_summary = pd.read_csv(results.run_dir / "multistart_summary.csv")
    exported_summary_xlsx = pd.read_excel(results.run_dir / "multistart_summary.xlsx")
    exported_solutions = pd.read_csv(results.run_dir / "multistart_local_solutions.csv")
    assert len(exported_summary) == 3
    assert len(exported_summary_xlsx) == 3
    assert set(exported_solutions["start_id"]) == {1, 2, 3}
    report_text = (results.run_dir / "nlogit_style_report.txt").read_text(encoding="utf-8")
    assert "RPOPIT NLOGIT-STYLE REPORT" in report_text
    assert "RANDOM PARAMETER SCALE/SD" in report_text
    assert "Not applicable for ordered probit" in report_text


def test_rpnb_compare_runs_exports_report(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=32,
        observations_per_group=2,
        fixed_betas={"x1": 0.25},
        random_means={"z1": -0.2},
        random_sds={"z1": 0.25},
        alpha=0.5,
        seed=202614,
    )
    run_dirs = []
    for seed in [11, 22]:
        model = RandomParametersNegativeBinomial(
            dependent="crashes",
            offset="log_exposure",
            fixed=["x1"],
            random=["z1"],
            group_id="group",
            draws=8,
            seed=seed,
            maxiter=8,
            tolerance=1e-4,
            checkpoint_interval=0,
        )
        results = model.fit(
            data,
            save_run=True,
            output_dir=tmp_path / "runs",
            export=True,
        )
        run_dirs.append(results.run_dir)

    report = compare_runs(run_dirs)
    paths = report.export(tmp_path / "comparison_report")

    assert len(report.metrics) == 2
    assert {"LL", "AIC", "BIC", "alpha", "convergence_quality", "n_parameters"}.issubset(
        report.metrics.columns
    )
    assert "beta_random_mean[z1]" in set(report.random_parameter_means["parameter"])
    assert "beta_random_sd[z1]" in set(report.random_parameter_sds["parameter"])
    assert paths["metrics_csv"].exists()
    assert paths["excel"].exists()
    assert paths["html"].exists()
    assert "model_rank" in report.metrics.columns
    assert "random_sd_significance" in report.metrics.columns
    assert "n_lr_keep_random" in report.metrics.columns
    assert "n_lr_treat_fixed" in report.metrics.columns
    assert "n_screen_keep_random" in report.metrics.columns
    assert "n_screen_convert_to_fixed" in report.metrics.columns
    assert report.metrics["model_rank"].min() == 1
    assert report.metrics["n_lr_keep_random"].notna().all()
    assert report.metrics["n_lr_treat_fixed"].notna().all()
    assert (
        report.metrics["n_lr_keep_random"] + report.metrics["n_lr_treat_fixed"]
    ).ge(1).all()
    assert (
        report.metrics["n_screen_keep_random"]
        + report.metrics["n_screen_convert_to_fixed"]
    ).ge(1).all()

    cli_out = tmp_path / "comparison_report_cli"
    status = compare_runs_main(
        [
            "--runs",
            *(str(path) for path in run_dirs),
            "--out",
            str(cli_out),
        ]
    )
    assert status == 0
    assert (cli_out / "comparison_metrics.csv").exists()
    assert (run_dirs[0] / "random_parameter_tests.csv").exists()
    assert (run_dirs[0] / "random_parameter_tests.xlsx").exists()
    assert (run_dirs[0] / "random_parameter_tests.html").exists()
    assert (run_dirs[0] / "random_parameter_screening.csv").exists()
    random_tests = pd.read_csv(run_dirs[0] / "random_parameter_tests.csv")
    assert {
        "parameter",
        "unrestricted_log_likelihood",
        "restricted_log_likelihood",
        "lr_statistic",
        "p_value",
        "restricted_converged",
        "restricted_status",
        "recommendation",
    }.issubset(random_tests.columns)
    assert random_tests["restricted_log_likelihood"].notna().all()
    assert set(random_tests["recommendation"]).issubset({"Keep Random", "Treat as Fixed"})
    screening = pd.read_csv(run_dirs[0] / "random_parameter_screening.csv")
    assert {
        "parameter",
        "mean_estimate",
        "sd_estimate",
        "sd_p_value",
        "sd_effectively_zero",
        "sd_not_statistically_significant",
        "recommendation",
    }.issubset(screening.columns)
    assert set(screening["recommendation"]).issubset({"Keep Random", "Convert to Fixed"})


def test_rpnb_auto_simplify_random_parameters_exports_both_models(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=24,
        observations_per_group=2,
        fixed_betas={"x1": 0.2},
        random_means={"z1": 0.0},
        random_sds={"z1": 0.02},
        alpha=0.4,
        seed=202615,
    )
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=6,
        seed=33,
        maxiter=6,
        tolerance=1e-4,
        checkpoint_interval=0,
        auto_simplify_random_parameters=True,
    )

    results = model.fit(data, save_run=True, output_dir=tmp_path, export=True)

    assert results.auto_simplify_summary is not None
    assert (results.run_dir / "auto_simplify_summary.csv").exists()
    assert (results.run_dir / "auto_simplified_model").exists()
    assert (results.run_dir / "auto_simplified_model" / "coefficients.csv").exists()
    summary = pd.read_csv(results.run_dir / "auto_simplify_summary.csv")
    assert {"full_random", "simplified_fixed"}.issubset(set(summary["model"]))
    assert summary["selected"].isin([True, False]).all()


def test_rpnb_audit_nlogit_writes_markdown_report(tmp_path):
    out_path = tmp_path / "nlogit_audit_report.md"

    status = audit_nlogit_main(["--out", str(out_path)])

    assert status == 0
    text = out_path.read_text(encoding="utf-8")
    assert "Likelihood Formulation" in text
    assert "Random Categorical Variables" in text
    assert "Offset Handling" in text
    assert "Panel Likelihood" in text


def _assert_common_estimates_close(left, right, atol: float = 2e-3) -> None:
    left_estimates = left.set_index("parameter")["estimate"].sort_index()
    right_estimates = right.set_index("parameter")["estimate"].sort_index()
    common = left_estimates.index.intersection(right_estimates.index)

    assert len(common) == len(left_estimates) == len(right_estimates)
    np.testing.assert_allclose(
        left_estimates.loc[common].to_numpy(),
        right_estimates.loc[common].to_numpy(),
        atol=atol,
        rtol=atol,
    )


def _assert_multistart_outputs(results, parameter_name: str) -> None:
    summary = results.multistart_summary
    local_solutions = results.local_solutions
    assert summary is not None
    assert local_solutions is not None
    assert len(summary) == 3
    assert summary["optimizer"].tolist() == ["bfgs", "bfgs", "bfgs"]
    assert summary["starting_log_likelihood"].notna().all()
    assert summary["final_log_likelihood"].notna().all()
    assert summary["AIC"].notna().all()
    assert summary["BIC"].notna().all()
    assert summary["gradient_norm"].notna().all()
    assert summary["starting_parameter_vector"].notna().all()
    assert summary["final_parameter_vector"].notna().all()
    assert set(summary["convergence_quality"]).issubset(
        {"converged_clean", "near_converged", "usable_warning", "not_converged"}
    )
    assert results.convergence["convergence_quality"] in {
        "converged_clean",
        "near_converged",
        "usable_warning",
        "not_converged",
    }
    assert summary["is_best"].sum() == 1
    assert results.fit_statistics["multistart_completed"] == 3
    assert results.convergence["selected_start_id"] == int(
        summary.loc[summary["is_best"], "start_id"].iloc[0]
    )
    assert results.log_likelihood == pytest.approx(summary["final_log_likelihood"].max())
    assert parameter_name in set(local_solutions["parameter"])
