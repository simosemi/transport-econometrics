import pandas as pd
import yaml

from rpnb.benchmark_nlogit import main as benchmark_nlogit_main
from rpnb.model import RandomParametersNegativeBinomial
from rpnb.nlogit_comparison import (
    compare_with_nlogit,
    load_nlogit_canonical_table,
    load_nlogit_predicted_means,
    rpnb_canonical_table,
)
from rpnb.simulation import simulate_negative_binomial_data


def test_load_nlogit_canonical_table_accepts_count_model_aliases(tmp_path):
    path = tmp_path / "nlogit.csv"
    pd.DataFrame(
        [
            {"component": "coef", "variable": "const", "estimate": -0.2},
            {"component": "coef", "variable": "x1", "estimate": 0.4},
            {"component": "mean", "variable": "z1", "estimate": -0.5},
            {"component": "sd", "variable": "z1", "estimate": 0.3},
            {"component": "dispersion", "variable": "dispersion", "estimate": 0.6},
            {"component": "offset", "variable": "log_exposure", "estimate": 1.0},
            {"component": "loglikelihood", "variable": "LL", "estimate": -123.4},
        ]
    ).to_csv(path, index=False)

    table = load_nlogit_canonical_table(path)

    assert set(table["component"]) == {
        "coefficient",
        "random_mean",
        "random_sd",
        "alpha",
        "offset_coefficient",
        "log_likelihood",
    }
    assert "Intercept" in set(table["variable"])
    assert "alpha" in set(table["variable"])


def test_load_nlogit_predicted_means_accepts_row_template_format(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"row_index": 10, "predicted_mean": 0.7},
            {"row_index": 11, "predicted_mean": 1.2},
        ]
    ).to_csv(path, index=False)

    table = load_nlogit_predicted_means(path)

    assert list(table["component"]) == ["predicted_mean", "predicted_mean"]
    assert list(table["variable"]) == ["10", "11"]
    assert list(table["estimate"]) == [0.7, 1.2]


def test_compare_with_nlogit_exports_count_model_tables(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=45,
        observations_per_group=2,
        intercept=-0.35,
        fixed_betas={"x1": 0.4},
        random_means={"z1": -0.45},
        random_sds={"z1": 0.25},
        alpha=0.5,
        seed=321,
    )
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=16,
        draw_type="halton",
        maxiter=8,
        tolerance=1e-4,
    )
    results = model.fit(data, save_run=False)
    nlogit_table = rpnb_canonical_table(results)
    nlogit_params = nlogit_table.loc[nlogit_table["component"] != "predicted_mean"].copy()
    nlogit_params.loc[nlogit_params["component"] == "coefficient", "estimate"] += 0.01
    nlogit_path = tmp_path / "nlogit.csv"
    nlogit_params.to_csv(nlogit_path, index=False)

    nlogit_predictions = results.predictions.loc[:, ["row_index", "predicted_count"]].rename(
        columns={"predicted_count": "predicted_mean"}
    )
    nlogit_predictions_path = tmp_path / "nlogit_predictions.csv"
    nlogit_predictions.to_csv(nlogit_predictions_path, index=False)

    report = compare_with_nlogit(results, nlogit_path, nlogit_predictions_path)
    paths = report.export(tmp_path / "report")

    assert not report.coefficients.empty
    assert not report.random_means.empty
    assert not report.random_sds.empty
    assert not report.alpha.empty
    assert not report.offset_handling.empty
    assert not report.predicted_means.empty
    assert not report.log_likelihood.empty
    assert report.coefficients["abs_difference"].max() > 0.0
    assert report.predicted_means["abs_difference"].max() < 1e-12
    assert paths["html"].exists()


def test_benchmark_nlogit_generates_offset_templates(tmp_path):
    output = tmp_path / "benchmark"
    status = benchmark_nlogit_main(
        [
            "--out",
            str(output),
            "--groups",
            "20",
            "--observations-per-group",
            "2",
            "--draws",
            "8",
        ]
    )

    data = pd.read_csv(output / "simulated_rpnb_nlogit_benchmark_data.csv")
    template = pd.read_csv(output / "nlogit_results_template.csv")
    prediction_template = pd.read_csv(output / "nlogit_predicted_means_template.csv")
    with (output / "rpnb_benchmark_model.yaml").open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    assert status == 0
    assert {"crashes", "log_exposure", "x1", "z1", "group"}.issubset(data.columns)
    assert spec["model"]["offset"] == "log_exposure"
    assert "alpha" in set(template["component"])
    assert "offset_coefficient" in set(template["component"])
    assert len(prediction_template) == len(data)
    assert (output / "nlogit_run_instructions.md").exists()
