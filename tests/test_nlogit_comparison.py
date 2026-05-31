import pandas as pd
import yaml

from rpopit.benchmark_nlogit import main as benchmark_nlogit_main
from rpopit.nlogit_comparison import (
    compare_with_nlogit,
    load_nlogit_canonical_table,
    rpopit_canonical_table,
)
from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data


def test_load_nlogit_canonical_table_accepts_aliases(tmp_path):
    path = tmp_path / "nlogit.csv"
    pd.DataFrame(
        [
            {"component": "coef", "variable": "x", "estimate": 0.1},
            {"component": "cutpoint", "variable": "mu1", "estimate": -0.2},
            {"component": "mean", "variable": "z", "estimate": -0.3},
            {"component": "sd", "variable": "z", "estimate": 0.4},
            {"component": "loglikelihood", "variable": "LL", "estimate": -12.5},
        ]
    ).to_csv(path, index=False)

    table = load_nlogit_canonical_table(path)

    assert set(table["component"]) == {
        "coefficient",
        "threshold",
        "random_mean",
        "random_sd",
        "log_likelihood",
    }
    assert "threshold[1]" in set(table["variable"])


def test_compare_with_nlogit_exports_difference_tables(tmp_path):
    data, _ = simulate_ordered_probit_data(
        n_groups=45,
        observations_per_group=2,
        fixed_betas={"x": 0.5},
        random_means={"z": -0.4},
        random_sds={"z": 0.25},
        seed=321,
    )
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=["z"],
        group_id="group",
        categories=[0, 1, 2],
        draws=16,
        draw_type="halton",
        maxiter=8,
        tolerance=1e-4,
    )
    results = model.fit(data, save_run=False)
    nlogit_table = rpopit_canonical_table(results)
    nlogit_table.loc[nlogit_table["component"] == "coefficient", "estimate"] += 0.01
    nlogit_path = tmp_path / "nlogit.csv"
    nlogit_table.to_csv(nlogit_path, index=False)

    report = compare_with_nlogit(results, nlogit_path)
    paths = report.export(tmp_path / "report")

    assert not report.coefficients.empty
    assert not report.thresholds.empty
    assert not report.random_means.empty
    assert not report.random_sds.empty
    assert not report.log_likelihood.empty
    assert report.combined["abs_difference"].max() > 0.0
    assert paths["html"].exists()


def test_benchmark_nlogit_categories_four_generates_three_thresholds(tmp_path):
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
            "--categories",
            "4",
        ]
    )

    data = pd.read_csv(output / "simulated_nlogit_benchmark_data.csv")
    template = pd.read_csv(output / "nlogit_results_template.csv")
    with (output / "rpopit_benchmark_model.yaml").open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    assert status == 0
    assert set(data["severity"].unique()).issubset({0, 1, 2, 3})
    assert spec["model"]["categories"] == [0, 1, 2, 3]
    assert (template["component"] == "threshold").sum() == 3
    assert "threshold[3]" in set(template["variable"])
