from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rpnb.config import load_model_spec, parse_model_spec
from rpnb.model import RandomParametersNegativeBinomial


def test_parse_model_spec_accepts_offset_and_random_parameters():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "crashes",
                "offset": "log_exposure",
                "fixed": ["x1"],
                "random": {"z1": {"start_mean": -0.1, "start_sd": 0.4}},
                "group_id": "segment",
                "correlated_random_parameters": True,
            },
            "simulation": {"draws": 64, "draw_type": "sobol"},
            "estimation": {
                "start_alpha": 0.7,
                "optimizer": "lbfgsb",
                "multistart": 3,
                "random_seed": 987,
            },
        }
    )

    assert spec.dependent == "crashes"
    assert spec.offset == "log_exposure"
    assert spec.fixed == ("x1",)
    assert spec.random[0].name == "z1"
    assert spec.random[0].start_sd == 0.4
    assert spec.correlated_random_parameters
    assert spec.start_alpha == 0.7
    assert spec.optimizer == "lbfgsb"
    assert spec.multistart == 3
    assert spec.multistart_random_seed == 987


def test_parse_model_spec_requires_offset():
    with pytest.raises(ValueError, match="offset"):
        parse_model_spec({"model": {"dependent": "crashes"}})


def test_example_yaml_loads():
    spec = load_model_spec(Path("examples/rpnb_model.yaml"))

    assert spec.dependent == "crashes"
    assert spec.offset == "log_exposure"
    assert spec.draws == 128


def test_parse_generic_model_spec_with_categorical_and_random_continuous():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "crashes",
                "offset": "log_exposure",
                "fixed": {
                    "continuous": ["speed_mean", "Log_Hourly_volume"],
                    "categorical": {
                        "Hour": {"reference": 0},
                        "Year": {"reference": 2017},
                    },
                },
                "random": {
                    "continuous": {
                        "speed_std": {
                            "distribution": "normal",
                            "start_mean": 0.0,
                            "start_sd": 0.3,
                        },
                        "TruckPercent": {
                            "distribution": "normal",
                            "start_mean": 0.0,
                            "start_sd": 0.4,
                        },
                    }
                },
                "group_id": "UniqueID",
            }
        }
    )

    assert spec.fixed == ("speed_mean", "Log_Hourly_volume")
    assert [item.name for item in spec.fixed_categorical] == ["Hour", "Year"]
    assert [item.reference for item in spec.fixed_categorical] == [0, 2017]
    assert [item.name for item in spec.random] == ["speed_std", "TruckPercent"]
    assert spec.random[1].start_sd == 0.4


def test_parse_generic_spec_rejects_duplicate_roles():
    with pytest.raises(ValueError, match="multiple roles"):
        parse_model_spec(
            {
                "model": {
                    "dependent": "crashes",
                    "offset": "log_exposure",
                    "fixed": {"continuous": ["speed_mean"]},
                    "random": {"continuous": {"speed_mean": {"start_sd": 0.3}}},
                }
            }
        )


def test_generic_categorical_dummies_drop_reference_and_do_not_modify_raw_data():
    data = _generic_fake_count_data()
    original = data.copy(deep=True)
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["speed_mean", "Log_Hourly_volume"],
        fixed_categorical=[
            {"Hour": {"reference": 0}},
            {"Year": {"reference": 2017}},
        ],
        random=["speed_std"],
        group_id="UniqueID",
        draws=8,
        maxiter=2,
        checkpoint_interval=0,
    )

    results = model.fit(data, save_run=False)
    parameters = set(results.parameter_table["parameter"])

    assert "beta_fixed[Hour_0]" not in parameters
    assert "beta_fixed[Year_2017]" not in parameters
    assert "beta_fixed[Hour_1]" in parameters
    assert "beta_fixed[Hour_2]" in parameters
    assert "beta_fixed[Year_2018]" in parameters
    assert "beta_random_mean[speed_std]" in parameters
    pd.testing.assert_frame_equal(data, original)


def test_generic_categorical_reference_must_exist():
    data = _generic_fake_count_data()
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["speed_mean"],
        fixed_categorical=[{"Hour": {"reference": 99}}],
        random=[],
        maxiter=1,
    )

    with pytest.raises(ValueError, match="Reference category"):
        model.fit(data, save_run=False)


def test_missing_drop_removes_nan_blank_and_infinite_values_and_reports_sample(tmp_path):
    data = pd.DataFrame(
        {
            "crashes": [0, 1, 2, 1, 0, 3, 1, 2],
            "log_exposure": [0.0, 0.1, "", 0.2, 0.3, 0.4, 0.2, 0.1],
            "x": [0.2, 0.4, 0.1, np.nan, 0.3, np.inf, -0.1, 0.0],
            "Hour": [0, 1, 2, 0, 1, 2, 0, 1],
            "group": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x"],
        fixed_categorical=[{"Hour": {"reference": 0}}],
        group_id="group",
        random=[],
        maxiter=1,
        checkpoint_interval=0,
        missing="drop",
    )

    results = model.fit(data, save_run=True, output_dir=tmp_path, export=True)
    stats = results.fit_statistics

    assert stats["missing_checked_columns"] == "crashes,log_exposure,x,Hour,group"
    assert stats["n_rows_original"] == 8
    assert stats["n_rows_removed_missing"] == 3
    assert stats["n_rows_final_estimation_sample"] == 5
    assert stats["n_observations"] == 5
    assert set(results.predictions["row_index"]) == {0, 1, 4, 6, 7}

    summary = results.preprocessing_summary
    assert summary is not None
    x_summary = summary.loc[
        (summary["section"] == "variable_summary")
        & (summary["variable_name"] == "x")
    ].iloc[0]
    assert x_summary["number_missing"] == 2
    assert x_summary["number_unique_values"] == 5
    assert x_summary["mean"] == pytest.approx(np.mean([0.2, 0.4, 0.3, -0.1, 0.0]))
    assert x_summary["minimum"] == pytest.approx(-0.1)
    assert x_summary["maximum"] == pytest.approx(0.4)

    hour_summary = summary.loc[
        (summary["section"] == "variable_summary")
        & (summary["variable_name"] == "Hour")
    ].iloc[0]
    assert hour_summary["number_missing"] == 0
    assert hour_summary["number_unique_values"] == 2
    assert hour_summary["reference_category"] == "0"
    assert hour_summary["generated_dummy_variables"] == "Hour_1"

    hour_frequency = summary.loc[
        (summary["section"] == "categorical_frequency")
        & (summary["variable_name"] == "Hour")
    ]
    hour_counts = {
        str(row.category_value): int(row.category_count)
        for row in hour_frequency.itertuples()
    }
    assert hour_counts == {"0": 2, "1": 3}

    exported = pd.read_csv(results.run_dir / "fit_statistics.csv")
    assert exported.loc[0, "n_rows_original"] == 8
    assert exported.loc[0, "n_rows_removed_missing"] == 3
    assert exported.loc[0, "n_rows_final_estimation_sample"] == 5

    assert (results.run_dir / "preprocessing_summary.csv").exists()
    assert (results.run_dir / "preprocessing_summary.xlsx").exists()
    assert (results.run_dir / "preprocessing_summary.html").exists()
    exported_summary = pd.read_csv(results.run_dir / "preprocessing_summary.csv")
    assert "standard_deviation" in exported_summary.columns
    assert "generated_dummy_variables" in exported_summary.columns


def test_missing_error_raises_for_nan_blank_and_infinite_values():
    data = pd.DataFrame(
        {
            "crashes": [0, 1, 2],
            "log_exposure": [0.0, "", 0.2],
            "x": [0.2, np.inf, 0.1],
        }
    )
    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x"],
        random=[],
        maxiter=1,
        missing="error",
    )

    with pytest.raises(ValueError, match="missing or non-finite"):
        model.fit(data, save_run=False)


def _generic_fake_count_data(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(202606)
    speed_mean = rng.normal(55.0, 5.0, size=n)
    log_hourly_volume = rng.normal(6.5, 0.35, size=n)
    speed_std = rng.normal(6.0, 1.0, size=n)
    hour = np.resize(np.array([2, 0, 1]), n)
    year = np.resize(np.array([2018, 2017]), n)
    group = np.repeat(np.arange(n // 3), 3)
    exposure = rng.uniform(0.5, 2.0, size=n)
    eta = (
        np.log(exposure)
        - 2.2
        + 0.012 * speed_mean
        + 0.05 * log_hourly_volume
        - 0.02 * speed_std
        + 0.15 * (hour == 1)
        - 0.10 * (hour == 2)
        + 0.08 * (year == 2018)
    )
    mu = np.exp(eta)
    crashes = rng.poisson(mu)
    return pd.DataFrame(
        {
            "crashes": crashes,
            "log_exposure": np.log(exposure),
            "speed_mean": speed_mean,
            "Log_Hourly_volume": log_hourly_volume,
            "speed_std": speed_std,
            "Hour": hour,
            "Year": year,
            "UniqueID": group,
        }
    )
