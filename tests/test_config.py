import numpy as np
import pandas as pd
import pytest

from rpopit.config import parse_model_spec
from rpopit.model import RandomParametersOrderedProbit


def test_parse_random_parameter_mapping_list():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "severity",
                "fixed": ["x"],
                "random": [
                    {"name": "z", "distribution": "normal", "start_mean": -0.2, "start_sd": 0.5}
                ],
            }
        }
    )
    assert spec.dependent == "severity"
    assert spec.random[0].name == "z"
    assert spec.random[0].start_sd == 0.5


def test_parse_generic_ordered_model_spec():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "severity",
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
                        }
                    }
                },
                "group_id": "UniqueID",
                "categories": [0, 1, 2],
            }
        }
    )

    assert spec.fixed == ("speed_mean", "Log_Hourly_volume")
    assert [item.name for item in spec.fixed_categorical] == ["Hour", "Year"]
    assert [item.reference for item in spec.fixed_categorical] == [0, 2017]
    assert spec.random[0].name == "speed_std"


def test_parse_generic_ordered_spec_rejects_duplicate_roles():
    with pytest.raises(ValueError, match="multiple roles"):
        parse_model_spec(
            {
                "model": {
                    "dependent": "severity",
                    "fixed": {"continuous": ["speed_mean"]},
                    "random": {"continuous": {"speed_mean": {"start_sd": 0.3}}},
                }
            }
        )


def test_ordered_generic_categorical_dummies_drop_reference_and_do_not_modify_raw_data():
    data = _generic_fake_ordered_data()
    original = data.copy(deep=True)
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["speed_mean", "Log_Hourly_volume"],
        fixed_categorical=[
            {"Hour": {"reference": 0}},
            {"Year": {"reference": 2017}},
        ],
        random=["speed_std"],
        group_id="UniqueID",
        categories=[0, 1, 2],
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


def test_ordered_generic_categorical_reference_must_exist():
    data = _generic_fake_ordered_data()
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["speed_mean"],
        fixed_categorical=[{"Hour": {"reference": 99}}],
        random=[],
        categories=[0, 1, 2],
        maxiter=1,
    )

    with pytest.raises(ValueError, match="Reference category"):
        model.fit(data, save_run=False)


def _generic_fake_ordered_data(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(202607)
    speed_mean = rng.normal(55.0, 5.0, size=n)
    log_hourly_volume = rng.normal(6.5, 0.35, size=n)
    speed_std = rng.normal(6.0, 1.0, size=n)
    hour = np.resize(np.array([2, 0, 1]), n)
    year = np.resize(np.array([2018, 2017]), n)
    group = np.repeat(np.arange(n // 3), 3)
    latent = (
        0.015 * speed_mean
        + 0.04 * log_hourly_volume
        - 0.03 * speed_std
        + 0.12 * (hour == 1)
        - 0.08 * (hour == 2)
        + 0.06 * (year == 2018)
        + rng.normal(size=n)
    )
    severity = np.digitize(latent, [-0.25, 0.9])
    return pd.DataFrame(
        {
            "severity": severity,
            "speed_mean": speed_mean,
            "Log_Hourly_volume": log_hourly_volume,
            "speed_std": speed_std,
            "Hour": hour,
            "Year": year,
            "UniqueID": group,
        }
    )
