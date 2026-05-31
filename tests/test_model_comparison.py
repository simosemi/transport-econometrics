import numpy as np

from rpopit.config import parse_model_spec
from rpopit.model import _unpack_thresholds
from rpopit.model_comparison import compare_ordered_models
from rpopit.simulation import simulate_ordered_probit_data


def test_threshold_parameterization_guarantees_ordering():
    packed = np.array([-100.0, -20.0, 20.0])
    thresholds = _unpack_thresholds(packed)

    assert thresholds[0] < thresholds[1] < thresholds[2]


def test_model_comparison_report_exports_expected_tables(tmp_path):
    data, _ = simulate_ordered_probit_data(
        n_groups=35,
        observations_per_group=2,
        fixed_betas={"x": 0.4},
        random_means={"z": -0.3},
        random_sds={"z": 0.2},
        seed=111,
    )
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "severity",
                "fixed": ["x"],
                "random": {"z": {"start_mean": -0.2, "start_sd": 0.2}},
                "group_id": "group",
                "categories": [0, 1, 2],
                "missing": "drop",
            },
            "simulation": {
                "draws": 12,
                "draw_type": "halton",
                "chunk_size": 20,
                "workers": 1,
                "seed": 22,
            },
            "estimation": {"maxiter": 4, "tolerance": 1e-4},
        }
    )

    report = compare_ordered_models(data, spec)
    paths = report.export(tmp_path)

    assert set(report.metrics["model"]) == {
        "Ordered Probit",
        "Random Parameters Ordered Probit",
    }
    assert {"LL", "AIC", "BIC", "McFadden_pseudo_R2"}.issubset(report.metrics.columns)
    assert "random_sd" in set(report.random_parameter_sds["component"])
    assert paths["metrics"].exists()
    assert paths["coefficients"].exists()
    assert paths["random_sds"].exists()
