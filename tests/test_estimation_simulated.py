import numpy as np

from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data


def _estimate_map(table):
    return dict(zip(table["parameter"], table["estimate"]))


def test_simulated_random_parameter_recovery():
    data, truth = simulate_ordered_probit_data(
        n_groups=160,
        observations_per_group=3,
        fixed_betas={"x": 0.7},
        random_means={"z": -0.8},
        random_sds={"z": 0.45},
        thresholds=(-0.35, 0.85),
        seed=202406,
    )

    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=["z"],
        group_id="group",
        categories=[0, 1, 2],
        draws=96,
        draw_type="halton",
        seed=99,
        maxiter=180,
        tolerance=1e-5,
    )
    results = model.fit(data, save_run=False)
    estimates = _estimate_map(results.parameter_table)

    assert results.converged
    assert results.log_likelihood < 0.0
    assert estimates["threshold[1]"] < estimates["threshold[2]"]
    assert abs(estimates["beta_fixed[x]"] - truth["fixed_betas"]["x"]) < 0.45
    assert abs(estimates["beta_random_mean[z]"] - truth["random_means"]["z"]) < 0.25
    assert abs(estimates["beta_random_sd[z]"] - truth["random_sds"]["z"]) < 0.25
    assert np.isfinite(results.parameter_table["std_error"]).all()
