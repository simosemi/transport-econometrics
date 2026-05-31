import numpy as np

from rpnb.model import RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data


def _estimate_map(table):
    return dict(zip(table["parameter"], table["estimate"]))


def test_simulated_random_parameter_recovery():
    data, truth = simulate_negative_binomial_data(
        n_groups=140,
        observations_per_group=3,
        intercept=-0.45,
        fixed_betas={"x1": 0.4},
        random_means={"z1": -0.55},
        random_sds={"z1": 0.35},
        alpha=0.55,
        seed=202406,
    )

    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=96,
        draw_type="halton",
        seed=99,
        maxiter=220,
        tolerance=1e-5,
    )
    results = model.fit(data, save_run=False)
    estimates = _estimate_map(results.parameter_table)

    assert results.log_likelihood < 0.0
    assert abs(estimates["beta_fixed[Intercept]"] - truth["intercept"]) < 0.45
    assert abs(estimates["beta_fixed[x1]"] - truth["fixed_betas"]["x1"]) < 0.25
    assert abs(estimates["beta_random_mean[z1]"] - truth["random_means"]["z1"]) < 0.35
    assert abs(estimates["beta_random_sd[z1]"] - truth["random_sds"]["z1"]) < 0.35
    assert abs(estimates["alpha"] - truth["alpha"]) < 0.45
    assert np.isfinite(results.parameter_table["std_error"]).all()
    assert (results.predictions["predicted_count"] > 0).all()
    assert set(results.marginal_effects["variable"]) == {"x1", "z1"}
