import numpy as np
import pytest

from rpnb.model import RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data


def _estimate_map(table):
    return dict(zip(table["parameter"], table["estimate"]))


def test_fixed_only_negative_binomial_matches_statsmodels():
    discrete_model = pytest.importorskip("statsmodels.discrete.discrete_model")

    data, _ = simulate_negative_binomial_data(
        n_groups=700,
        observations_per_group=1,
        intercept=-0.35,
        fixed_betas={"x1": 0.55},
        random_means={},
        alpha=0.7,
        seed=202405,
    )

    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        intercept=True,
        maxiter=400,
        tolerance=1e-7,
        covariance="hessian",
    )
    rpnb_results = model.fit(data, save_run=False)
    rpnb_estimates = _estimate_map(rpnb_results.parameter_table)

    x = data[["x1"]].copy()
    x.insert(0, "const", 1.0)
    sm_model = discrete_model.NegativeBinomial(
        data["crashes"],
        x,
        loglike_method="nb2",
        offset=data["log_exposure"],
    )
    sm_results = sm_model.fit(disp=0, maxiter=400)
    sm_params = sm_results.params

    assert np.isclose(rpnb_estimates["beta_fixed[Intercept]"], sm_params["const"], atol=1e-4)
    assert np.isclose(rpnb_estimates["beta_fixed[x1]"], sm_params["x1"], atol=1e-4)
    assert np.isclose(rpnb_estimates["alpha"], sm_params["alpha"], atol=1e-4)
    assert np.isclose(rpnb_results.log_likelihood, sm_results.llf, atol=1e-5)

    rpnb_means = rpnb_results.predictions["predicted_count"].to_numpy()
    sm_means = sm_results.predict(x, offset=data["log_exposure"])
    np.testing.assert_allclose(rpnb_means, sm_means, rtol=5e-5, atol=5e-5)
