import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from rpopit.config import RandomParameterSpec
from rpopit.draws import generate_draws
from rpopit.likelihood import simulated_log_likelihood
from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data


def _fixed_only_data(seed=123, n=350):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    thresholds = np.array([-0.7, 0.25, 1.15])
    latent = 0.75 * x1 - 0.45 * x2 + rng.normal(size=n)
    y = np.searchsorted(thresholds, latent, side="right")
    return pd.DataFrame({"severity": y, "x1": x1, "x2": x2})


def _estimate_map(table):
    return dict(zip(table["parameter"], table["estimate"]))


def _thresholds(results):
    rows = results.parameter_table[results.parameter_table["component"] == "threshold"]
    return rows.sort_values("parameter")["estimate"].to_numpy()


def _predicted_probability_matrix(results, categories):
    columns = [f"Pr({category})" for category in categories]
    return results.predicted_probabilities[columns].to_numpy()


def test_estimated_thresholds_are_ordered_for_four_categories():
    data = _fixed_only_data()
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x1", "x2"],
        categories=[0, 1, 2, 3],
        maxiter=400,
        tolerance=1e-5,
    )
    results = model.fit(data, save_run=False)
    mu1, mu2, mu3 = _thresholds(results)

    assert mu1 < mu2 < mu3


def test_fixed_only_matches_statsmodels_ordered_model():
    data = _fixed_only_data()
    categories = [0, 1, 2, 3]
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x1", "x2"],
        categories=categories,
        maxiter=400,
        tolerance=1e-5,
    )
    results = model.fit(data, save_run=False)
    estimates = _estimate_map(results.parameter_table)

    statsmodels_model = OrderedModel(
        data["severity"],
        data[["x1", "x2"]],
        distr="probit",
    )
    statsmodels_results = statsmodels_model.fit(method="bfgs", disp=False)
    statsmodels_thresholds = statsmodels_model.transform_threshold_params(
        statsmodels_results.params
    )[1:-1]
    statsmodels_probabilities = statsmodels_model.predict(
        statsmodels_results.params,
        exog=data[["x1", "x2"]],
    )
    rpopit_probabilities = _predicted_probability_matrix(results, categories)

    assert abs(results.log_likelihood - statsmodels_results.llf) < 1e-4
    assert abs(estimates["beta_fixed[x1]"] - statsmodels_results.params["x1"]) < 1e-4
    assert abs(estimates["beta_fixed[x2]"] - statsmodels_results.params["x2"]) < 1e-4
    np.testing.assert_allclose(_thresholds(results), statsmodels_thresholds, atol=1e-4)
    np.testing.assert_allclose(
        rpopit_probabilities,
        statsmodels_probabilities,
        atol=1e-4,
    )


def test_random_parameter_near_zero_sd_matches_fixed_only_ordered_probit():
    rng = np.random.default_rng(777)
    n = 250
    x1 = rng.normal(size=n)
    z = rng.normal(size=n)
    thresholds = np.array([-0.6, 0.35, 1.1])
    latent = 0.5 * x1 - 0.7 * z + rng.normal(size=n)
    severity = np.searchsorted(thresholds, latent, side="right")
    data = pd.DataFrame(
        {"severity": severity, "x1": x1, "z": z, "case_id": np.arange(n)}
    )

    fixed_results = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x1", "z"],
        categories=[0, 1, 2, 3],
        maxiter=400,
        tolerance=1e-5,
    ).fit(data, save_run=False)
    random_results = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x1"],
        random=[RandomParameterSpec("z", start_mean=-0.7, start_sd=1e-4)],
        group_id="case_id",
        categories=[0, 1, 2, 3],
        draws=80,
        draw_type="halton",
        seed=2,
        maxiter=300,
        tolerance=1e-4,
    ).fit(data, save_run=False)

    fixed = _estimate_map(fixed_results.parameter_table)
    random = _estimate_map(random_results.parameter_table)

    assert abs(fixed_results.log_likelihood - random_results.log_likelihood) < 1e-3
    assert abs(fixed["beta_fixed[x1]"] - random["beta_fixed[x1]"]) < 1e-3
    assert abs(fixed["beta_fixed[z]"] - random["beta_random_mean[z]"]) < 1e-3
    np.testing.assert_allclose(
        _thresholds(fixed_results), _thresholds(random_results), atol=1e-3
    )
    assert random["beta_random_sd[z]"] < 1e-3


def test_increasing_halton_draws_stabilizes_random_parameter_log_likelihood():
    data, _ = simulate_ordered_probit_data(
        n_groups=70,
        observations_per_group=2,
        fixed_betas={"x": 0.6},
        random_means={"z": -0.5},
        random_sds={"z": 0.35},
        thresholds=(-0.4, 0.6),
        seed=912,
    )
    group_values = data["group"].to_numpy()
    groups = [np.flatnonzero(group_values == group) for group in sorted(data["group"].unique())]

    def log_likelihood_with_draw_count(n_draws):
        draws = generate_draws(len(groups), n_draws, 1, "halton", seed=13)
        return simulated_log_likelihood(
            beta_fixed=np.array([0.6]),
            random_means=np.array([-0.5]),
            random_sds=np.array([0.35]),
            thresholds=np.array([-0.4, 0.6]),
            x_fixed=data[["x"]].to_numpy(),
            x_random=data[["z"]].to_numpy(),
            y_codes=data["severity"].to_numpy(),
            group_indices=groups,
            draws=draws,
        )

    ll_50 = log_likelihood_with_draw_count(50)
    ll_100 = log_likelihood_with_draw_count(100)
    ll_500 = log_likelihood_with_draw_count(500)
    ll_reference = log_likelihood_with_draw_count(2000)

    assert abs(ll_100 - ll_reference) < abs(ll_50 - ll_reference)
    assert abs(ll_500 - ll_reference) < abs(ll_100 - ll_reference)
    assert abs(ll_500 - ll_reference) < 0.01
