import numpy as np

from rpopit.likelihood import (
    ordered_probit_probabilities,
    simulated_log_likelihood,
    stable_logsumexp,
)


def test_ordered_probit_probabilities_sum_to_one():
    eta = np.array([-3.0, -0.2, 0.5, 3.0])
    thresholds = np.array([-0.5, 1.1])
    probabilities = ordered_probit_probabilities(eta, thresholds)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(eta.size))
    assert (probabilities > 0.0).all()


def test_ordered_probit_probabilities_are_valid_for_extreme_eta():
    eta = np.linspace(-12.0, 12.0, 101)
    thresholds = np.array([-1.5, -0.2, 0.9, 2.0])
    probabilities = ordered_probit_probabilities(eta, thresholds)

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(eta.size), atol=1e-12)


def test_stable_logsumexp_matches_direct_when_safe():
    values = np.array([-2.0, -1.0, -3.0])
    expected = np.log(np.exp(values).sum())
    assert np.isclose(stable_logsumexp(values), expected)


def test_simulated_log_likelihood_is_finite_for_panel_data():
    x_fixed = np.array([[0.2], [1.0], [-0.5], [0.1]])
    x_random = np.array([[1.0], [0.4], [-0.2], [0.3]])
    y = np.array([1, 2, 0, 1])
    groups = [np.array([0, 1]), np.array([2, 3])]
    draws = np.zeros((2, 5, 1))
    ll = simulated_log_likelihood(
        beta_fixed=np.array([0.4]),
        random_means=np.array([-0.3]),
        random_sds=np.array([0.2]),
        thresholds=np.array([-0.4, 0.8]),
        x_fixed=x_fixed,
        x_random=x_random,
        y_codes=y,
        group_indices=groups,
        draws=draws,
    )
    assert np.isfinite(ll)
