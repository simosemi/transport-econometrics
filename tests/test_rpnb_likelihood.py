import numpy as np
from scipy.stats import nbinom

from rpnb.likelihood import negative_binomial_logpmf, simulated_log_likelihood


def test_negative_binomial_logpmf_matches_scipy_nb2_parameterization():
    y = np.array([0, 1, 3, 8])
    mu = np.array([0.4, 1.2, 2.5, 7.0])
    alpha = 0.65
    size = 1.0 / alpha
    prob = size / (size + mu)

    expected = nbinom.logpmf(y, size, prob)
    actual = negative_binomial_logpmf(y, mu, alpha)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fixed_only_log_likelihood_equals_direct_sum():
    y = np.array([0, 2, 1, 4])
    offset = np.log(np.array([1.0, 1.4, 0.8, 2.0]))
    x_fixed = np.column_stack([np.ones(4), np.array([-1.0, 0.3, 0.8, 1.2])])
    beta = np.array([-0.2, 0.45])
    alpha = 0.4
    eta = offset + x_fixed @ beta
    mu = np.exp(eta)

    actual = simulated_log_likelihood(
        beta_fixed=beta,
        random_means=np.array([]),
        alpha=alpha,
        x_fixed=x_fixed,
        x_random=np.zeros((4, 0)),
        offset=offset,
        y=y,
        group_indices=[np.arange(4)],
        draws=np.zeros((1, 1, 0)),
    )
    expected = np.sum(negative_binomial_logpmf(y, mu, alpha))

    assert np.isclose(actual, expected)
