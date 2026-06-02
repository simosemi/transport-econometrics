import numpy as np
import pytest

from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data
from rpnb.model import RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data


def test_rpnb_bfgs_and_lbfgsb_produce_similar_estimates():
    data, _ = simulate_negative_binomial_data(
        n_groups=80,
        observations_per_group=2,
        intercept=-0.6,
        fixed_betas={"x1": 0.35},
        random_means={},
        alpha=0.45,
        seed=202610,
    )

    bfgs = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        optimizer="bfgs",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)
    lbfgsb = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=[],
        optimizer="lbfgsb",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)

    assert bfgs.convergence["optimizer_method"] == "bfgs"
    assert lbfgsb.convergence["optimizer_method"] == "lbfgsb"
    _assert_common_estimates_close(bfgs.parameter_table, lbfgsb.parameter_table)


def test_rpopit_bfgs_and_lbfgsb_produce_similar_estimates():
    data, _ = simulate_ordered_probit_data(
        n_groups=80,
        observations_per_group=2,
        fixed_betas={"x": 0.65},
        random_means={},
        random_sds={},
        thresholds=(-0.35, 0.75),
        seed=202611,
    )

    bfgs = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=[],
        categories=[0, 1, 2],
        optimizer="bfgs",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)
    lbfgsb = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=[],
        categories=[0, 1, 2],
        optimizer="lbfgsb",
        maxiter=120,
        tolerance=1e-5,
        checkpoint_interval=0,
    ).fit(data, save_run=False)

    assert bfgs.convergence["optimizer_method"] == "bfgs"
    assert lbfgsb.convergence["optimizer_method"] == "lbfgsb"
    _assert_common_estimates_close(bfgs.parameter_table, lbfgsb.parameter_table)


def test_unsupported_optimizer_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        RandomParametersNegativeBinomial(
            dependent="crashes",
            offset="log_exposure",
            optimizer="newton-cg",
        )


def _assert_common_estimates_close(left, right, atol: float = 2e-3) -> None:
    left_estimates = left.set_index("parameter")["estimate"].sort_index()
    right_estimates = right.set_index("parameter")["estimate"].sort_index()
    common = left_estimates.index.intersection(right_estimates.index)

    assert len(common) == len(left_estimates) == len(right_estimates)
    np.testing.assert_allclose(
        left_estimates.loc[common].to_numpy(),
        right_estimates.loc[common].to_numpy(),
        atol=atol,
        rtol=atol,
    )
