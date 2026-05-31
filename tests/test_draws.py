import numpy as np

from rpopit.draws import generate_draws


def test_pseudo_draws_are_reproducible():
    first = generate_draws(4, 8, 2, "pseudo", seed=10)
    second = generate_draws(4, 8, 2, "pseudo", seed=10)
    assert first.shape == (4, 8, 2)
    np.testing.assert_allclose(first, second)


def test_halton_and_sobol_draws_are_finite():
    for draw_type in ("halton", "sobol"):
        draws = generate_draws(3, 16, 2, draw_type, seed=10)
        assert draws.shape == (3, 16, 2)
        assert np.isfinite(draws).all()
