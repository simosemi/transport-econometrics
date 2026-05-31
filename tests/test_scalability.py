import numpy as np

from rpopit.draws import generate_draws
from rpopit.likelihood import group_structure_from_indices, simulated_log_likelihood
from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data


def _likelihood_fixture():
    data, _ = simulate_ordered_probit_data(
        n_groups=24,
        observations_per_group=3,
        fixed_betas={"x": 0.5},
        random_means={"z": -0.4},
        random_sds={"z": 0.3},
        thresholds=(-0.3, 0.7),
        seed=1234,
    )
    group_values = data["group"].to_numpy()
    groups = [np.flatnonzero(group_values == group) for group in sorted(data["group"].unique())]
    order, starts, counts = group_structure_from_indices(groups, len(data))
    draws = generate_draws(len(groups), 32, 1, "halton", seed=9)
    return data, groups, order, starts, counts, draws


def test_chunked_likelihood_matches_unchunked_likelihood():
    data, groups, order, starts, counts, draws = _likelihood_fixture()
    kwargs = {
        "beta_fixed": np.array([0.5]),
        "random_means": np.array([-0.4]),
        "random_sds": np.array([0.3]),
        "thresholds": np.array([-0.3, 0.7]),
        "x_fixed": data[["x"]].to_numpy()[order],
        "x_random": data[["z"]].to_numpy()[order],
        "y_codes": data["severity"].to_numpy()[order],
        "group_indices": groups,
        "group_starts": starts,
        "group_counts": counts,
        "draws": draws,
    }

    unchunked = simulated_log_likelihood(**kwargs, chunk_size=None)
    chunked = simulated_log_likelihood(**kwargs, chunk_size=10)

    assert abs(unchunked - chunked) < 1e-10


def test_worker_likelihood_matches_serial_likelihood():
    data, groups, order, starts, counts, draws = _likelihood_fixture()
    kwargs = {
        "beta_fixed": np.array([0.5]),
        "random_means": np.array([-0.4]),
        "random_sds": np.array([0.3]),
        "thresholds": np.array([-0.3, 0.7]),
        "x_fixed": data[["x"]].to_numpy()[order],
        "x_random": data[["z"]].to_numpy()[order],
        "y_codes": data["severity"].to_numpy()[order],
        "group_indices": groups,
        "group_starts": starts,
        "group_counts": counts,
        "draws": draws,
        "chunk_size": 10,
    }

    serial = simulated_log_likelihood(**kwargs, workers=1)
    parallel = simulated_log_likelihood(**kwargs, workers=2)

    assert abs(serial - parallel) < 1e-10


def test_fit_records_timing_diagnostics():
    data, _ = simulate_ordered_probit_data(
        n_groups=30,
        observations_per_group=2,
        fixed_betas={"x": 0.4},
        random_means={"z": -0.2},
        random_sds={"z": 0.2},
        seed=2027,
    )
    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=["z"],
        group_id="group",
        categories=[0, 1, 2],
        draws=16,
        chunk_size=12,
        maxiter=2,
        tolerance=1e-4,
    )
    results = model.fit(data, save_run=False)

    assert results.timing["objective_calls"] > 0
    assert results.timing["average_objective_seconds"] > 0
    assert results.convergence["chunk_size"] == 12
    assert results.convergence["workers_used"] == 1
