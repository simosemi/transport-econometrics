import yaml

from rpopit.checkpoint import load_latest_checkpoint as load_rpopit_checkpoint
from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data
from rpnb.checkpoint import load_latest_checkpoint as load_rpnb_checkpoint
from rpnb.checkpoint import load_run_metadata
from rpnb.cli import main as rpnb_cli_main
from rpnb.model import RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data


def test_rpopit_checkpoint_and_resume(tmp_path):
    data, _ = simulate_ordered_probit_data(
        n_groups=24,
        observations_per_group=2,
        fixed_betas={"x": 0.5},
        random_means={"z": -0.4},
        random_sds={"z": 0.25},
        seed=123,
    )
    data_path = tmp_path / "rpopit_data.csv"
    data.to_csv(data_path, index=False)

    model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=["z"],
        group_id="group",
        categories=[0, 1, 2],
        draws=8,
        maxiter=1,
        checkpoint_interval=1,
    )
    first = model.fit(data_path, save_run=True, output_dir=tmp_path / "runs")
    checkpoint = load_rpopit_checkpoint(first.run_dir)

    assert checkpoint.iteration >= 1
    assert checkpoint.params.size == first.fit_statistics["n_parameters"]

    resumed_model = RandomParametersOrderedProbit(
        dependent="severity",
        fixed=["x"],
        random=["z"],
        group_id="group",
        categories=[0, 1, 2],
        draws=8,
        maxiter=2,
        checkpoint_interval=1,
    )
    resumed = resumed_model.fit(
        data_path,
        save_run=True,
        resume_from=first.run_dir,
    )

    assert resumed.run_dir == first.run_dir
    assert resumed.convergence["resumed_from_iteration"] == checkpoint.iteration
    assert resumed.convergence["iterations"] >= checkpoint.iteration


def test_rpnb_checkpoint_and_resume(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=28,
        observations_per_group=2,
        fixed_betas={"x1": 0.35},
        random_means={"z1": -0.35},
        random_sds={"z1": 0.2},
        alpha=0.5,
        seed=456,
    )
    data_path = tmp_path / "rpnb_data.csv"
    data.to_csv(data_path, index=False)

    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=8,
        maxiter=1,
        checkpoint_interval=1,
    )
    first = model.fit(data_path, save_run=True, output_dir=tmp_path / "runs")
    checkpoint = load_rpnb_checkpoint(first.run_dir)

    assert checkpoint.iteration >= 1
    assert checkpoint.params.size == first.fit_statistics["n_parameters"]

    resumed_model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=8,
        maxiter=2,
        checkpoint_interval=1,
    )
    resumed = resumed_model.fit(
        data_path,
        save_run=True,
        resume_from=first.run_dir,
    )

    assert resumed.run_dir == first.run_dir
    assert resumed.convergence["resumed_from_iteration"] == checkpoint.iteration
    assert resumed.convergence["iterations"] >= checkpoint.iteration


def test_rpnb_cli_resume_uses_previous_run_metadata(tmp_path):
    data, _ = simulate_negative_binomial_data(
        n_groups=24,
        observations_per_group=2,
        fixed_betas={"x1": 0.3},
        random_means={"z1": -0.25},
        random_sds={"z1": 0.2},
        alpha=0.45,
        seed=789,
    )
    data_path = tmp_path / "rpnb_cli_data.csv"
    data.to_csv(data_path, index=False)
    spec_path = tmp_path / "rpnb_cli_model.yaml"
    spec = {
        "model": {
            "dependent": "crashes",
            "offset": "log_exposure",
            "fixed": ["x1"],
            "random": {"z1": {"start_mean": -0.1, "start_sd": 0.2}},
            "group_id": "group",
        },
        "simulation": {"draws": 8, "draw_type": "halton", "seed": 99},
        "estimation": {"maxiter": 1, "checkpoint_interval": 1},
    }
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    out_dir = tmp_path / "cli_runs"

    status = rpnb_cli_main(
        [
            "fit",
            "--data",
            str(data_path),
            "--spec",
            str(spec_path),
            "--out",
            str(out_dir),
            "--no-export",
        ]
    )
    run_dir = next(out_dir.glob("rpnb_*"))
    metadata = load_run_metadata(run_dir)

    assert status == 0
    assert metadata["data_path"] == str(data_path.resolve())
    assert (run_dir / "checkpoints" / "checkpoint_latest.npz").exists()

    resume_status = rpnb_cli_main(["fit", "--resume", str(run_dir), "--no-export"])

    assert resume_status == 0
    assert load_rpnb_checkpoint(run_dir).iteration >= 1
