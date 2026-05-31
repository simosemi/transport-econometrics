"""Run a small RPNB example with simulated data."""

from __future__ import annotations

from pathlib import Path

from rpnb import RandomParametersNegativeBinomial, simulate_negative_binomial_data


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data, truth = simulate_negative_binomial_data(
        n_groups=80,
        observations_per_group=3,
        intercept=-0.4,
        fixed_betas={"x1": 0.35},
        random_means={"z1": -0.45},
        random_sds={"z1": 0.30},
        alpha=0.55,
        seed=2026,
    )
    example_data = root / "examples" / "rpnb_simulated.csv"
    data.to_csv(example_data, index=False)

    model = RandomParametersNegativeBinomial(
        dependent="crashes",
        offset="log_exposure",
        fixed=["x1"],
        random=["z1"],
        group_id="group",
        draws=128,
        draw_type="halton",
        seed=2026,
        maxiter=300,
        tolerance=1e-5,
        output_dir=str(root / "runs"),
    )
    results = model.fit(example_data, export=True)
    print(results.summary())
    print(f"Truth: {truth}")
    if results.run_dir is not None:
        print(f"Run directory: {results.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
