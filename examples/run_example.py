"""Simulate data and fit an rpopit model."""

from pathlib import Path

from rpopit.config import load_model_spec
from rpopit.model import RandomParametersOrderedProbit
from rpopit.simulation import simulate_ordered_probit_data


def main() -> None:
    data, truth = simulate_ordered_probit_data(seed=77)
    data_path = Path("examples/output/simulated_crashes.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    spec = load_model_spec("examples/model.yaml")
    model = RandomParametersOrderedProbit.from_spec(spec)
    results = model.fit(data_path, export=True)
    print(results.summary())
    print("Truth:", truth)
    print("Run directory:", results.run_dir)


if __name__ == "__main__":
    main()
