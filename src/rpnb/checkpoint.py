"""Checkpoint helpers for interrupted simulated maximum likelihood runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class OptimizerCheckpoint:
    """Saved optimizer progress for restartable runs."""

    iteration: int
    params: np.ndarray
    objective_value: float
    log_likelihood: float
    metadata: dict[str, Any]
    path: Path


def save_checkpoint(
    run_dir: str | Path,
    iteration: int,
    params: np.ndarray,
    objective_value: float,
    log_likelihood: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the current optimizer state and parameter vector."""

    directory = Path(run_dir) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": int(iteration),
        "objective_value": float(objective_value),
        "log_likelihood": float(log_likelihood),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **dict(metadata or {}),
    }
    params = np.asarray(params, dtype=float)
    target = directory / f"checkpoint_iter_{int(iteration):06d}.npz"
    latest = directory / "checkpoint_latest.npz"
    _write_npz_atomic(target, params, payload)
    _write_npz_atomic(latest, params, payload)
    _write_json_atomic(directory / "checkpoint_latest.json", params, payload)
    return target


def load_latest_checkpoint(run_dir: str | Path) -> OptimizerCheckpoint:
    """Load the latest checkpoint from a run directory."""

    path = Path(run_dir) / "checkpoints" / "checkpoint_latest.npz"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}.")
    with np.load(path, allow_pickle=False) as loaded:
        params = np.asarray(loaded["params"], dtype=float)
        metadata = json.loads(str(loaded["metadata_json"].item()))
    return OptimizerCheckpoint(
        iteration=int(metadata["iteration"]),
        params=params,
        objective_value=float(metadata["objective_value"]),
        log_likelihood=float(metadata["log_likelihood"]),
        metadata=metadata,
        path=path,
    )


def write_run_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Write restart metadata such as the source CSV path."""

    path = Path(run_dir) / "run_metadata.yaml"
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
    existing.update(metadata)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(existing, handle, sort_keys=False)
    return path


def load_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    """Load restart metadata from a run directory."""

    path = Path(run_dir) / "run_metadata.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_npz_atomic(path: Path, params: np.ndarray, metadata: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            params=params,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
    tmp.replace(path)


def _write_json_atomic(path: Path, params: np.ndarray, metadata: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(metadata)
    payload["params"] = params.tolist()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
