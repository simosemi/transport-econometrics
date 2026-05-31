from pathlib import Path

import pytest

from rpnb.config import load_model_spec, parse_model_spec


def test_parse_model_spec_accepts_offset_and_random_parameters():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "crashes",
                "offset": "log_exposure",
                "fixed": ["x1"],
                "random": {"z1": {"start_mean": -0.1, "start_sd": 0.4}},
                "group_id": "segment",
                "correlated_random_parameters": True,
            },
            "simulation": {"draws": 64, "draw_type": "sobol"},
            "estimation": {"start_alpha": 0.7},
        }
    )

    assert spec.dependent == "crashes"
    assert spec.offset == "log_exposure"
    assert spec.fixed == ("x1",)
    assert spec.random[0].name == "z1"
    assert spec.random[0].start_sd == 0.4
    assert spec.correlated_random_parameters
    assert spec.start_alpha == 0.7


def test_parse_model_spec_requires_offset():
    with pytest.raises(ValueError, match="offset"):
        parse_model_spec({"model": {"dependent": "crashes"}})


def test_example_yaml_loads():
    spec = load_model_spec(Path("examples/rpnb_model.yaml"))

    assert spec.dependent == "crashes"
    assert spec.offset == "log_exposure"
    assert spec.draws == 128
