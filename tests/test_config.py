from rpopit.config import parse_model_spec


def test_parse_random_parameter_mapping_list():
    spec = parse_model_spec(
        {
            "model": {
                "dependent": "severity",
                "fixed": ["x"],
                "random": [
                    {"name": "z", "distribution": "normal", "start_mean": -0.2, "start_sd": 0.5}
                ],
            }
        }
    )
    assert spec.dependent == "severity"
    assert spec.random[0].name == "z"
    assert spec.random[0].start_sd == 0.5
