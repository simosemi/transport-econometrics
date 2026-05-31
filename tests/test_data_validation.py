import pandas as pd

from rpopit.data_validation import load_schema, validate_dataframe


def _schema():
    return {
        "columns": {
            "severity": {
                "type": "integer",
                "required": True,
                "allow_missing": False,
                "allowed_values": [0, 1, 2, 3],
            },
            "x": {"type": "numeric", "required": True, "allow_missing": False},
            "flag": {
                "type": "binary",
                "required": True,
                "allow_missing": False,
                "allowed_values": [0, 1],
            },
            "optional_year": {
                "type": "integer",
                "required": False,
                "allow_missing": True,
            },
        }
    }


def test_data_validation_accepts_valid_schema():
    data = pd.DataFrame(
        {
            "severity": [0, 1, 2, 3],
            "x": [1.2, 2.3, 3.4, 4.5],
            "flag": [0, 1, 0, 1],
            "optional_year": [2020, 2020, None, 2021],
        }
    )
    report = validate_dataframe(data, _schema(), missing="drop")

    assert report.valid
    assert report.summary["n_rows_validated"] == 4


def test_missing_required_values_can_be_dropped():
    data = pd.DataFrame(
        {
            "severity": [0, 1, 2],
            "x": [1.0, None, 3.0],
            "flag": [0, 1, 1],
        }
    )
    report = validate_dataframe(data, _schema(), missing="drop")

    assert report.valid
    assert report.summary["dropped_rows_missing"] == 1
    assert report.summary["n_rows_validated"] == 2


def test_missing_required_values_can_error():
    data = pd.DataFrame(
        {
            "severity": [0, 1, 2],
            "x": [1.0, None, 3.0],
            "flag": [0, 1, 1],
        }
    )
    report = validate_dataframe(data, _schema(), missing="error")

    assert not report.valid
    assert report.summary["n_errors"] == 1
    assert report.summary["n_rows_validated"] == 3


def test_schema_csv_loader_parses_allowed_values(tmp_path):
    schema_path = tmp_path / "schema.csv"
    schema_path.write_text(
        "column,role,type,required,allow_missing,allowed_values,description\n"
        "severity,dependent,integer,true,false,0|1|2|3,Severity\n",
        encoding="utf-8",
    )
    schema = load_schema(schema_path)

    assert schema["columns"]["severity"]["allowed_values"] == [0, 1, 2, 3]
