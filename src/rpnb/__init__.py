"""Random Parameters Negative Binomial regression with log-offset support."""

from rpnb.model import RPNBModel, RandomParametersNegativeBinomial
from rpnb.simulation import simulate_negative_binomial_data

__all__ = [
    "RPNBModel",
    "RandomParametersNegativeBinomial",
    "simulate_negative_binomial_data",
]

__version__ = "0.1.0"
