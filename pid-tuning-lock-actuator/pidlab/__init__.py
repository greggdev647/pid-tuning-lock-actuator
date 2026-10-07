"""pidlab: PID tuning for a simulated door-lock actuator."""

from .controller import PID, Gains
from .metrics import Metrics, compute_metrics
from .plant import Plant
from .simulate import Result, Scenario, simulate
from .tuning import optimize_gains, tuning_cost, ziegler_nichols

__all__ = [
    "PID",
    "Gains",
    "Metrics",
    "compute_metrics",
    "Plant",
    "Result",
    "Scenario",
    "simulate",
    "optimize_gains",
    "tuning_cost",
    "ziegler_nichols",
]
