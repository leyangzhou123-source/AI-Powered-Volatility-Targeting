"""Estimator-controller pair abstraction for strategy routing."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyPair:
    """Container for a compatible estimator + controller pair."""

    name: str
    estimator: Any
    controller: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimator_name(self) -> str:
        return self.estimator.__class__.__name__

    @property
    def controller_name(self) -> str:
        return self.controller.__class__.__name__

    def to_record(self) -> dict[str, Any]:
        return {
            "pair": self.name,
            "estimator": self.estimator_name,
            "controller": self.controller_name,
        }
