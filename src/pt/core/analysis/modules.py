"""Composable analysis modules for a single phenotyping observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AnalysisObservation:
    """Inputs shared by analysis modules for one device observation."""

    device_id: str
    image_path: str | None = None
    frame: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    previous: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Module-produced metric fields and alert records."""

    metrics: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)


class AnalysisModule(Protocol):
    name: str

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        """Produce metrics and alerts from an observation and its context."""


class AnalysisRegistry:
    def __init__(self, modules: list[AnalysisModule] | None = None):
        self._modules: dict[str, AnalysisModule] = {}
        self._disabled: set[str] = set()
        for module in modules or []:
            self.register(module)

    def register(self, module: AnalysisModule) -> None:
        if not module.name or module.name in self._modules:
            raise ValueError(f"Analysis module name must be unique and non-empty: {module.name!r}")
        self._modules[module.name] = module
        self._disabled.discard(module.name)

    def enable(self, name: str) -> None:
        if name not in self._modules:
            raise KeyError(f"Unknown analysis module: {name}")
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        if name not in self._modules:
            raise KeyError(f"Unknown analysis module: {name}")
        self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        return name in self._modules and name not in self._disabled

    def names(self, enabled_only: bool = False) -> list[str]:
        return [
            name for name in self._modules
            if not enabled_only or name not in self._disabled
        ]

    def module_versions(self) -> dict[str, str]:
        return {
            name: str(getattr(module, "version", "1.0.0"))
            for name, module in self._modules.items()
        }

    def run(self, observation: AnalysisObservation, module_names: set[str] | None = None) -> dict[str, AnalysisResult]:
        results = {}
        for name, module in self._modules.items():
            if name in self._disabled or (module_names is not None and name not in module_names):
                continue
            result = module.analyze(observation)
            if not isinstance(result, AnalysisResult):
                raise TypeError(f"Analysis module {name!r} must return AnalysisResult")
            results[name] = result
            observation.metrics.update(result.metrics)
        return results


class FunctionAnalysisModule:
    """Adapter for existing pure analysis functions during incremental migration."""

    def __init__(self, name: str, analyzer):
        self.name = name
        self._analyzer = analyzer

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        return self._analyzer(observation)