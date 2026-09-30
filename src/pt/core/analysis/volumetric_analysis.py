"""Module adapter for visual-hull reconstruction."""

from pt.core.analysis.modules import AnalysisObservation, AnalysisResult


class VolumetricModule:
    name = "volumetric"
    version = "1.0.0"

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        reconstruct = observation.context.get("reconstruct")
        if reconstruct is None:
            return AnalysisResult()
        result = reconstruct(observation.context.get("device_masks") or {})
        return AnalysisResult(metrics={"volumetric": result})