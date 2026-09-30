"""Image-level canopy segmentation module."""

from pt.core.analysis.modules import AnalysisObservation, AnalysisResult


class CanopyModule:
    name = "canopy"
    version = "1.0.0"

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        calculator = observation.context.get("canopy_calculator")
        if calculator is None:
            return AnalysisResult()
        canopy = calculator(
            observation.frame,
            observation.context.get("scale_px_per_mm"),
            observation.device_id,
            auto_ignore_mask=observation.context.get("auto_ignore_mask"),
        )
        return AnalysisResult(metrics={"canopy": canopy})
