"""Default biological analysis module registry."""

from pt.core.analysis.canopy import CanopyModule
from pt.core.analysis.circadian import CircadianModule
from pt.core.analysis.color_index import ColorIndexModule
from pt.core.analysis.growth import GrowthModule
from pt.core.analysis.modules import AnalysisRegistry
from pt.core.analysis.volumetric_analysis import VolumetricModule


ANALYSIS_REGISTRY = AnalysisRegistry([
    CanopyModule(),
    GrowthModule(),
    CircadianModule(),
    ColorIndexModule(),
    VolumetricModule(),
])