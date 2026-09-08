"""
NIRIKSHAK -- zero-shot visual quality inspection on Snapdragon.

Author:  Jash Pasad, Indian Institute of Technology Gandhinagar
Repo:    https://github.com/jashpasad/NIRIKSHAK
Licence: Apache-2.0
"""

__version__ = "0.3.0"
__author__ = "Jash Pasad"
__affiliation__ = "IIT Gandhinagar"

from .tier1_screen.detector import AnomalyDetector, InspectionResult, DefectRegion
from .tier1_screen.embedder import ReferencePatchEmbedder, OnnxPatchEmbedder
from .tier2_explain.vlm import DefectExplainer
from .tier3_reason.shift_report import ShiftReporter, compute_stats, InspectionEvent
from .pipeline import InspectionCascade, CascadePolicy

__all__ = [
    "AnomalyDetector", "InspectionResult", "DefectRegion",
    "ReferencePatchEmbedder", "OnnxPatchEmbedder",
    "DefectExplainer", "ShiftReporter", "compute_stats", "InspectionEvent",
    "InspectionCascade", "CascadePolicy",
]
