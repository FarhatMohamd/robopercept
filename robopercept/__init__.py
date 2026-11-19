"""RoboPercept: real-time perception stack for autonomous robots and vehicles."""

__version__ = "1.0.0"
__author__ = "FARHAT Mohamad"

from .structures import Detection, FrameResult, RiskLevel, SafetyDecision, Track, WorldState

__all__ = [
    "Detection",
    "FrameResult",
    "RiskLevel",
    "SafetyDecision",
    "Track",
    "WorldState",
    "__version__",
]
