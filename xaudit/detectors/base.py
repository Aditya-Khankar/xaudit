from abc import ABC, abstractmethod
from dataclasses import dataclass
from xaudit.recorder.trace_recorder import AgentTrace


@dataclass
class DetectorResult:
    detector_name: str
    detected: bool
    score: float        # always 0.0–1.0
    threshold: float
    evidence: dict
    interpretation: str
    detector_version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "evidence": self.evidence,
            "interpretation": self.interpretation,
            "version": self.detector_version,
        }


class BaseDetector(ABC):
    name: str
    threshold: float
    version: str = "1.0.0"

    @abstractmethod
    def detect(self, trace: AgentTrace) -> DetectorResult:
        pass

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _insufficient(self, reason: str) -> DetectorResult:
        """Standard result for traces that can't be evaluated."""
        return DetectorResult(
            detector_name=self.name,
            detected=False,
            score=0.0,
            threshold=self.threshold,
            evidence={"reason": reason},
            interpretation=reason,
            detector_version=self.version,
        )
