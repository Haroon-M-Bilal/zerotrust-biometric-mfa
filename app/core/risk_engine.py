"""
Composite risk scoring for access decisions.
Combines biometric confidence, IP/device/time signals, and transaction context
into a single 0.0–1.0 score, mapped to ALLOW / CHALLENGE / DENY.

Weights are tunable via settings. Signals are normalized to 0.0 (safe) – 1.0 (risky)
before weighted combination.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from config.settings import get_settings


class Decision(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


class RiskSignals(BaseModel):
    """Raw signals fed into the engine. All optional — engine handles missing data."""
    biometric_confidence: float | None = Field(None, ge=0.0, le=1.0, description="0=no face match, 1=perfect")
    ip_known: bool | None = None
    device_known: bool | None = None
    request_time: datetime | None = None
    transaction_amount: float | None = Field(None, ge=0.0)
    requests_last_minute: int | None = Field(None, ge=0)


class RiskAssessment(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    component_scores: dict[str, float]
    reasons: list[str]


# Weights — sum to 1.0. Tunable later via settings.
WEIGHTS = {
    "biometric": 0.35,
    "ip":        0.15,
    "device":    0.15,
    "time":      0.10,
    "amount":    0.15,
    "velocity":  0.10,
}


class RiskEngine:
    """Stateless. Single instance reusable across requests."""

    def __init__(self) -> None:
        self.settings = get_settings()

    # ---- Per-signal scorers (each returns 0.0 safe → 1.0 risky) ----

    @staticmethod
    def _score_biometric(confidence: float | None) -> float:
        if confidence is None:
            return 0.5  # unknown = mid-risk
        return max(0.0, min(1.0, 1.0 - confidence))

    @staticmethod
    def _score_ip(ip_known: bool | None) -> float:
        if ip_known is None:
            return 0.5
        return 0.0 if ip_known else 0.8

    @staticmethod
    def _score_device(device_known: bool | None) -> float:
        if device_known is None:
            return 0.5
        return 0.0 if device_known else 0.8

    @staticmethod
    def _score_time(request_time: datetime | None) -> float:
        """Off-hours (midnight–6 AM local) = elevated risk."""
        if request_time is None:
            return 0.2
        h = request_time.hour
        if 0 <= h < 6:
            return 0.7
        if 6 <= h < 9 or 22 <= h <= 23:
            return 0.3
        return 0.1

    @staticmethod
    def _score_amount(amount: float | None) -> float:
        """Log-scaled: $0 = 0.0, $1k ≈ 0.3, $10k ≈ 0.6, $100k+ = 1.0."""
        if amount is None or amount <= 0:
            return 0.0
        if amount >= 100_000:
            return 1.0
        # log10(amount) / log10(100_000) ≈ 0..1 for $1..$100k
        import math
        return max(0.0, min(1.0, math.log10(max(amount, 1.0)) / 5.0))

    @staticmethod
    def _score_velocity(requests_last_minute: int | None) -> float:
        if requests_last_minute is None:
            return 0.0
        if requests_last_minute >= 30:
            return 1.0
        return min(1.0, requests_last_minute / 30.0)

    # ---- Composite ----

    def assess(self, signals: RiskSignals) -> RiskAssessment:
        components = {
            "biometric": self._score_biometric(signals.biometric_confidence),
            "ip":        self._score_ip(signals.ip_known),
            "device":    self._score_device(signals.device_known),
            "time":      self._score_time(signals.request_time),
            "amount":    self._score_amount(signals.transaction_amount),
            "velocity":  self._score_velocity(signals.requests_last_minute),
        }
        score = sum(components[k] * WEIGHTS[k] for k in components)
        score = max(0.0, min(1.0, score))

        if score < self.settings.risk_threshold_low:
            decision = Decision.ALLOW
        elif score < self.settings.risk_threshold_high:
            decision = Decision.CHALLENGE
        else:
            decision = Decision.DENY

        reasons = self._explain_components(components)
        return RiskAssessment(
            score=round(score, 4),
            decision=decision,
            component_scores={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
        )

    @staticmethod
    def _explain_components(components: dict[str, float]) -> list[str]:
        """Human-readable reasons for the top risky components (score >= 0.4)."""
        labels = {
            "biometric": "low biometric confidence",
            "ip":        "unrecognized IP address",
            "device":    "unrecognized device",
            "time":      "off-hours request",
            "amount":    "high transaction amount",
            "velocity":  "elevated request rate",
        }
        flagged = [(k, v) for k, v in components.items() if v >= 0.4]
        flagged.sort(key=lambda kv: kv[1], reverse=True)
        return [f"{labels[k]} ({v:.2f})" for k, v in flagged]