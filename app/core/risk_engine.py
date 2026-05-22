"""
Composite risk scoring for access decisions.
Combines biometric confidence, IP/device/time signals, and transaction context
into a single 0.0–1.0 score, mapped to ALLOW / CHALLENGE / DENY.

Hybrid mode: when the trained Random Forest classifier is available, its
fraud probability is blended with the rule-based score (default 60% rules,
40% ML). Set use_ml=False to use rules only.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from config.settings import get_settings


class Decision(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


class RiskSignals(BaseModel):
    biometric_confidence: float | None = Field(None, ge=0.0, le=1.0)
    ip_known: bool | None = None
    device_known: bool | None = None
    request_time: datetime | None = None
    transaction_amount: float | None = Field(None, ge=0.0)
    requests_last_minute: int | None = Field(None, ge=0)
    # Extra signals consumed only by the ML classifier
    originator_balance: float | None = Field(None, ge=0.0)
    destination_known: bool | None = None
    is_transfer: bool | None = None


class RiskAssessment(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    component_scores: dict[str, float]
    reasons: list[str]
    rule_score: float
    ml_score: float | None = None


WEIGHTS = {
    "biometric": 0.35,
    "ip":        0.15,
    "device":    0.15,
    "time":      0.10,
    "amount":    0.15,
    "velocity":  0.10,
}

# Hybrid blend: rule_weight + ml_weight = 1.0
RULE_BLEND = 0.7
ML_BLEND = 0.3


class RiskEngine:
    def __init__(self, use_ml: bool = True) -> None:
        self.settings = get_settings()
        self.classifier = None
        if use_ml:
            try:
                from app.ml.risk_classifier import RiskClassifier
                self.classifier = RiskClassifier.load()
            except (FileNotFoundError, ImportError):
                self.classifier = None  # fallback to rules-only

    # ---- Per-signal rule scorers (0.0 safe → 1.0 risky) ----

    @staticmethod
    def _score_biometric(c: float | None) -> float:
        return 0.5 if c is None else max(0.0, min(1.0, 1.0 - c))

    @staticmethod
    def _score_ip(known: bool | None) -> float:
        return 0.5 if known is None else (0.0 if known else 0.8)

    @staticmethod
    def _score_device(known: bool | None) -> float:
        return 0.5 if known is None else (0.0 if known else 0.8)

    @staticmethod
    def _score_time(t: datetime | None) -> float:
        if t is None:
            return 0.2
        h = t.hour
        if 0 <= h < 6:   return 0.7
        if 6 <= h < 9 or 22 <= h <= 23: return 0.3
        return 0.1

    @staticmethod
    def _score_amount(amount: float | None) -> float:
        if amount is None or amount <= 0:
            return 0.0
        if amount >= 100_000:
            return 1.0
        import math
        return max(0.0, min(1.0, math.log10(max(amount, 1.0)) / 5.0))

    @staticmethod
    def _score_velocity(rpm: int | None) -> float:
        if rpm is None:
            return 0.0
        if rpm >= 30:
            return 1.0
        return min(1.0, rpm / 30.0)

    def _rule_score(self, signals: RiskSignals) -> tuple[float, dict[str, float]]:
        components = {
            "biometric": self._score_biometric(signals.biometric_confidence),
            "ip":        self._score_ip(signals.ip_known),
            "device":    self._score_device(signals.device_known),
            "time":      self._score_time(signals.request_time),
            "amount":    self._score_amount(signals.transaction_amount),
            "velocity":  self._score_velocity(signals.requests_last_minute),
        }
        score = sum(components[k] * WEIGHTS[k] for k in components)
        return max(0.0, min(1.0, score)), components

    def _ml_score(self, signals: RiskSignals) -> float | None:
        """Run RF classifier; return None if unavailable or insufficient signals."""
        if self.classifier is None:
            return None
        if signals.transaction_amount is None:
            return None
        import math
        amount = max(signals.transaction_amount, 0.0)
        origin_bal = signals.originator_balance if signals.originator_balance is not None else amount + 1.0
        hour = signals.request_time.hour if signals.request_time else 12
        is_transfer = 1 if signals.is_transfer else 0
        dest_new = 0 if signals.destination_known else 1
        balance_drained = 1 if (origin_bal > 0 and amount >= origin_bal * 0.95) else 0

        features = {
            "amount_log":        math.log10(amount + 1),
            "hour_of_day":       hour,
            "is_off_hours":      1 if 0 <= hour < 6 else 0,
            "is_transfer":       is_transfer,
            "balance_drained":   balance_drained,
            "amount_to_balance": amount / (origin_bal + 1.0),
            "dest_new":          dest_new,
        }
        try:
            return self.classifier.predict_proba(features)
        except Exception:
            return None

    def assess(self, signals: RiskSignals) -> RiskAssessment:
        rule_score, components = self._rule_score(signals)
        ml_score = self._ml_score(signals)

        if ml_score is not None:
            final_score = RULE_BLEND * rule_score + ML_BLEND * ml_score
        else:
            final_score = rule_score
        final_score = max(0.0, min(1.0, final_score))

        if final_score < self.settings.risk_threshold_low:
            decision = Decision.ALLOW
        elif final_score < self.settings.risk_threshold_high:
            decision = Decision.CHALLENGE
        else:
            decision = Decision.DENY

        reasons = self._explain(components, ml_score)
        return RiskAssessment(
            score=round(final_score, 4),
            decision=decision,
            component_scores={k: round(v, 4) for k, v in components.items()},
            reasons=reasons,
            rule_score=round(rule_score, 4),
            ml_score=round(ml_score, 4) if ml_score is not None else None,
        )

    @staticmethod
    def _explain(components: dict[str, float], ml_score: float | None) -> list[str]:
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
        reasons = [f"{labels[k]} ({v:.2f})" for k, v in flagged]
        if ml_score is not None and ml_score >= 0.5:
            reasons.insert(0, f"ML classifier flagged transaction (P_fraud={ml_score:.2f})")
        return reasons