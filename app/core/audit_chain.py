"""
Append-only audit log with SHA-256 hash chaining.
Each entry's hash includes the previous entry's hash, making tampering detectable.
Async LLM explanation generated on each entry (best-effort, failures don't block).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.llm_explainer import AccessDecisionContext, LLMExplainer
from app.models.audit import AuditLog

GENESIS_HASH = "0" * 64


class AuditChain:
    def __init__(self, db: Session, llm: LLMExplainer | None = None) -> None:
        self.db = db
        self.llm = llm

    def _get_last_hash(self) -> str:
        last = self.db.query(AuditLog).order_by(AuditLog.id.desc()).first()
        return last.entry_hash if last else GENESIS_HASH

    @staticmethod
    def _build_payload(
        *,
        timestamp: datetime,
        user_id: int | None,
        event_type: str,
        decision: str,
        risk_score: float | None,
        biometric_confidence: float | None,
        ip_address: str | None,
        context: dict[str, Any] | None,
        llm_explanation: str | None,
    ) -> dict[str, Any]:
        """Canonical payload representation — used identically by append and verify."""
        # Normalize timestamp to UTC, drop tz info, microsecond=0 for SQLite-safe roundtrip
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        timestamp = timestamp.replace(microsecond=0)
        return {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "event_type": event_type,
            "decision": decision,
            "risk_score": risk_score,
            "biometric_confidence": biometric_confidence,
            "ip_address": ip_address,
            "context": context or {},
            "llm_explanation": llm_explanation,
        }

    @staticmethod
    def _compute_hash(payload: dict[str, Any], prev_hash: str) -> str:
        canonical = json.dumps(
            {"prev": prev_hash, "data": payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def append(
        self,
        *,
        event_type: str,
        decision: str,
        user_id: int | None = None,
        risk_score: float | None = None,
        biometric_confidence: float | None = None,
        ip_address: str | None = None,
        context: dict[str, Any] | None = None,
        generate_explanation: bool = True,
    ) -> AuditLog:
        prev_hash = self._get_last_hash()
        timestamp = datetime.now(timezone.utc).replace(microsecond=0)

        explanation: str | None = None
        if generate_explanation and self.llm is not None:
            try:
                ctx = AccessDecisionContext(
                    user_id=str(user_id) if user_id is not None else "anonymous",
                    resource=f"banking:{event_type}",
                    action=event_type,
                    decision=decision,
                    risk_score=risk_score if risk_score is not None else 0.0,
                    biometric_confidence=biometric_confidence if biometric_confidence is not None else 0.0,
                    signals={
                        "ip_address": ip_address,
                        **(context or {}),
                    },
                )
                result = await self.llm.explain(ctx)
                explanation = result.explanation
            except Exception:
                explanation = None

        payload = self._build_payload(
            timestamp=timestamp,
            user_id=user_id,
            event_type=event_type,
            decision=decision,
            risk_score=risk_score,
            biometric_confidence=biometric_confidence,
            ip_address=ip_address,
            context=context,
            llm_explanation=explanation,
        )
        entry_hash = self._compute_hash(payload, prev_hash)

        entry = AuditLog(
            timestamp=timestamp.replace(tzinfo=None),
            user_id=user_id,
            event_type=event_type,
            decision=decision,
            risk_score=risk_score,
            biometric_confidence=biometric_confidence,
            ip_address=ip_address,
            context=context,
            llm_explanation=explanation,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def verify_chain(self) -> tuple[bool, int | None]:
        entries = self.db.query(AuditLog).order_by(AuditLog.id.asc()).all()
        expected_prev = GENESIS_HASH
        for entry in entries:
            if entry.prev_hash != expected_prev:
                return False, entry.id
            payload = self._build_payload(
                timestamp=entry.timestamp,
                user_id=entry.user_id,
                event_type=entry.event_type,
                decision=entry.decision,
                risk_score=entry.risk_score,
                biometric_confidence=entry.biometric_confidence,
                ip_address=entry.ip_address,
                context=entry.context,
                llm_explanation=entry.llm_explanation,
            )
            recomputed = self._compute_hash(payload, entry.prev_hash)
            if recomputed != entry.entry_hash:
                return False, entry.id
            expected_prev = entry.entry_hash
        return True, None