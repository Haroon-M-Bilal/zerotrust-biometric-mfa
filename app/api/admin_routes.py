"""
Admin routes — audit log inspection and integrity verification.
The /verify endpoint is the demo kill-shot: walk the SHA-256 hash chain,
report any tampered row.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from app.core.audit_chain import AuditChain
from app.db.database import get_db
from app.models.audit import AuditLog
from app.models.user import User
from app.security.auth_dependency import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])


class AuditEntry(BaseModel):
    id: int
    timestamp: str
    user_id: int | None
    event_type: str
    decision: str
    risk_score: float | None
    biometric_confidence: float | None
    ip_address: str | None
    context: dict[str, Any] | None
    llm_explanation: str | None
    prev_hash: str | None
    entry_hash: str

    @classmethod
    def from_orm_entry(cls, e: AuditLog) -> "AuditEntry":
        return cls(
            id=e.id,
            timestamp=e.timestamp.isoformat() if e.timestamp else "",
            user_id=e.user_id,
            event_type=e.event_type,
            decision=e.decision,
            risk_score=e.risk_score,
            biometric_confidence=e.biometric_confidence,
            ip_address=e.ip_address,
            context=e.context,
            llm_explanation=e.llm_explanation,
            prev_hash=e.prev_hash,
            entry_hash=e.entry_hash,
        )


class AuditListResponse(BaseModel):
    total: int
    returned: int
    entries: list[AuditEntry]


class ChainVerifyResponse(BaseModel):
    valid: bool
    total_entries: int
    broken_at_id: int | None
    message: str


def _require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")


@router.get("/audit", response_model=AuditListResponse)
async def list_audit_entries(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """List audit entries newest-first. Admin only."""
    _require_admin(user)
    total = db.query(AuditLog).count()
    rows = (
        db.query(AuditLog)
        .order_by(AuditLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    entries = [AuditEntry.from_orm_entry(r) for r in rows]
    return AuditListResponse(total=total, returned=len(entries), entries=entries)


@router.get("/audit/verify", response_model=ChainVerifyResponse)
async def verify_audit_chain(
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Walk the entire SHA-256 hash chain. Returns valid=False and the
    broken entry ID if any row was modified. Demo kill-shot.
    """
    _require_admin(user)
    chain = AuditChain(db, llm=None)
    valid, broken_id = chain.verify_chain()
    total = db.query(AuditLog).count()
    if valid:
        message = f"All {total} audit entries verified. Chain integrity intact."
    else:
        message = f"TAMPERING DETECTED at entry id={broken_id}. Hash chain broken."
    return ChainVerifyResponse(
        valid=valid,
        total_entries=total,
        broken_at_id=broken_id,
        message=message,
    )