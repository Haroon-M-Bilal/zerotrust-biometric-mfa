"""
Transaction routes — money transfer with step-up biometric MFA.
Pipeline: risk engine → decision → (CHALLENGE: face re-verify) → audit chain → LLM explanation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from app.core.audit_chain import AuditChain
from app.core.biometric import BiometricService, FaceEmbedding
from app.core.llm_explainer import LLMExplainer
from app.core.risk_engine import Decision, RiskEngine, RiskSignals
from app.db.database import get_db
from app.models.account import Account
from app.models.transaction import Transaction
from app.models.user import User
from app.security.auth_dependency import get_current_user

router = APIRouter(prefix="/transactions", tags=["transactions"])

risk_engine = RiskEngine(use_ml=True)
biometric = BiometricService()
from config.settings import get_settings
_settings = get_settings()
llm = LLMExplainer(base_url=_settings.llm_base_url, model=_settings.llm_model)

UPLOAD_DIR = Path("data") / "uploads" / "face_captures"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Schemas ----------

class TransferRequest(BaseModel):
    to_account: str = Field(..., min_length=4, max_length=32)
    amount: float = Field(..., gt=0)
    force_off_hours: bool = Field(False, description="Demo flag — pretend it's 3 AM")


class TransferResponse(BaseModel):
    transaction_id: int | None
    status: str  # completed | pending_challenge | denied
    decision: str  # ALLOW | CHALLENGE | DENY
    risk_score: float
    rule_score: float
    ml_score: float | None
    reasons: list[str]
    challenge_token: str | None = None
    message: str


class ChallengeVerifyResponse(BaseModel):
    transaction_id: int
    status: str
    biometric_distance: float
    biometric_confidence: float
    message: str


# ---------- Helpers ----------

# In-memory pending-challenge store (demo only; production would use Redis)
_pending_challenges: dict[str, dict] = {}


def _save_face(upload: UploadFile, user_id: int) -> Path:
    suffix = Path(upload.filename or "f.jpg").suffix.lower() or ".jpg"
    target = UPLOAD_DIR / f"verify_{user_id}_{uuid.uuid4().hex[:8]}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return target


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ---------- Routes ----------

@router.post("/transfer", response_model=TransferResponse)
async def transfer(
    payload: TransferRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    # Lookup source account
    source = db.query(Account).filter(Account.user_id == user.id).first()
    if source is None:
        raise HTTPException(status_code=404, detail="No source account found for user")
    if float(source.balance) < payload.amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    dest = db.query(Account).filter(Account.account_number == payload.to_account).first()
    destination_known = dest is not None

    # Build risk signals
    now = datetime.now(timezone.utc)
    if payload.force_off_hours:
        now = now.replace(hour=3, minute=0)

    signals = RiskSignals(
        biometric_confidence=0.95,  # assume strong from login (real flow would track session quality)
        ip_known=True,               # would compare against past sessions
        device_known=request.headers.get("user-agent") is not None,
        request_time=now,
        transaction_amount=payload.amount,
        requests_last_minute=1,
        originator_balance=float(source.balance),
        destination_known=destination_known,
        is_transfer=True,
    )
    assessment = risk_engine.assess(signals)

    chain = AuditChain(db, llm=llm)
    decision_str = assessment.decision.value

    if assessment.decision == Decision.DENY:
        txn = Transaction(
            from_account_id=source.id,
            to_account_id=dest.id if dest else None,
            amount=payload.amount,
            transaction_type="transfer",
            status="denied",
            risk_score=assessment.score,
            decision=decision_str,
        )
        db.add(txn)
        db.commit()
        db.refresh(txn)
        await chain.append(
            event_type="transfer",
            decision=decision_str,
            user_id=user.id,
            risk_score=assessment.score,
            biometric_confidence=signals.biometric_confidence,
            ip_address=_client_ip(request),
            context={"amount": payload.amount, "to": payload.to_account, "transaction_id": txn.id},
        )
        return TransferResponse(
            transaction_id=txn.id,
            status="denied",
            decision=decision_str,
            risk_score=assessment.score,
            rule_score=assessment.rule_score,
            ml_score=assessment.ml_score,
            reasons=assessment.reasons,
            message="Transaction blocked. Our security team has been notified.",
        )

    if assessment.decision == Decision.CHALLENGE:
        # Persist pending transaction, issue challenge token
        txn = Transaction(
            from_account_id=source.id,
            to_account_id=dest.id if dest else None,
            amount=payload.amount,
            transaction_type="transfer",
            status="pending",
            risk_score=assessment.score,
            decision=decision_str,
        )
        db.add(txn)
        db.commit()
        db.refresh(txn)
        token = uuid.uuid4().hex
        _pending_challenges[token] = {
            "transaction_id": txn.id,
            "user_id": user.id,
            "issued_at": now,
        }
        await chain.append(
            event_type="transfer_challenge",
            decision=decision_str,
            user_id=user.id,
            risk_score=assessment.score,
            biometric_confidence=signals.biometric_confidence,
            ip_address=_client_ip(request),
            context={"amount": payload.amount, "to": payload.to_account, "transaction_id": txn.id, "reasons": assessment.reasons},
        )
        return TransferResponse(
            transaction_id=txn.id,
            status="pending_challenge",
            decision=decision_str,
            risk_score=assessment.score,
            rule_score=assessment.rule_score,
            ml_score=assessment.ml_score,
            reasons=assessment.reasons,
            challenge_token=token,
            message="Additional verification required. Please complete face re-verification.",
        )

    # ALLOW path — execute the transfer
    source.balance = float(source.balance) - payload.amount
    if dest:
        dest.balance = float(dest.balance) + payload.amount

    txn = Transaction(
        from_account_id=source.id,
        to_account_id=dest.id if dest else None,
        amount=payload.amount,
        transaction_type="transfer",
        status="completed",
        risk_score=assessment.score,
        decision=decision_str,
        completed_at=now.replace(tzinfo=None),
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)
    await chain.append(
        event_type="transfer",
        decision=decision_str,
        user_id=user.id,
        risk_score=assessment.score,
        biometric_confidence=signals.biometric_confidence,
        ip_address=_client_ip(request),
        context={"amount": payload.amount, "to": payload.to_account, "transaction_id": txn.id},
    )
    return TransferResponse(
        transaction_id=txn.id,
        status="completed",
        decision=decision_str,
        risk_score=assessment.score,
        rule_score=assessment.rule_score,
        ml_score=assessment.ml_score,
        reasons=assessment.reasons,
        message="Transaction completed.",
    )


@router.post("/verify-challenge", response_model=ChallengeVerifyResponse)
async def verify_challenge(
    challenge_token: str = Form(...),
    face: UploadFile = File(...),
    request: Request = None,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Complete a CHALLENGE transaction with face re-verification."""
    pending = _pending_challenges.get(challenge_token)
    if pending is None or pending["user_id"] != user.id:
        raise HTTPException(status_code=404, detail="Challenge not found or expired")

    if not user.face_embedding:
        raise HTTPException(status_code=400, detail="No enrolled face on file")

    face_path = _save_face(face, user.id)
    stored = FaceEmbedding(user_id=str(user.id), embedding=user.face_embedding, model_name="Facenet512")
    result = biometric.verify(stored, str(face_path), require_liveness=True)

    txn = db.query(Transaction).filter(Transaction.id == pending["transaction_id"]).first()
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    chain = AuditChain(db, llm=llm)

    if not result.verified:
        txn.status = "denied"
        txn.decision = "DENY"
        db.commit()
        await chain.append(
            event_type="challenge_verify_failed",
            decision="DENY",
            user_id=user.id,
            risk_score=txn.risk_score,
            biometric_confidence=result.confidence,
            ip_address=request.client.host if request and request.client else "unknown",
            context={"transaction_id": txn.id, "distance": result.distance},
        )
        _pending_challenges.pop(challenge_token, None)
        return ChallengeVerifyResponse(
            transaction_id=txn.id,
            status="denied",
            biometric_distance=result.distance,
            biometric_confidence=result.confidence,
            message="Face verification failed. Transaction blocked.",
        )

    # Face matched — execute transfer
    source = db.query(Account).filter(Account.id == txn.from_account_id).first()
    dest = db.query(Account).filter(Account.id == txn.to_account_id).first() if txn.to_account_id else None
    if source is None or float(source.balance) < float(txn.amount):
        raise HTTPException(status_code=400, detail="Insufficient balance at execution time")

    source.balance = float(source.balance) - float(txn.amount)
    if dest:
        dest.balance = float(dest.balance) + float(txn.amount)
    txn.status = "completed"
    txn.decision = "ALLOW"
    txn.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    await chain.append(
        event_type="challenge_verify_passed",
        decision="ALLOW",
        user_id=user.id,
        risk_score=txn.risk_score,
        biometric_confidence=result.confidence,
        ip_address=request.client.host if request and request.client else "unknown",
        context={"transaction_id": txn.id, "distance": result.distance},
    )
    _pending_challenges.pop(challenge_token, None)

    return ChallengeVerifyResponse(
        transaction_id=txn.id,
        status="completed",
        biometric_distance=result.distance,
        biometric_confidence=result.confidence,
        message="Face verified. Transaction completed.",
    )