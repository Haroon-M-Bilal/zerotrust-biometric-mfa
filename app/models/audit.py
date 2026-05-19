"""Audit log — hash-chained append-only record of access decisions."""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text

from app.db.database import Base


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)  # nullable for anonymous attempts
    event_type = Column(String(64), nullable=False, index=True)  # login, verify, transfer, deny, etc.
    decision = Column(String(16), nullable=False)  # ALLOW, CHALLENGE, DENY
    risk_score = Column(Float, nullable=True)
    biometric_confidence = Column(Float, nullable=True)
    ip_address = Column(String(45), nullable=True)
    context = Column(JSON, nullable=True)  # arbitrary structured context
    llm_explanation = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=True)  # SHA-256 of previous entry
    entry_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 of this entry