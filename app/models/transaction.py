"""Transaction model — money movement record with risk metadata."""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Numeric, Float, ForeignKey

from app.db.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    from_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True, index=True)
    to_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True, index=True)
    amount = Column(Numeric(15, 2), nullable=False)
    currency = Column(String(3), nullable=False, default="USD")
    transaction_type = Column(String(32), nullable=False)  # transfer, deposit, withdrawal
    status = Column(String(16), nullable=False, default="pending")  # pending, completed, denied, blocked
    risk_score = Column(Float, nullable=True)
    decision = Column(String(16), nullable=True)  # ALLOW, CHALLENGE, DENY
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)