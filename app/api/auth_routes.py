"""
Auth routes — registration, login, token refresh.
Wires together: User model, Argon2id password hashing, JWT issuance,
biometric enrollment, and Session tracking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session as DBSession

from app.core.biometric import BiometricService
from app.db.database import get_db
from app.models.account import Account
from app.models.session import Session as SessionModel
from app.models.user import User
from app.security.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from app.security.password_hasher import hash_password, verify_password
from config.settings import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()
biometric = BiometricService()

UPLOAD_DIR = Path("data") / "uploads" / "face_captures"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Pydantic schemas ----------

class RegisterResponse(BaseModel):
    user_id: int
    username: str
    account_number: str
    balance: float
    message: str


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------- Helpers ----------

def _save_uploaded_face(file: UploadFile, user_id: int) -> Path:
    """Save an uploaded face image to disk and return the path."""
    suffix = Path(file.filename or "face.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Image must be JPG or PNG")
    target = UPLOAD_DIR / f"user_{user_id}_{uuid.uuid4().hex[:8]}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return target


def _generate_account_number() -> str:
    return str(uuid.uuid4().int)[:10]


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ---------- Routes ----------

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    username: str = Form(..., min_length=3, max_length=64),
    email: EmailStr = Form(...),
    password: str = Form(..., min_length=6),
    face: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    """
    Register a new user: store hashed password, enroll face embedding,
    create a checking account with seed balance.
    """
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=409, detail="Username or email already registered")

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        is_active=True,
    )
    db.add(user)
    db.flush()  # get user.id without committing yet

    face_path = _save_uploaded_face(face, user.id)
    try:
        enrollment = biometric.enroll(user_id=str(user.id), image_path=str(face_path))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Face enrollment failed: {e}")
    user.face_embedding = enrollment.embedding

    account = Account(
        user_id=user.id,
        account_number=_generate_account_number(),
        account_type="checking",
        balance=10_000.00,  # demo seed balance
        currency="USD",
    )
    db.add(account)
    db.commit()
    db.refresh(user)
    db.refresh(account)

    return RegisterResponse(
        user_id=user.id,
        username=user.username,
        account_number=account.account_number,
        balance=float(account.balance),
        message="Registration successful",
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request, db: DBSession = Depends(get_db)):
    """Password-only login. Step-up face verification happens per-transaction."""
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    access_token, access_jti = create_access_token(str(user.id))
    refresh_token, refresh_jti = create_refresh_token(str(user.id))

    db.add(SessionModel(
        user_id=user.id,
        jti=access_jti,
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent", "")[:512],
        device_fingerprint=request.headers.get("x-device-fingerprint"),
        is_active=True,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes),
    ))
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshRequest, db: DBSession = Depends(get_db)):
    """Issue a new access token from a valid refresh token."""
    decoded = verify_refresh_token(payload.refresh_token)
    if decoded is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    user = db.query(User).filter(User.id == int(decoded.sub)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")
    access_token, _ = create_access_token(str(user.id))
    return TokenResponse(
        access_token=access_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )