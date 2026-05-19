"""
JWT issuance + verification using python-jose (HS256).
Access tokens: short-lived (15 min). Refresh tokens: longer (7 days), separate type claim.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import JWTError, jwt
from pydantic import BaseModel

from config.settings import get_settings

settings = get_settings()


class TokenPayload(BaseModel):
    sub: str           # subject (user id as string)
    jti: str           # unique token id (for session tracking / revocation)
    type: str          # "access" or "refresh"
    iat: int           # issued-at (epoch seconds)
    exp: int           # expiry (epoch seconds)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(payload: dict[str, Any]) -> str:
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, extra_claims: dict[str, Any] | None = None) -> tuple[str, str]:
    """Returns (token, jti). jti is stored in the sessions table for revocation."""
    jti = str(uuid4())
    now = _now()
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": user_id,
        "jti": jti,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return _encode(payload), jti


def create_refresh_token(user_id: str) -> tuple[str, str]:
    """Returns (token, jti)."""
    jti = str(uuid4())
    now = _now()
    expire = now + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "jti": jti,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return _encode(payload), jti


def decode_token(token: str) -> TokenPayload | None:
    """Return decoded payload or None if invalid/expired."""
    try:
        raw = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return TokenPayload(**raw)
    except (JWTError, ValueError):
        return None


def verify_access_token(token: str) -> TokenPayload | None:
    """Decode and require type='access'. Returns None on any failure."""
    payload = decode_token(token)
    if payload is None or payload.type != "access":
        return None
    return payload


def verify_refresh_token(token: str) -> TokenPayload | None:
    payload = decode_token(token)
    if payload is None or payload.type != "refresh":
        return None
    return payload