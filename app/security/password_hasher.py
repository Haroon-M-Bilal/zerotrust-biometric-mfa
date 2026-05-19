"""
Argon2id password hashing.
Wraps argon2-cffi with safe defaults (m=64 MiB, t=3, p=4).
"""
from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# Defaults from OWASP Argon2id guidance: m=64MB, t=3, p=4
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Return an Argon2id hash string (includes salt + params)."""
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Return True if password matches the hash. Never raises."""
    try:
        return _hasher.verify(hashed, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """True if hash params are outdated and should be re-hashed on next login."""
    return _hasher.check_needs_rehash(hashed)