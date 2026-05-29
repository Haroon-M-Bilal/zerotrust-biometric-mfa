"""
Biometric face verification service using DeepFace.
Enrollment captures a face embedding; verification compares a probe against it.
Now with optional liveness (anti-spoofing) check using DeepFace's MiniFASNet.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from deepface import DeepFace
from pydantic import BaseModel, Field

from config.settings import get_settings


class FaceEmbedding(BaseModel):
    user_id: str
    embedding: List[float] = Field(..., description="Face embedding vector (Facenet512 = 512 dims)")
    model_name: str = "Facenet512"


class LivenessResult(BaseModel):
    is_real: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Anti-spoof model confidence")


class VerificationResult(BaseModel):
    verified: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    distance: float
    threshold: float
    liveness: LivenessResult | None = None


class BiometricError(Exception):
    """Raised when biometric processing fails (no face, spoof detected, etc.)."""


class BiometricService:
    """Framework-agnostic face verification with anti-spoofing."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_name = "Facenet512"
        self.detector_backend = "opencv"

    def _extract_embedding(self, image_path: str) -> List[float]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        reps = DeepFace.represent(
            img_path=str(path),
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=True,
        )
        if not reps:
            raise ValueError("No face detected in image")
        return reps[0]["embedding"]

    @staticmethod
    def _embedding_distance(a: List[float], b: List[float]) -> float:
        """Cosine distance: 0.0 = identical, 1.0 = orthogonal."""
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0:
            return 1.0
        cosine_sim = float(np.dot(va, vb) / denom)
        return 1.0 - cosine_sim

    def check_liveness(self, image_path: str) -> LivenessResult:
        """
        Detect spoofing attempts (printed photo, phone screen, mask).
        Returns LivenessResult with is_real=False if a spoof is suspected.
        Uses DeepFace's MiniFASNet anti-spoof model.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            results = DeepFace.extract_faces(
                img_path=str(path),
                detector_backend=self.detector_backend,
                anti_spoofing=True,
                enforce_detection=True,
            )
        except Exception as e:
            raise BiometricError(f"Liveness check failed: {e}")

        if not results:
            return LivenessResult(is_real=False, confidence=0.0)

        face = results[0]
        is_real = bool(face.get("is_real", False))
        antispoof_score = float(face.get("antispoof_score", 0.0))
        return LivenessResult(is_real=is_real, confidence=antispoof_score)

    def enroll(self, user_id: str, image_path: str, require_liveness: bool = False) -> FaceEmbedding:
        if require_liveness:
            liveness = self.check_liveness(image_path)
            if not liveness.is_real:
                raise BiometricError(
                    f"Liveness check failed during enrollment (confidence={liveness.confidence:.2f}). "
                    "Possible spoof attempt — printed photo, screen replay, or mask."
                )
        embedding = self._extract_embedding(image_path)
        return FaceEmbedding(user_id=user_id, embedding=embedding, model_name=self.model_name)

    def verify(
        self,
        stored: FaceEmbedding,
        probe_image_path: str,
        require_liveness: bool = False,
    ) -> VerificationResult:
        liveness_result: LivenessResult | None = None
        if require_liveness:
            liveness_result = self.check_liveness(probe_image_path)
            if not liveness_result.is_real:
                # Short-circuit: skip embedding comparison, return failed verification
                return VerificationResult(
                    verified=False,
                    confidence=0.0,
                    distance=1.0,
                    threshold=self.settings.biometric_match_threshold,
                    liveness=liveness_result,
                )

        probe_embedding = self._extract_embedding(probe_image_path)
        distance = self._embedding_distance(stored.embedding, probe_embedding)
        threshold = self.settings.biometric_match_threshold
        verified = distance <= threshold
        confidence = max(0.0, min(1.0, 1.0 - distance))
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            distance=distance,
            threshold=threshold,
            liveness=liveness_result,
        )