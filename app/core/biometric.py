"""
Biometric face verification service using DeepFace.
Enrollment captures a face embedding; verification compares a probe against it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import math

import numpy as np
from deepface import DeepFace
from pydantic import BaseModel, Field

from config.settings import get_settings


class FaceEmbedding(BaseModel):
    user_id: str
    embedding: List[float] = Field(..., description="Face embedding vector (Facenet512 = 512 dims)")
    model_name: str = "Facenet512"


class VerificationResult(BaseModel):
    verified: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    distance: float
    threshold: float


class BiometricService:
    """Framework-agnostic face verification. DB wiring comes later."""

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

    def enroll(self, user_id: str, image_path: str) -> FaceEmbedding:
        embedding = self._extract_embedding(image_path)
        return FaceEmbedding(user_id=user_id, embedding=embedding, model_name=self.model_name)

    def verify(self, stored: FaceEmbedding, probe_image_path: str) -> VerificationResult:
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
        )