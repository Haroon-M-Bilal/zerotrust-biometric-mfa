"""
LLM Explainer — async client for the locally-hosted Qwen 2.5 14B Instruct model
served by LM Studio on the OpenAI-compatible /v1/chat/completions endpoint.

This module is framework-agnostic: no FastAPI imports. It exposes a single
async class that any caller (route handler, background worker, CLI script)
can use to generate human-readable explanations of access-control decisions.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AccessDecisionContext(BaseModel):
    """Structured input describing the access event the LLM must explain."""

    user_id: str
    resource: str
    action: str
    decision: str = Field(..., description="ALLOW | DENY | CHALLENGE")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    biometric_confidence: float = Field(..., ge=0.0, le=1.0)
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual signals: ip, device, time_of_day, geo, etc.",
    )


class LLMExplanation(BaseModel):
    """Structured output returned to callers."""

    explanation: str
    model: str
    tokens_used: int | None = None


class LLMExplainerError(Exception):
    """Raised when the LLM call fails (timeout, network, malformed response)."""


class LLMExplainer:
    """Async client for LM Studio's OpenAI-compatible chat completions endpoint."""

    SYSTEM_PROMPT = (
        "You are a security audit assistant for a Zero-Trust access control system. "
        "Given an access decision and its supporting signals, produce a concise, "
        "factual, human-readable explanation (2-4 sentences). Do not speculate beyond "
        "the provided signals. Do not give advice. State what was decided and why."
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def explain(self, context: AccessDecisionContext) -> LLMExplanation:
        """Generate an explanation for the given access decision."""
        user_prompt = self._build_user_prompt(context)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            logger.exception("LLM request failed")
            raise LLMExplainerError(f"LLM request failed: {exc}") from exc

        try:
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens")
        except (KeyError, IndexError, AttributeError) as exc:
            logger.exception("Malformed LLM response: %s", data)
            raise LLMExplainerError(f"Malformed LLM response: {exc}") from exc

        return LLMExplanation(
            explanation=text,
            model=self._model,
            tokens_used=tokens,
        )

    async def health(self) -> bool:
        """Cheap probe: confirms LM Studio is reachable and the model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/models")
                response.raise_for_status()
                data = response.json()
                model_ids = {m.get("id") for m in data.get("data", [])}
                return self._model in model_ids
        except httpx.HTTPError:
            return False

    @staticmethod
    def _build_user_prompt(ctx: AccessDecisionContext) -> str:
        lines = [
            f"Decision: {ctx.decision}",
            f"User: {ctx.user_id}",
            f"Resource: {ctx.resource}",
            f"Action: {ctx.action}",
            f"Risk score: {ctx.risk_score:.2f} (0=low, 1=high)",
            f"Biometric confidence: {ctx.biometric_confidence:.2f}",
        ]
        if ctx.signals:
            lines.append("Signals:")
            for key, value in ctx.signals.items():
                lines.append(f"  - {key}: {value}")
        lines.append("\nExplain this decision.")
        return "\n".join(lines)