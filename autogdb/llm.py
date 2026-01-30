from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4-1106-preview"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 1.0
    streaming: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        streaming: Optional[bool] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "LLMConfig":
        import os

        resolved_provider = provider or os.getenv("LLM_PROVIDER") or "openai"
        resolved_model = model or os.getenv("LLM_MODEL") or "gpt-4-1106-preview"
        resolved_api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_api_base = (
            api_base
            or os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )
        resolved_temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "1"))
        )
        resolved_streaming = (
            streaming
            if streaming is not None
            else os.getenv("LLM_STREAMING", "true").lower() == "true"
        )
        resolved_extra_kwargs = extra_kwargs or {}
        return cls(
            provider=resolved_provider,
            model=resolved_model,
            api_key=resolved_api_key,
            api_base=resolved_api_base,
            temperature=resolved_temperature,
            streaming=resolved_streaming,
            extra_kwargs=resolved_extra_kwargs,
        )


def create_llm(config: LLMConfig, *, callbacks=None):
    provider = (config.provider or "openai").lower()
    if provider == "openai":
        from langchain.chat_models.openai import ChatOpenAI

        return ChatOpenAI(
            temperature=config.temperature,
            model_name=config.model,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base,
            streaming=config.streaming,
            callbacks=callbacks,
            **config.extra_kwargs,
        )
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires langchain-google-genai. "
                "Install it or switch LLM_PROVIDER=openai."
            ) from exc
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            streaming=config.streaming,
            callbacks=callbacks,
            **config.extra_kwargs,
        )
    if provider in {"qwen", "tongyi"}:
        try:
            from langchain_community.chat_models import ChatTongyi
        except ImportError as exc:
            raise ImportError(
                "Qwen provider requires langchain-community ChatTongyi. "
                "Install it or switch LLM_PROVIDER=openai."
            ) from exc
        return ChatTongyi(
            model_name=config.model,
            dashscope_api_key=config.api_key,
            temperature=config.temperature,
            streaming=config.streaming,
            callbacks=callbacks,
            **config.extra_kwargs,
        )
    if provider in {"glm", "zhipu", "chatglm"}:
        try:
            from langchain_community.chat_models import ChatZhipuAI
        except ImportError as exc:
            raise ImportError(
                "Zhipu provider requires langchain-community ChatZhipuAI. "
                "Install it or switch LLM_PROVIDER=openai."
            ) from exc
        return ChatZhipuAI(
            model_name=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            streaming=config.streaming,
            callbacks=callbacks,
            **config.extra_kwargs,
        )
    raise ValueError(
        f"Unsupported LLM provider '{config.provider}'. "
        "Set LLM_PROVIDER to openai, gemini, qwen, or glm."
    )
