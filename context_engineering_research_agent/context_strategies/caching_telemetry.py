"""Prompt Caching Telemetry Middleware.

모든 Provider의 cache 사용량을 모니터링하고 로깅합니다.
요청 변형 없이 응답의 cache 관련 메타데이터만 수집합니다.

Provider별 캐시 메타데이터 위치:
- Anthropic: cache_read_input_tokens, cache_creation_input_tokens
- OpenAI: cached_tokens (usage.prompt_tokens_details)
- Gemini 2.5/3: cached_content_token_count
- DeepSeek: cache_hit_tokens, cache_miss_tokens
- OpenRouter: 기반 모델의 메타데이터 형식 따름
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from context_engineering_research_agent.context_strategies.caching import (
    ProviderType,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheTelemetry:
    """Provider별 캐시 사용량 데이터."""

    provider: ProviderType
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_input_tokens: int = 0
    cache_hit_ratio: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)


def extract_anthropic_cache_metrics(response: ModelResponse) -> CacheTelemetry:
    """Anthropic 응답에서 캐시 메트릭을 추출합니다."""
    usage = getattr(response, "usage_metadata", {}) or {}
    response_meta = getattr(response, "response_metadata", {}) or {}
    usage_from_meta = response_meta.get("usage", {})

    cache_read = usage_from_meta.get("cache_read_input_tokens", 0)
    cache_creation = usage_from_meta.get("cache_creation_input_tokens", 0)
    input_tokens = usage.get("input_tokens", 0) or usage_from_meta.get(
        "input_tokens", 0
    )

    hit_ratio = cache_read / input_tokens if input_tokens > 0 else 0.0

    return CacheTelemetry(
        provider=ProviderType.ANTHROPIC,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_creation,
        total_input_tokens=input_tokens,
        cache_hit_ratio=hit_ratio,
        raw_metadata={"usage": usage, "response_metadata": response_meta},
    )


def extract_openai_cache_metrics(response: ModelResponse) -> CacheTelemetry:
    """OpenAI 응답에서 캐시 메트릭을 추출합니다."""
    usage = getattr(response, "usage_metadata", {}) or {}
    response_meta = getattr(response, "response_metadata", {}) or {}

    token_usage = response_meta.get("token_usage", {})
    prompt_details = token_usage.get("prompt_tokens_details", {})
    cached_tokens = prompt_details.get("cached_tokens", 0)
    input_tokens = usage.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0)

    hit_ratio = cached_tokens / input_tokens if input_tokens > 0 else 0.0

    return CacheTelemetry(
        provider=ProviderType.OPENAI,
        cache_read_tokens=cached_tokens,
        cache_write_tokens=0,
        total_input_tokens=input_tokens,
        cache_hit_ratio=hit_ratio,
        raw_metadata={"usage": usage, "token_usage": token_usage},
    )


def extract_gemini_cache_metrics(
    response: ModelResponse, provider: ProviderType = ProviderType.GEMINI
) -> CacheTelemetry:
    """Gemini 2.5/3 응답에서 캐시 메트릭을 추출합니다."""
    usage = getattr(response, "usage_metadata", {}) or {}
    response_meta = getattr(response, "response_metadata", {}) or {}

    cached_tokens = response_meta.get("cached_content_token_count", 0)
    input_tokens = usage.get("input_tokens", 0) or response_meta.get(
        "prompt_token_count", 0
    )

    hit_ratio = cached_tokens / input_tokens if input_tokens > 0 else 0.0

    return CacheTelemetry(
        provider=provider,
        cache_read_tokens=cached_tokens,
        cache_write_tokens=0,
        total_input_tokens=input_tokens,
        cache_hit_ratio=hit_ratio,
        raw_metadata={"usage": usage, "response_metadata": response_meta},
    )


def extract_deepseek_cache_metrics(response: ModelResponse) -> CacheTelemetry:
    """DeepSeek 응답에서 캐시 메트릭을 추출합니다."""
    usage = getattr(response, "usage_metadata", {}) or {}
    response_meta = getattr(response, "response_metadata", {}) or {}

    cache_hit = response_meta.get("cache_hit_tokens", 0)
    cache_miss = response_meta.get("cache_miss_tokens", 0)
    input_tokens = usage.get("input_tokens", 0) or (cache_hit + cache_miss)

    hit_ratio = cache_hit / input_tokens if input_tokens > 0 else 0.0

    return CacheTelemetry(
        provider=ProviderType.DEEPSEEK,
        cache_read_tokens=cache_hit,
        cache_write_tokens=cache_miss,
        total_input_tokens=input_tokens,
        cache_hit_ratio=hit_ratio,
        raw_metadata={"usage": usage, "response_metadata": response_meta},
    )


def extract_cache_telemetry(
    response: ModelResponse, provider: ProviderType
) -> CacheTelemetry:
    """응답에서 Provider별 캐시 텔레메트리를 추출합니다."""
    extractors: dict[ProviderType, Callable[[ModelResponse], CacheTelemetry]] = {
        ProviderType.ANTHROPIC: extract_anthropic_cache_metrics,
        ProviderType.OPENAI: extract_openai_cache_metrics,
        ProviderType.GEMINI: extract_gemini_cache_metrics,
        ProviderType.GEMINI_3: lambda r: extract_gemini_cache_metrics(
            r, ProviderType.GEMINI_3
        ),
        ProviderType.DEEPSEEK: extract_deepseek_cache_metrics,
    }

    extractor = extractors.get(provider)
    if extractor:
        return extractor(response)

    return CacheTelemetry(
        provider=provider,
        raw_metadata={
            "usage": getattr(response, "usage_metadata", {}),
            "response_metadata": getattr(response, "response_metadata", {}),
        },
    )


class PromptCachingTelemetryMiddleware(AgentMiddleware):
    """모든 Provider의 캐시 사용량을 모니터링하는 Middleware.

    요청을 변형하지 않고, 응답의 cache 관련 메타데이터만 수집/로깅합니다.
    """

    def __init__(self, log_level: int = logging.DEBUG) -> None:
        self._log_level = log_level
        self._telemetry_history: list[CacheTelemetry] = []

    @property
    def telemetry_history(self) -> list[CacheTelemetry]:
        return self._telemetry_history

    def get_aggregate_stats(self) -> dict[str, Any]:
        if not self._telemetry_history:
            return {"total_calls": 0, "total_cache_read_tokens": 0}

        total_read = sum(t.cache_read_tokens for t in self._telemetry_history)
        total_write = sum(t.cache_write_tokens for t in self._telemetry_history)
        total_input = sum(t.total_input_tokens for t in self._telemetry_history)

        return {
            "total_calls": len(self._telemetry_history),
            "total_cache_read_tokens": total_read,
            "total_cache_write_tokens": total_write,
            "total_input_tokens": total_input,
            "overall_cache_hit_ratio": total_read / total_input if total_input else 0.0,
        }

    def _log_telemetry(self, telemetry: CacheTelemetry) -> None:
        if telemetry.cache_read_tokens > 0 or telemetry.cache_write_tokens > 0:
            logger.log(
                self._log_level,
                f"[CacheTelemetry] {telemetry.provider.value}: "
                f"read={telemetry.cache_read_tokens}, "
                f"write={telemetry.cache_write_tokens}, "
                f"hit_ratio={telemetry.cache_hit_ratio:.2%}",
            )

    def _process_response(self, response: ModelResponse) -> ModelResponse:
        model = getattr(response, "response_metadata", {}).get("model", "")
        provider = self._detect_provider_from_response(response, model)
        telemetry = extract_cache_telemetry(response, provider)
        self._telemetry_history.append(telemetry)
        self._log_telemetry(telemetry)
        return response

    def _detect_provider_from_response(
        self, response: ModelResponse, model_name: str
    ) -> ProviderType:
        model_lower = model_name.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return ProviderType.ANTHROPIC
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            return ProviderType.OPENAI
        if "gemini-3" in model_lower or "gemini/3" in model_lower:
            return ProviderType.GEMINI_3
        if "gemini" in model_lower:
            return ProviderType.GEMINI
        if "deepseek" in model_lower:
            return ProviderType.DEEPSEEK
        if "groq" in model_lower or "kimi" in model_lower:
            return ProviderType.GROQ
        if "grok" in model_lower:
            return ProviderType.GROK
        return ProviderType.UNKNOWN

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        response = handler(request)
        return self._process_response(response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await handler(request)
        return self._process_response(response)
