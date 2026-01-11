"""Context Caching 전략 구현.

Multi-Provider Prompt Caching 전략을 구현합니다.
각 Provider별 특성에 맞게 캐싱을 적용합니다:

## Direct Provider Access
- Anthropic: Explicit caching (cache_control 마커 필요), Write 1.25x, Read 0.1x
- OpenAI: Automatic caching (자동), Read 0.5x, 1024+ tokens
- Gemini 2.5: Implicit caching (자동), Read 0.25x, 1028-2048+ tokens
- Gemini 3: Implicit caching (자동), Read 0.1x (90% 할인), 1024-4096+ tokens

## OpenRouter (Multi-Model Gateway)
OpenRouter는 기반 모델에 따라 다른 caching 방식 적용:
- Anthropic Claude: Explicit (cache_control 필요), Write 1.25x, Read 0.1x
- OpenAI: Automatic (자동), Read 0.5x
- Google Gemini: Implicit + Explicit 지원, Read 0.25x
- DeepSeek: Automatic (자동), Write 1x, Read 0.1x
- Groq (Kimi K2): Automatic (자동), Read 0.25x
- Grok (xAI): Automatic (자동), Read 0.25x

DeepAgents의 AnthropicPromptCachingMiddleware와 함께 사용 권장.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage


class ProviderType(Enum):
    """LLM Provider 유형."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    GEMINI_3 = "gemini_3"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    GROK = "grok"
    UNKNOWN = "unknown"


class OpenRouterSubProvider(Enum):
    """OpenRouter를 통해 접근하는 기반 모델 Provider."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    GROK = "grok"
    META_LLAMA = "meta-llama"
    MISTRAL = "mistral"
    UNKNOWN = "unknown"


PROVIDERS_REQUIRING_CACHE_CONTROL = {
    ProviderType.ANTHROPIC,
    OpenRouterSubProvider.ANTHROPIC,
}

PROVIDERS_WITH_AUTOMATIC_CACHING = {
    ProviderType.OPENAI,
    ProviderType.GEMINI,
    ProviderType.GEMINI_3,
    ProviderType.DEEPSEEK,
    ProviderType.GROQ,
    ProviderType.GROK,
    OpenRouterSubProvider.OPENAI,
    OpenRouterSubProvider.DEEPSEEK,
    OpenRouterSubProvider.GROQ,
    OpenRouterSubProvider.GROK,
}


def detect_provider(model: BaseChatModel | None) -> ProviderType:
    """모델 객체에서 Provider 유형을 감지합니다."""
    if model is None:
        return ProviderType.UNKNOWN

    class_name = model.__class__.__name__.lower()
    module_name = model.__class__.__module__.lower()

    if "anthropic" in class_name or "anthropic" in module_name:
        return ProviderType.ANTHROPIC

    if "openai" in class_name or "openai" in module_name:
        base_url = _get_base_url(model)
        if "openrouter" in base_url:
            return ProviderType.OPENROUTER
        return ProviderType.OPENAI

    if "google" in class_name or "gemini" in class_name or "google" in module_name:
        model_name = _get_model_name(model)
        if "gemini-3" in model_name or "gemini/3" in model_name:
            return ProviderType.GEMINI_3
        return ProviderType.GEMINI

    if "deepseek" in class_name or "deepseek" in module_name:
        return ProviderType.DEEPSEEK

    if "groq" in class_name or "groq" in module_name:
        return ProviderType.GROQ

    return ProviderType.UNKNOWN


def _get_base_url(model: BaseChatModel) -> str:
    for attr in ("openai_api_base", "base_url", "api_base"):
        if hasattr(model, attr):
            url = getattr(model, attr, "") or ""
            if url:
                return url.lower()
    return ""


def _get_model_name(model: BaseChatModel) -> str:
    for attr in ("model_name", "model", "model_id"):
        if hasattr(model, attr):
            name = getattr(model, attr, "") or ""
            if name:
                return name.lower()
    return ""


def detect_openrouter_sub_provider(model_name: str) -> OpenRouterSubProvider:
    """OpenRouter 모델명에서 기반 Provider를 감지합니다.

    OpenRouter 모델명 패턴: "provider/model-name" (예: "anthropic/claude-3-sonnet")
    """
    name_lower = model_name.lower()

    if "anthropic" in name_lower or "claude" in name_lower:
        return OpenRouterSubProvider.ANTHROPIC
    if "openai" in name_lower or "gpt" in name_lower or name_lower.startswith("o1"):
        return OpenRouterSubProvider.OPENAI
    if "google" in name_lower or "gemini" in name_lower:
        return OpenRouterSubProvider.GEMINI
    if "deepseek" in name_lower:
        return OpenRouterSubProvider.DEEPSEEK
    if "groq" in name_lower or "kimi" in name_lower:
        return OpenRouterSubProvider.GROQ
    if "grok" in name_lower or "xai" in name_lower:
        return OpenRouterSubProvider.GROK
    if "meta" in name_lower or "llama" in name_lower:
        return OpenRouterSubProvider.META_LLAMA
    if "mistral" in name_lower:
        return OpenRouterSubProvider.MISTRAL

    return OpenRouterSubProvider.UNKNOWN


def requires_cache_control_marker(
    provider: ProviderType,
    sub_provider: OpenRouterSubProvider | None = None,
) -> bool:
    """해당 Provider가 cache_control 마커를 필요로 하는지 확인합니다.

    Anthropic (직접 또는 OpenRouter 경유) 만 True 반환.
    """
    if provider == ProviderType.ANTHROPIC:
        return True
    if (
        provider == ProviderType.OPENROUTER
        and sub_provider == OpenRouterSubProvider.ANTHROPIC
    ):
        return True
    return False


@dataclass
class CachingConfig:
    """Context Caching 설정.

    Attributes:
        min_cacheable_tokens: 캐싱할 최소 토큰 수 (기본값: 1024)
        cache_control_type: 캐시 제어 유형 (기본값: "ephemeral")
        enable_for_system_prompt: 시스템 프롬프트 캐싱 활성화 (기본값: True)
        enable_for_tools: 도구 정의 캐싱 활성화 (기본값: True)
    """

    min_cacheable_tokens: int = 1024
    cache_control_type: str = "ephemeral"
    enable_for_system_prompt: bool = True
    enable_for_tools: bool = True


@dataclass
class CachingResult:
    """Context Caching 결과.

    Attributes:
        was_cached: 캐싱이 적용되었는지 여부
        cached_content_type: 캐싱된 컨텐츠 유형 (예: "system_prompt")
        estimated_tokens_cached: 캐싱된 추정 토큰 수
    """

    was_cached: bool
    cached_content_type: str | None = None
    estimated_tokens_cached: int = 0


class ContextCachingStrategy(AgentMiddleware):
    """Multi-Provider Prompt Caching 전략.

    Anthropic (직접 또는 OpenRouter 경유)만 cache_control 마커를 적용하고,
    OpenAI/Gemini/DeepSeek/Groq/Grok은 자동 캐싱이므로 pass-through합니다.
    """

    def __init__(
        self,
        config: CachingConfig | None = None,
        model: BaseChatModel | None = None,
        openrouter_model_name: str | None = None,
    ) -> None:
        self.config = config or CachingConfig()
        self._model = model
        self._provider: ProviderType | None = None
        self._sub_provider: OpenRouterSubProvider | None = None
        self._openrouter_model_name = openrouter_model_name

    def set_model(
        self,
        model: BaseChatModel,
        openrouter_model_name: str | None = None,
    ) -> None:
        """런타임에 모델을 설정합니다."""
        self._model = model
        self._provider = None
        self._sub_provider = None
        if openrouter_model_name:
            self._openrouter_model_name = openrouter_model_name

    @property
    def provider(self) -> ProviderType:
        if self._provider is None:
            self._provider = detect_provider(self._model)
        return self._provider

    @property
    def sub_provider(self) -> OpenRouterSubProvider | None:
        if self.provider != ProviderType.OPENROUTER:
            return None
        if self._sub_provider is None and self._openrouter_model_name:
            self._sub_provider = detect_openrouter_sub_provider(
                self._openrouter_model_name
            )
        return self._sub_provider

    @property
    def should_apply_cache_markers(self) -> bool:
        return requires_cache_control_marker(self.provider, self.sub_provider)

    def _add_cache_control(self, content: Any) -> Any:
        if isinstance(content, str):
            return {
                "type": "text",
                "text": content,
                "cache_control": {"type": self.config.cache_control_type},
            }
        elif isinstance(content, dict) and content.get("type") == "text":
            return {
                **content,
                "cache_control": {"type": self.config.cache_control_type},
            }
        elif isinstance(content, list):
            if not content:
                return content
            result = list(content)
            last_item = result[-1]
            if isinstance(last_item, dict):
                result[-1] = {
                    **last_item,
                    "cache_control": {"type": self.config.cache_control_type},
                }
            return result
        return content

    def _process_system_message(self, message: SystemMessage) -> SystemMessage:
        cached_content = self._add_cache_control(message.content)
        # Ensure cached_content is a list of dicts for SystemMessage compatibility
        if isinstance(cached_content, str):
            cached_content = [{"type": "text", "text": cached_content}]
        elif isinstance(cached_content, dict):
            cached_content = [cached_content]
        # Type ignore is needed due to complex SystemMessage content type requirements
        return SystemMessage(content=cached_content)  # type: ignore[arg-type]

    def _estimate_tokens(self, content: Any) -> int:
        if isinstance(content, str):
            return len(content) // 4
        elif isinstance(content, list):
            return sum(self._estimate_tokens(item) for item in content)
        elif isinstance(content, dict):
            return self._estimate_tokens(content.get("text", ""))
        return 0

    def _should_cache(self, content: Any) -> bool:
        estimated_tokens = self._estimate_tokens(content)
        return estimated_tokens >= self.config.min_cacheable_tokens

    def apply_caching(
        self,
        messages: list[BaseMessage],
        model: BaseChatModel | None = None,
        openrouter_model_name: str | None = None,
    ) -> tuple[list[BaseMessage], CachingResult]:
        """메시지 리스트에 캐싱을 적용합니다.

        Anthropic (직접 또는 OpenRouter 경유)만 cache_control 마커를 적용합니다.
        다른 provider는 자동 캐싱이므로 메시지를 변형하지 않습니다.

        Args:
            messages: 처리할 메시지 리스트
            model: LLM 모델 (Provider 감지용)
            openrouter_model_name: OpenRouter 사용 시 모델명 (예: "anthropic/claude-3-sonnet")

        Returns:
            캐싱이 적용된 메시지 리스트와 캐싱 결과 튜플
        """
        if model is not None:
            self.set_model(model, openrouter_model_name)

        if not messages:
            return messages, CachingResult(was_cached=False)

        if not self.should_apply_cache_markers:
            provider_info = self.provider.value
            if self.provider == ProviderType.OPENROUTER and self.sub_provider:
                provider_info = f"openrouter/{self.sub_provider.value}"
            return messages, CachingResult(
                was_cached=False,
                cached_content_type=f"auto_cached_by_{provider_info}",
            )

        result_messages = list(messages)
        cached = False
        cached_type = None
        tokens_cached = 0

        for i, msg in enumerate(result_messages):
            if isinstance(msg, SystemMessage):
                if self.config.enable_for_system_prompt and self._should_cache(
                    msg.content
                ):
                    result_messages[i] = self._process_system_message(msg)
                    cached = True
                    cached_type = "system_prompt"
                    tokens_cached = self._estimate_tokens(msg.content)
                    break

        return result_messages, CachingResult(
            was_cached=cached,
            cached_content_type=cached_type,
            estimated_tokens_cached=tokens_cached,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """동기 모델 호출을 래핑합니다 (기본 동작).

        Args:
            request: 모델 요청
            handler: 다음 핸들러 함수

        Returns:
            모델 응답
        """
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """비동기 모델 호출을 래핑합니다 (기본 동작).

        Args:
            request: 모델 요청
            handler: 다음 핸들러 함수

        Returns:
            모델 응답
        """
        return await handler(request)


CACHING_SYSTEM_PROMPT = """## Context Caching

시스템 프롬프트와 도구 정의가 자동으로 캐싱됩니다.

이점:
- API 호출 비용 절감
- 응답 속도 향상
- 동일 세션 내 반복 호출 최적화
"""
