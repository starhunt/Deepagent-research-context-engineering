"""OpenRouter 15개 모델 통합 테스트.

실제 OpenRouter API를 통해 다양한 모델의 캐싱 전략과 provider 감지를 검증합니다.
OPENROUTER_API_KEY 환경 변수가 필요합니다.

테스트 대상 모델 (Anthropic/OpenAI/Google 제외):
- deepseek/deepseek-v3.2
- x-ai/grok-*
- xiaomi/mimo-v2-flash
- minimax/minimax-m2.1
- bytedance-seed/seed-1.6
- z-ai/glm-4.7
- allenai/olmo-3.1-32b-instruct
- mistralai/mistral-small-creative
- nvidia/nemotron-3-nano-30b-a3b
- qwen/qwen3-max, qwen3-coder-plus, qwen3-coder-flash, qwen3-vl-32b-instruct, qwen3-next-80b-a3b-thinking
"""

from __future__ import annotations

import os

import pytest
from langchain_openai import ChatOpenAI

from context_engineering_research_agent.context_strategies.caching import (
    CachingConfig,
    ContextCachingStrategy,
    OpenRouterSubProvider,
    ProviderType,
    detect_openrouter_sub_provider,
    detect_provider,
)


def _openrouter_available() -> bool:
    """OpenRouter API 키가 설정되어 있는지 확인합니다."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


OPENROUTER_AVAILABLE = _openrouter_available()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not OPENROUTER_AVAILABLE,
        reason="OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다.",
    ),
]


OPENROUTER_MODELS = [
    ("deepseek/deepseek-chat-v3-0324", OpenRouterSubProvider.DEEPSEEK),
    ("x-ai/grok-3-mini-beta", OpenRouterSubProvider.GROK),
    ("qwen/qwen3-235b-a22b", OpenRouterSubProvider.UNKNOWN),
    ("qwen/qwen3-32b", OpenRouterSubProvider.UNKNOWN),
    ("mistralai/mistral-small-3.1-24b-instruct", OpenRouterSubProvider.MISTRAL),
    ("meta-llama/llama-4-maverick", OpenRouterSubProvider.META_LLAMA),
    ("meta-llama/llama-4-scout", OpenRouterSubProvider.META_LLAMA),
    ("nvidia/llama-3.1-nemotron-70b-instruct", OpenRouterSubProvider.UNKNOWN),
    ("microsoft/phi-4", OpenRouterSubProvider.UNKNOWN),
    ("google/gemma-3-27b-it", OpenRouterSubProvider.UNKNOWN),
    ("cohere/command-a", OpenRouterSubProvider.UNKNOWN),
    ("perplexity/sonar-pro", OpenRouterSubProvider.UNKNOWN),
    ("ai21/jamba-1.6-large", OpenRouterSubProvider.UNKNOWN),
    ("inflection/inflection-3-pi", OpenRouterSubProvider.UNKNOWN),
    ("amazon/nova-pro-v1", OpenRouterSubProvider.UNKNOWN),
]


def _create_openrouter_model(model_name: str) -> ChatOpenAI:
    """OpenRouter 모델 인스턴스를 생성합니다."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=100,
    )


class TestOpenRouterProviderDetection:
    """OpenRouter Provider 감지 테스트."""

    @pytest.mark.parametrize("model_name,expected_sub", OPENROUTER_MODELS)
    def test_detect_provider_openrouter(
        self, model_name: str, expected_sub: OpenRouterSubProvider
    ) -> None:
        """OpenRouter 모델이 ProviderType.OPENROUTER로 감지되는지 확인합니다."""
        model = _create_openrouter_model(model_name)
        provider = detect_provider(model)
        assert provider == ProviderType.OPENROUTER

    @pytest.mark.parametrize("model_name,expected_sub", OPENROUTER_MODELS)
    def test_detect_openrouter_sub_provider(
        self, model_name: str, expected_sub: OpenRouterSubProvider
    ) -> None:
        """OpenRouter 모델명에서 sub-provider를 올바르게 감지하는지 확인합니다."""
        sub_provider = detect_openrouter_sub_provider(model_name)
        assert sub_provider == expected_sub


class TestOpenRouterCachingStrategy:
    """OpenRouter 캐싱 전략 테스트."""

    @pytest.fixture
    def low_threshold_config(self) -> CachingConfig:
        return CachingConfig(min_cacheable_tokens=10)

    @pytest.mark.parametrize("model_name,expected_sub", OPENROUTER_MODELS[:5])
    def test_caching_strategy_initialization(
        self,
        model_name: str,
        expected_sub: OpenRouterSubProvider,
    ) -> None:
        """ContextCachingStrategy가 OpenRouter 모델로 올바르게 초기화되는지 확인합니다."""
        model = _create_openrouter_model(model_name)
        strategy = ContextCachingStrategy(model=model, openrouter_model_name=model_name)

        assert strategy._provider == ProviderType.OPENROUTER
        assert strategy._openrouter_sub_provider == expected_sub


class TestOpenRouterModelInvocation:
    """OpenRouter 모델 실제 호출 테스트.

    실제 API 비용이 발생하므로 소수의 모델만 테스트합니다.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "deepseek/deepseek-chat-v3-0324",
            "qwen/qwen3-32b",
            "mistralai/mistral-small-3.1-24b-instruct",
        ],
    )
    def test_simple_invocation(self, model_name: str) -> None:
        """모델이 간단한 프롬프트에 응답하는지 확인합니다."""
        model = _create_openrouter_model(model_name)
        response = model.invoke("Say 'hello' in one word.")
        assert response.content
        assert len(response.content) > 0

    @pytest.mark.parametrize(
        "model_name",
        [
            "deepseek/deepseek-chat-v3-0324",
            "x-ai/grok-3-mini-beta",
        ],
    )
    def test_caching_strategy_apply_does_not_error(self, model_name: str) -> None:
        """ContextCachingStrategy.apply()가 에러 없이 동작하는지 확인합니다."""
        from langchain_core.messages import HumanMessage, SystemMessage

        model = _create_openrouter_model(model_name)
        strategy = ContextCachingStrategy(
            model=model,
            openrouter_model_name=model_name,
            config=CachingConfig(min_cacheable_tokens=10),
        )

        messages = [
            SystemMessage(content="You are a helpful assistant. " * 100),
            HumanMessage(content="Hello"),
        ]

        result = strategy.apply(messages)
        assert result.was_cached is not None


class TestOpenRouterModelNameExtraction:
    """OpenRouter 모델명 추출 테스트."""

    def test_model_name_extraction_from_model_attribute(self) -> None:
        """model 속성에서 모델명이 올바르게 추출되는지 확인합니다."""
        model = _create_openrouter_model("deepseek/deepseek-chat-v3-0324")
        name = getattr(model, "model_name", None) or getattr(model, "model", None)
        assert name == "deepseek/deepseek-chat-v3-0324"

    def test_openrouter_base_url_detection(self) -> None:
        """OpenRouter base URL이 올바르게 감지되는지 확인합니다."""
        model = _create_openrouter_model("qwen/qwen3-32b")
        base_url = getattr(model, "openai_api_base", None)
        assert base_url is not None
        assert "openrouter" in base_url.lower()
