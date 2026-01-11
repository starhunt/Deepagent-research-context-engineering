from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from context_engineering_research_agent.context_strategies.caching import (
    CachingConfig,
    CachingResult,
    ContextCachingStrategy,
    OpenRouterSubProvider,
    ProviderType,
    detect_openrouter_sub_provider,
    detect_provider,
    requires_cache_control_marker,
)
from context_engineering_research_agent.context_strategies.caching_telemetry import (
    CacheTelemetry,
    PromptCachingTelemetryMiddleware,
    extract_anthropic_cache_metrics,
    extract_cache_telemetry,
    extract_deepseek_cache_metrics,
    extract_gemini_cache_metrics,
    extract_openai_cache_metrics,
)


class TestCachingConfig:
    def test_default_values(self):
        config = CachingConfig()

        assert config.min_cacheable_tokens == 1024
        assert config.cache_control_type == "ephemeral"
        assert config.enable_for_system_prompt is True
        assert config.enable_for_tools is True

    def test_custom_values(self):
        config = CachingConfig(
            min_cacheable_tokens=2048,
            cache_control_type="permanent",
            enable_for_system_prompt=False,
            enable_for_tools=False,
        )

        assert config.min_cacheable_tokens == 2048
        assert config.cache_control_type == "permanent"
        assert config.enable_for_system_prompt is False
        assert config.enable_for_tools is False


class TestCachingResult:
    def test_not_cached(self):
        result = CachingResult(was_cached=False)

        assert result.was_cached is False
        assert result.cached_content_type is None
        assert result.estimated_tokens_cached == 0

    def test_cached(self):
        result = CachingResult(
            was_cached=True,
            cached_content_type="system_prompt",
            estimated_tokens_cached=5000,
        )

        assert result.was_cached is True
        assert result.cached_content_type == "system_prompt"
        assert result.estimated_tokens_cached == 5000


class TestContextCachingStrategy:
    @pytest.fixture
    def mock_anthropic_model(self) -> MagicMock:
        mock = MagicMock()
        mock.__class__.__name__ = "ChatAnthropic"
        mock.__class__.__module__ = "langchain_anthropic"
        return mock

    @pytest.fixture
    def strategy(self, mock_anthropic_model: MagicMock) -> ContextCachingStrategy:
        return ContextCachingStrategy(model=mock_anthropic_model)

    @pytest.fixture
    def strategy_low_threshold(
        self, mock_anthropic_model: MagicMock
    ) -> ContextCachingStrategy:
        return ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_anthropic_model,
        )

    def test_estimate_tokens_string(self, strategy: ContextCachingStrategy):
        content = "a" * 400
        estimated = strategy._estimate_tokens(content)

        assert estimated == 100

    def test_estimate_tokens_list(self, strategy: ContextCachingStrategy):
        content = [
            {"type": "text", "text": "a" * 200},
            {"type": "text", "text": "b" * 200},
        ]
        estimated = strategy._estimate_tokens(content)

        assert estimated == 100

    def test_estimate_tokens_dict(self, strategy: ContextCachingStrategy):
        content = {"type": "text", "text": "a" * 400}
        estimated = strategy._estimate_tokens(content)

        assert estimated == 100

    def test_should_cache_small_content(self, strategy: ContextCachingStrategy):
        small_content = "short text"

        assert strategy._should_cache(small_content) is False

    def test_should_cache_large_content(self, strategy: ContextCachingStrategy):
        large_content = "x" * 5000

        assert strategy._should_cache(large_content) is True

    def test_add_cache_control_string(self, strategy: ContextCachingStrategy):
        content = "test content"
        cached = strategy._add_cache_control(content)

        assert cached["type"] == "text"
        assert cached["text"] == "test content"
        assert cached["cache_control"]["type"] == "ephemeral"

    def test_add_cache_control_dict(self, strategy: ContextCachingStrategy):
        content = {"type": "text", "text": "test content"}
        cached = strategy._add_cache_control(content)

        assert cached["cache_control"]["type"] == "ephemeral"

    def test_add_cache_control_list(self, strategy: ContextCachingStrategy):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        cached = strategy._add_cache_control(content)

        assert "cache_control" not in cached[0]
        assert cached[1]["cache_control"]["type"] == "ephemeral"

    def test_add_cache_control_empty_list(self, strategy: ContextCachingStrategy):
        content: list = []
        cached = strategy._add_cache_control(content)

        assert cached == []

    def test_process_system_message(self, strategy: ContextCachingStrategy):
        message = SystemMessage(content="You are a helpful assistant")
        processed = strategy._process_system_message(message)

        assert isinstance(processed, SystemMessage)
        assert isinstance(processed.content, list)
        assert processed.content[0]["cache_control"]["type"] == "ephemeral"

    def test_apply_caching_empty_messages(self, strategy: ContextCachingStrategy):
        messages: list = []
        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False
        assert cached == []

    def test_apply_caching_no_system_message(self, strategy: ContextCachingStrategy):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False

    def test_apply_caching_small_system_message(self, strategy: ContextCachingStrategy):
        messages = [
            SystemMessage(content="Be helpful"),
            HumanMessage(content="Hello"),
        ]
        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False

    def test_apply_caching_large_system_message(
        self, strategy_low_threshold: ContextCachingStrategy
    ):
        large_prompt = "You are a helpful assistant. " * 50
        messages = [
            SystemMessage(content=large_prompt),
            HumanMessage(content="Hello"),
        ]
        cached, result = strategy_low_threshold.apply_caching(messages)

        assert result.was_cached is True
        assert result.cached_content_type == "system_prompt"
        assert result.estimated_tokens_cached > 0

    def test_apply_caching_preserves_message_order(
        self, strategy_low_threshold: ContextCachingStrategy
    ):
        large_prompt = "System prompt " * 100
        messages = [
            SystemMessage(content=large_prompt),
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
        ]
        cached, _ = strategy_low_threshold.apply_caching(messages)

        assert len(cached) == 4
        assert isinstance(cached[0], SystemMessage)
        assert isinstance(cached[1], HumanMessage)
        assert isinstance(cached[2], AIMessage)
        assert isinstance(cached[3], HumanMessage)

    def test_custom_cache_control_type(self):
        strategy = ContextCachingStrategy(
            config=CachingConfig(
                cache_control_type="permanent",
                min_cacheable_tokens=10,
            )
        )
        content = "test content"
        cached = strategy._add_cache_control(content)

        assert cached["cache_control"]["type"] == "permanent"

    def test_disabled_system_prompt_caching(self):
        strategy = ContextCachingStrategy(
            config=CachingConfig(
                enable_for_system_prompt=False,
                min_cacheable_tokens=10,
            )
        )
        large_prompt = "System prompt " * 100
        messages = [
            SystemMessage(content=large_prompt),
            HumanMessage(content="Hello"),
        ]
        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False


class TestProviderDetection:
    def test_detect_anthropic_from_class_name(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic.chat_models"

        assert detect_provider(mock_model) == ProviderType.ANTHROPIC

    def test_detect_openai_from_class_name(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai.chat_models"
        mock_model.openai_api_base = None

        assert detect_provider(mock_model) == ProviderType.OPENAI

    def test_detect_gemini_from_class_name(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"

        assert detect_provider(mock_model) == ProviderType.GEMINI

    def test_detect_openrouter_from_base_url(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"
        mock_model.openai_api_base = "https://openrouter.ai/api/v1"

        assert detect_provider(mock_model) == ProviderType.OPENROUTER

    def test_detect_unknown_provider(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "CustomModel"
        mock_model.__class__.__module__ = "custom_module"

        assert detect_provider(mock_model) == ProviderType.UNKNOWN

    def test_detect_none_model(self):
        assert detect_provider(None) == ProviderType.UNKNOWN

    def test_requires_cache_control_marker_anthropic(self):
        assert requires_cache_control_marker(ProviderType.ANTHROPIC) is True

    def test_requires_cache_control_marker_openai(self):
        assert requires_cache_control_marker(ProviderType.OPENAI) is False

    def test_requires_cache_control_marker_gemini(self):
        assert requires_cache_control_marker(ProviderType.GEMINI) is False


class TestContextCachingStrategyMultiProvider:
    def test_anthropic_applies_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatAnthropic"
        mock_model.__class__.__module__ = "langchain_anthropic"

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
        )
        large_prompt = "System prompt " * 100
        messages = [SystemMessage(content=large_prompt)]

        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is True
        assert result.cached_content_type == "system_prompt"

    def test_openai_skips_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"
        mock_model.openai_api_base = None

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
        )
        large_prompt = "System prompt " * 100
        messages = [SystemMessage(content=large_prompt)]

        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False
        assert result.cached_content_type == "auto_cached_by_openai"

    def test_gemini_skips_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
        )
        large_prompt = "System prompt " * 100
        messages = [SystemMessage(content=large_prompt)]

        cached, result = strategy.apply_caching(messages)

        assert result.was_cached is False
        assert result.cached_content_type == "auto_cached_by_gemini"

    def test_set_model_updates_provider(self):
        strategy = ContextCachingStrategy()

        mock_anthropic = MagicMock()
        mock_anthropic.__class__.__name__ = "ChatAnthropic"
        mock_anthropic.__class__.__module__ = "langchain_anthropic"

        strategy.set_model(mock_anthropic)
        assert strategy.provider == ProviderType.ANTHROPIC

        mock_openai = MagicMock()
        mock_openai.__class__.__name__ = "ChatOpenAI"
        mock_openai.__class__.__module__ = "langchain_openai"
        mock_openai.openai_api_base = None

        strategy.set_model(mock_openai)
        assert strategy.provider == ProviderType.OPENAI


class TestCacheTelemetry:
    def test_default_values(self):
        telemetry = CacheTelemetry(provider=ProviderType.OPENAI)

        assert telemetry.cache_read_tokens == 0
        assert telemetry.cache_write_tokens == 0
        assert telemetry.cache_hit_ratio == 0.0

    def test_with_values(self):
        telemetry = CacheTelemetry(
            provider=ProviderType.ANTHROPIC,
            cache_read_tokens=1000,
            cache_write_tokens=500,
            total_input_tokens=2000,
            cache_hit_ratio=0.5,
        )

        assert telemetry.cache_read_tokens == 1000
        assert telemetry.cache_write_tokens == 500
        assert telemetry.cache_hit_ratio == 0.5


class TestCacheMetricsExtraction:
    def test_extract_anthropic_metrics(self):
        mock_response = MagicMock()
        mock_response.usage_metadata = {"input_tokens": 1000}
        mock_response.response_metadata = {
            "usage": {
                "input_tokens": 1000,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 200,
            }
        }

        telemetry = extract_anthropic_cache_metrics(mock_response)

        assert telemetry.provider == ProviderType.ANTHROPIC
        assert telemetry.cache_read_tokens == 800
        assert telemetry.cache_write_tokens == 200
        assert telemetry.cache_hit_ratio == 0.8

    def test_extract_openai_metrics(self):
        mock_response = MagicMock()
        mock_response.usage_metadata = {"input_tokens": 1000}
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 500},
            }
        }

        telemetry = extract_openai_cache_metrics(mock_response)

        assert telemetry.provider == ProviderType.OPENAI
        assert telemetry.cache_read_tokens == 500
        assert telemetry.cache_hit_ratio == 0.5

    def test_extract_gemini_metrics(self):
        mock_response = MagicMock()
        mock_response.usage_metadata = {"input_tokens": 1000}
        mock_response.response_metadata = {
            "prompt_token_count": 1000,
            "cached_content_token_count": 750,
        }

        telemetry = extract_gemini_cache_metrics(mock_response)

        assert telemetry.provider == ProviderType.GEMINI
        assert telemetry.cache_read_tokens == 750
        assert telemetry.cache_hit_ratio == 0.75

    def test_extract_cache_telemetry_unknown_provider(self):
        mock_response = MagicMock()
        mock_response.usage_metadata = {}
        mock_response.response_metadata = {}

        telemetry = extract_cache_telemetry(mock_response, ProviderType.UNKNOWN)

        assert telemetry.provider == ProviderType.UNKNOWN
        assert telemetry.cache_read_tokens == 0


class TestPromptCachingTelemetryMiddleware:
    def test_initialization(self):
        middleware = PromptCachingTelemetryMiddleware()

        assert middleware.telemetry_history == []

    def test_get_aggregate_stats_empty(self):
        middleware = PromptCachingTelemetryMiddleware()
        stats = middleware.get_aggregate_stats()

        assert stats["total_calls"] == 0
        assert stats["total_cache_read_tokens"] == 0

    def test_get_aggregate_stats_with_data(self):
        middleware = PromptCachingTelemetryMiddleware()
        middleware._telemetry_history = [
            CacheTelemetry(
                provider=ProviderType.ANTHROPIC,
                cache_read_tokens=800,
                cache_write_tokens=200,
                total_input_tokens=1000,
            ),
            CacheTelemetry(
                provider=ProviderType.ANTHROPIC,
                cache_read_tokens=900,
                cache_write_tokens=100,
                total_input_tokens=1000,
            ),
        ]

        stats = middleware.get_aggregate_stats()

        assert stats["total_calls"] == 2
        assert stats["total_cache_read_tokens"] == 1700
        assert stats["total_cache_write_tokens"] == 300
        assert stats["total_input_tokens"] == 2000
        assert stats["overall_cache_hit_ratio"] == 0.85

    def test_wrap_model_call_collects_telemetry(self):
        middleware = PromptCachingTelemetryMiddleware()

        mock_response = MagicMock()
        mock_response.response_metadata = {
            "model": "claude-3-sonnet",
            "usage": {"cache_read_input_tokens": 500},
        }
        mock_response.usage_metadata = {"input_tokens": 1000}

        def mock_handler(request):
            return mock_response

        mock_request = MagicMock()
        result = middleware.wrap_model_call(mock_request, mock_handler)

        assert result == mock_response
        assert len(middleware.telemetry_history) == 1


class TestOpenRouterSubProvider:
    def test_detect_anthropic_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("anthropic/claude-3-sonnet")
            == OpenRouterSubProvider.ANTHROPIC
        )
        assert (
            detect_openrouter_sub_provider("anthropic/claude-3.5-sonnet")
            == OpenRouterSubProvider.ANTHROPIC
        )

    def test_detect_openai_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("openai/gpt-4o")
            == OpenRouterSubProvider.OPENAI
        )
        assert (
            detect_openrouter_sub_provider("openai/o1-preview")
            == OpenRouterSubProvider.OPENAI
        )

    def test_detect_gemini_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("google/gemini-2.5-pro")
            == OpenRouterSubProvider.GEMINI
        )
        assert (
            detect_openrouter_sub_provider("google/gemini-3-flash")
            == OpenRouterSubProvider.GEMINI
        )

    def test_detect_deepseek_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("deepseek/deepseek-chat")
            == OpenRouterSubProvider.DEEPSEEK
        )

    def test_detect_groq_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("groq/kimi-k2") == OpenRouterSubProvider.GROQ
        )

    def test_detect_grok_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("xai/grok-2") == OpenRouterSubProvider.GROK
        )

    def test_detect_llama_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("meta-llama/llama-3.3-70b")
            == OpenRouterSubProvider.META_LLAMA
        )

    def test_detect_mistral_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("mistral/mistral-large")
            == OpenRouterSubProvider.MISTRAL
        )

    def test_detect_unknown_via_openrouter(self):
        assert (
            detect_openrouter_sub_provider("some-provider/some-model")
            == OpenRouterSubProvider.UNKNOWN
        )


class TestOpenRouterCachingStrategy:
    def test_openrouter_anthropic_applies_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"
        mock_model.openai_api_base = "https://openrouter.ai/api/v1"

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
            openrouter_model_name="anthropic/claude-3-sonnet",
        )

        assert strategy.provider == ProviderType.OPENROUTER
        assert strategy.sub_provider == OpenRouterSubProvider.ANTHROPIC
        assert strategy.should_apply_cache_markers is True

    def test_openrouter_openai_skips_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"
        mock_model.openai_api_base = "https://openrouter.ai/api/v1"

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
            openrouter_model_name="openai/gpt-4o",
        )

        assert strategy.provider == ProviderType.OPENROUTER
        assert strategy.sub_provider == OpenRouterSubProvider.OPENAI
        assert strategy.should_apply_cache_markers is False

    def test_openrouter_deepseek_skips_cache_markers(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatOpenAI"
        mock_model.__class__.__module__ = "langchain_openai"
        mock_model.openai_api_base = "https://openrouter.ai/api/v1"

        strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=10),
            model=mock_model,
            openrouter_model_name="deepseek/deepseek-chat",
        )

        assert strategy.provider == ProviderType.OPENROUTER
        assert strategy.sub_provider == OpenRouterSubProvider.DEEPSEEK
        assert strategy.should_apply_cache_markers is False


class TestGemini3Detection:
    def test_detect_gemini_3_from_model_name(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"
        mock_model.model_name = "gemini-3-pro-preview"

        assert detect_provider(mock_model) == ProviderType.GEMINI_3

    def test_detect_gemini_3_flash(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"
        mock_model.model_name = "gemini-3-flash-preview"

        assert detect_provider(mock_model) == ProviderType.GEMINI_3

    def test_detect_gemini_25_not_gemini_3(self):
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_model.__class__.__module__ = "langchain_google_genai"
        mock_model.model_name = "gemini-2.5-pro"

        assert detect_provider(mock_model) == ProviderType.GEMINI


class TestDeepSeekMetrics:
    def test_extract_deepseek_metrics(self):
        mock_response = MagicMock()
        mock_response.usage_metadata = {"input_tokens": 1000}
        mock_response.response_metadata = {
            "cache_hit_tokens": 700,
            "cache_miss_tokens": 300,
        }

        telemetry = extract_deepseek_cache_metrics(mock_response)

        assert telemetry.provider == ProviderType.DEEPSEEK
        assert telemetry.cache_read_tokens == 700
        assert telemetry.cache_write_tokens == 300
        assert telemetry.cache_hit_ratio == 0.7
