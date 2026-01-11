import pytest
from langchain_core.messages import ToolMessage

from context_engineering_research_agent.context_strategies.offloading import (
    ContextOffloadingStrategy,
    OffloadingConfig,
    OffloadingResult,
)


class TestOffloadingConfig:
    def test_default_values(self):
        config = OffloadingConfig()

        assert config.token_limit_before_evict == 20000
        assert config.eviction_path_prefix == "/large_tool_results"
        assert config.preview_lines == 10
        assert config.chars_per_token == 4

    def test_custom_values(self):
        config = OffloadingConfig(
            token_limit_before_evict=15000,
            eviction_path_prefix="/custom_path",
            preview_lines=5,
            chars_per_token=3,
        )

        assert config.token_limit_before_evict == 15000
        assert config.eviction_path_prefix == "/custom_path"
        assert config.preview_lines == 5
        assert config.chars_per_token == 3


class TestOffloadingResult:
    def test_not_offloaded(self):
        result = OffloadingResult(was_offloaded=False, original_size=100)

        assert result.was_offloaded is False
        assert result.original_size == 100
        assert result.file_path is None
        assert result.preview is None

    def test_offloaded(self):
        result = OffloadingResult(
            was_offloaded=True,
            original_size=100000,
            file_path="/large_tool_results/call_123",
            preview="first 10 lines...",
        )

        assert result.was_offloaded is True
        assert result.original_size == 100000
        assert result.file_path == "/large_tool_results/call_123"
        assert result.preview == "first 10 lines..."


class TestContextOffloadingStrategy:
    @pytest.fixture
    def strategy(self) -> ContextOffloadingStrategy:
        return ContextOffloadingStrategy()

    @pytest.fixture
    def strategy_low_threshold(self) -> ContextOffloadingStrategy:
        return ContextOffloadingStrategy(
            config=OffloadingConfig(token_limit_before_evict=100)
        )

    def test_estimate_tokens(self, strategy: ContextOffloadingStrategy):
        content = "a" * 400
        estimated = strategy._estimate_tokens(content)

        assert estimated == 100

    def test_estimate_tokens_with_custom_chars_per_token(self):
        strategy = ContextOffloadingStrategy(config=OffloadingConfig(chars_per_token=2))
        content = "a" * 400
        estimated = strategy._estimate_tokens(content)

        assert estimated == 200

    def test_should_offload_small_content(self, strategy: ContextOffloadingStrategy):
        small_content = "short text" * 100

        assert strategy._should_offload(small_content) is False

    def test_should_offload_large_content(self, strategy: ContextOffloadingStrategy):
        large_content = "x" * 100000

        assert strategy._should_offload(large_content) is True

    def test_should_offload_boundary(self):
        config = OffloadingConfig(token_limit_before_evict=100, chars_per_token=4)
        strategy = ContextOffloadingStrategy(config=config)

        exactly_at_limit = "x" * 400
        just_over_limit = "x" * 404

        assert strategy._should_offload(exactly_at_limit) is False
        assert strategy._should_offload(just_over_limit) is True

    def test_create_preview_short_content(self, strategy: ContextOffloadingStrategy):
        content = "line1\nline2\nline3"
        preview = strategy._create_preview(content)

        assert "line1" in preview
        assert "line2" in preview
        assert "line3" in preview

    def test_create_preview_long_content(self, strategy: ContextOffloadingStrategy):
        lines = [f"line_{i}" for i in range(100)]
        content = "\n".join(lines)
        preview = strategy._create_preview(content)

        assert "line_0" in preview
        assert "line_9" in preview
        assert "line_10" not in preview

    def test_create_preview_with_custom_lines(self):
        strategy = ContextOffloadingStrategy(config=OffloadingConfig(preview_lines=3))
        lines = [f"line_{i}" for i in range(10)]
        content = "\n".join(lines)
        preview = strategy._create_preview(content)

        assert "line_0" in preview
        assert "line_2" in preview
        assert "line_3" not in preview

    def test_create_preview_truncates_long_lines(
        self, strategy: ContextOffloadingStrategy
    ):
        long_line = "x" * 2000
        preview = strategy._create_preview(long_line)

        assert len(preview.split("\t")[1]) == 1000

    def test_create_offload_message(self, strategy: ContextOffloadingStrategy):
        message = strategy._create_offload_message(
            tool_call_id="call_123",
            file_path="/large_tool_results/call_123",
            preview="preview content",
        )

        assert "/large_tool_results/call_123" in message
        assert "preview content" in message
        assert "read_file" in message

    def test_sanitize_tool_call_id(self, strategy: ContextOffloadingStrategy):
        normal_id = "call_abc123"
        special_id = "call/with:special@chars!"

        assert strategy._sanitize_tool_call_id(normal_id) == "call_abc123"
        assert strategy._sanitize_tool_call_id(special_id) == "call_with_special_chars_"

    def test_process_tool_result_small_content(
        self, strategy: ContextOffloadingStrategy
    ):
        tool_result = ToolMessage(content="small content", tool_call_id="call_123")

        class MockRuntime:
            state = {}
            config = {}

        processed, result = strategy.process_tool_result(tool_result, MockRuntime())  # type: ignore

        assert result.was_offloaded is False
        assert processed.content == "small content"

    def test_process_tool_result_no_backend(
        self, strategy_low_threshold: ContextOffloadingStrategy
    ):
        large_content = "x" * 1000
        tool_result = ToolMessage(content=large_content, tool_call_id="call_123")

        class MockRuntime:
            state = {}
            config = {}

        processed, result = strategy_low_threshold.process_tool_result(
            tool_result,
            MockRuntime(),  # type: ignore
        )

        assert result.was_offloaded is False
        assert processed.content == large_content
