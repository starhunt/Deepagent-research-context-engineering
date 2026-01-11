import pytest

from context_engineering_research_agent.context_strategies.retrieval import (
    ContextRetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
)


class TestRetrievalConfig:
    def test_default_values(self):
        config = RetrievalConfig()

        assert config.default_read_limit == 500
        assert config.max_grep_results == 100
        assert config.max_glob_results == 100
        assert config.truncate_line_length == 2000

    def test_custom_values(self):
        config = RetrievalConfig(
            default_read_limit=1000,
            max_grep_results=50,
            max_glob_results=200,
            truncate_line_length=3000,
        )

        assert config.default_read_limit == 1000
        assert config.max_grep_results == 50
        assert config.max_glob_results == 200
        assert config.truncate_line_length == 3000


class TestRetrievalResult:
    def test_basic_result(self):
        result = RetrievalResult(
            tool_used="grep",
            query="TODO",
            result_count=10,
        )

        assert result.tool_used == "grep"
        assert result.query == "TODO"
        assert result.result_count == 10
        assert result.was_truncated is False

    def test_truncated_result(self):
        result = RetrievalResult(
            tool_used="glob",
            query="**/*.py",
            result_count=100,
            was_truncated=True,
        )

        assert result.was_truncated is True


class TestContextRetrievalStrategy:
    @pytest.fixture
    def strategy(self) -> ContextRetrievalStrategy:
        return ContextRetrievalStrategy()

    @pytest.fixture
    def strategy_with_custom_config(self) -> ContextRetrievalStrategy:
        return ContextRetrievalStrategy(
            config=RetrievalConfig(
                default_read_limit=100,
                max_grep_results=10,
                max_glob_results=10,
            )
        )

    def test_initialization(self, strategy: ContextRetrievalStrategy):
        assert strategy.config is not None
        assert len(strategy.tools) == 3

    def test_creates_read_file_tool(self, strategy: ContextRetrievalStrategy):
        tool_names = [t.name for t in strategy.tools]

        assert "read_file" in tool_names

    def test_creates_grep_tool(self, strategy: ContextRetrievalStrategy):
        tool_names = [t.name for t in strategy.tools]

        assert "grep" in tool_names

    def test_creates_glob_tool(self, strategy: ContextRetrievalStrategy):
        tool_names = [t.name for t in strategy.tools]

        assert "glob" in tool_names

    def test_read_file_tool_description(self, strategy: ContextRetrievalStrategy):
        read_file_tool = next(t for t in strategy.tools if t.name == "read_file")

        assert "500" in read_file_tool.description
        assert "offset" in read_file_tool.description.lower()
        assert "limit" in read_file_tool.description.lower()

    def test_grep_tool_description(self, strategy: ContextRetrievalStrategy):
        grep_tool = next(t for t in strategy.tools if t.name == "grep")

        assert "100" in grep_tool.description
        assert "pattern" in grep_tool.description.lower()

    def test_glob_tool_description(self, strategy: ContextRetrievalStrategy):
        glob_tool = next(t for t in strategy.tools if t.name == "glob")

        assert "100" in glob_tool.description
        assert "**/*.py" in glob_tool.description

    def test_custom_config_affects_tool_descriptions(
        self, strategy_with_custom_config: ContextRetrievalStrategy
    ):
        read_file_tool = next(
            t for t in strategy_with_custom_config.tools if t.name == "read_file"
        )
        grep_tool = next(
            t for t in strategy_with_custom_config.tools if t.name == "grep"
        )

        assert "100" in read_file_tool.description
        assert "10" in grep_tool.description

    def test_no_backend_read_file(self, strategy: ContextRetrievalStrategy):
        read_file_tool = next(t for t in strategy.tools if t.name == "read_file")

        class MockRuntime:
            state = {}
            config = {}

        result = read_file_tool.func(  # type: ignore
            file_path="/test.txt",
            runtime=MockRuntime(),
        )

        assert "백엔드가 설정되지 않았습니다" in result

    def test_no_backend_grep(self, strategy: ContextRetrievalStrategy):
        grep_tool = next(t for t in strategy.tools if t.name == "grep")

        class MockRuntime:
            state = {}
            config = {}

        result = grep_tool.func(  # type: ignore
            pattern="TODO",
            runtime=MockRuntime(),
        )

        assert "백엔드가 설정되지 않았습니다" in result

    def test_no_backend_glob(self, strategy: ContextRetrievalStrategy):
        glob_tool = next(t for t in strategy.tools if t.name == "glob")

        class MockRuntime:
            state = {}
            config = {}

        result = glob_tool.func(  # type: ignore
            pattern="**/*.py",
            runtime=MockRuntime(),
        )

        assert "백엔드가 설정되지 않았습니다" in result
