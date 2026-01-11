import pytest

from context_engineering_research_agent.context_strategies.isolation import (
    ContextIsolationStrategy,
    IsolationConfig,
    IsolationResult,
)


class TestIsolationConfig:
    def test_default_values(self):
        config = IsolationConfig()

        assert config.default_model == "gpt-4.1"
        assert config.include_general_purpose_agent is True
        assert config.excluded_state_keys == (
            "messages",
            "todos",
            "structured_response",
        )

    def test_custom_values(self):
        config = IsolationConfig(
            default_model="claude-3-5-sonnet",
            include_general_purpose_agent=False,
            excluded_state_keys=("messages", "custom_key"),
        )

        assert config.default_model == "claude-3-5-sonnet"
        assert config.include_general_purpose_agent is False
        assert config.excluded_state_keys == ("messages", "custom_key")


class TestIsolationResult:
    def test_successful_result(self):
        result = IsolationResult(
            subagent_name="researcher",
            was_successful=True,
            result_length=500,
        )

        assert result.subagent_name == "researcher"
        assert result.was_successful is True
        assert result.result_length == 500
        assert result.error is None

    def test_failed_result(self):
        result = IsolationResult(
            subagent_name="researcher",
            was_successful=False,
            result_length=0,
            error="SubAgent not found",
        )

        assert result.was_successful is False
        assert result.error == "SubAgent not found"


class TestContextIsolationStrategy:
    @pytest.fixture
    def strategy(self) -> ContextIsolationStrategy:
        return ContextIsolationStrategy()

    @pytest.fixture
    def strategy_with_subagents(self) -> ContextIsolationStrategy:
        subagents = [
            {
                "name": "researcher",
                "description": "Research agent",
                "system_prompt": "You are a researcher",
                "tools": [],
            },
            {
                "name": "coder",
                "description": "Coding agent",
                "system_prompt": "You are a coder",
                "tools": [],
            },
        ]
        return ContextIsolationStrategy(subagents=subagents)  # type: ignore

    def test_initialization(self, strategy: ContextIsolationStrategy):
        assert strategy.config is not None
        assert len(strategy.tools) == 1

    def test_creates_task_tool(self, strategy: ContextIsolationStrategy):
        assert strategy.tools[0].name == "task"

    def test_task_tool_description(
        self, strategy_with_subagents: ContextIsolationStrategy
    ):
        task_tool = strategy_with_subagents.tools[0]

        assert "researcher" in task_tool.description
        assert "coder" in task_tool.description
        assert "SubAgent" in task_tool.description

    def test_get_subagent_descriptions(
        self, strategy_with_subagents: ContextIsolationStrategy
    ):
        descriptions = strategy_with_subagents._get_subagent_descriptions()

        assert "researcher" in descriptions
        assert "Research agent" in descriptions
        assert "coder" in descriptions
        assert "Coding agent" in descriptions

    def test_prepare_subagent_state(self, strategy: ContextIsolationStrategy):
        state = {
            "messages": [{"role": "user", "content": "old message"}],
            "todos": ["task1", "task2"],
            "structured_response": {"key": "value"},
            "files": {"path": "/test"},
        }

        prepared = strategy._prepare_subagent_state(state, "New task description")

        assert "todos" not in prepared
        assert "structured_response" not in prepared
        assert "files" in prepared
        assert len(prepared["messages"]) == 1
        assert prepared["messages"][0].content == "New task description"

    def test_prepare_subagent_state_custom_excluded_keys(self):
        strategy = ContextIsolationStrategy(
            config=IsolationConfig(excluded_state_keys=("messages", "custom_exclude"))
        )
        state = {
            "messages": [{"role": "user", "content": "old"}],
            "custom_exclude": "should be excluded",
            "keep_this": "should be kept",
        }

        prepared = strategy._prepare_subagent_state(state, "New task")

        assert "custom_exclude" not in prepared
        assert "keep_this" in prepared

    def test_compile_subagents_empty(self, strategy: ContextIsolationStrategy):
        agents = strategy._compile_subagents()

        assert agents == {}

    def test_compile_subagents_caches_result(
        self, strategy_with_subagents: ContextIsolationStrategy
    ):
        strategy_with_subagents._compiled_agents = {"cached": "value"}  # type: ignore

        agents = strategy_with_subagents._compile_subagents()

        assert agents == {"cached": "value"}

    def test_task_without_subagents(self, strategy: ContextIsolationStrategy):
        task_tool = strategy.tools[0]

        class MockRuntime:
            state = {}
            config = {}
            tool_call_id = "call_123"

        result = task_tool.func(  # type: ignore
            description="Test task",
            subagent_type="researcher",
            runtime=MockRuntime(),
        )

        assert "존재하지 않습니다" in str(result)
