import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from context_engineering_research_agent.context_strategies.reduction import (
    ContextReductionStrategy,
    ReductionConfig,
    ReductionResult,
)


class TestReductionConfig:
    def test_default_values(self):
        config = ReductionConfig()

        assert config.context_threshold == 0.85
        assert config.model_context_window == 200000
        assert config.compaction_age_threshold == 10
        assert config.min_messages_to_keep == 5
        assert config.chars_per_token == 4

    def test_custom_values(self):
        config = ReductionConfig(
            context_threshold=0.90,
            model_context_window=100000,
            compaction_age_threshold=5,
            min_messages_to_keep=3,
        )

        assert config.context_threshold == 0.90
        assert config.model_context_window == 100000
        assert config.compaction_age_threshold == 5
        assert config.min_messages_to_keep == 3


class TestReductionResult:
    def test_not_reduced(self):
        result = ReductionResult(was_reduced=False)

        assert result.was_reduced is False
        assert result.technique_used is None
        assert result.original_message_count == 0
        assert result.reduced_message_count == 0
        assert result.estimated_tokens_saved == 0

    def test_reduced_with_compaction(self):
        result = ReductionResult(
            was_reduced=True,
            technique_used="compaction",
            original_message_count=50,
            reduced_message_count=30,
            estimated_tokens_saved=5000,
        )

        assert result.was_reduced is True
        assert result.technique_used == "compaction"
        assert result.original_message_count == 50
        assert result.reduced_message_count == 30
        assert result.estimated_tokens_saved == 5000


class TestContextReductionStrategy:
    @pytest.fixture
    def strategy(self) -> ContextReductionStrategy:
        return ContextReductionStrategy()

    @pytest.fixture
    def strategy_low_threshold(self) -> ContextReductionStrategy:
        return ContextReductionStrategy(
            config=ReductionConfig(
                context_threshold=0.1,
                model_context_window=1000,
            )
        )

    def test_estimate_tokens(self, strategy: ContextReductionStrategy):
        messages = [
            HumanMessage(content="a" * 400),
            AIMessage(content="b" * 400),
        ]
        estimated = strategy._estimate_tokens(messages)

        assert estimated == 200

    def test_get_context_usage_ratio(self, strategy: ContextReductionStrategy):
        messages = [
            HumanMessage(content="x" * 40000),
        ]
        ratio = strategy._get_context_usage_ratio(messages)

        assert ratio == pytest.approx(0.05, rel=0.01)

    def test_should_reduce_below_threshold(self, strategy: ContextReductionStrategy):
        messages = [HumanMessage(content="short message")]

        assert strategy._should_reduce(messages) is False

    def test_should_reduce_above_threshold(
        self, strategy_low_threshold: ContextReductionStrategy
    ):
        messages = [HumanMessage(content="x" * 1000)]

        assert strategy_low_threshold._should_reduce(messages) is True

    def test_apply_compaction_removes_old_tool_calls(
        self, strategy: ContextReductionStrategy
    ):
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"question {i}"))
            ai_msg = AIMessage(
                content=f"answer {i}",
                tool_calls=(
                    [{"id": f"call_{i}", "name": "search", "args": {"q": "test"}}]
                    if i < 15
                    else []
                ),
            )
            messages.append(ai_msg)
            if i < 15:
                messages.append(
                    ToolMessage(content=f"result {i}", tool_call_id=f"call_{i}")
                )

        compacted, result = strategy.apply_compaction(messages)

        assert result.was_reduced is True
        assert result.technique_used == "compaction"
        assert len(compacted) < len(messages)

    def test_apply_compaction_keeps_recent_messages(
        self, strategy: ContextReductionStrategy
    ):
        messages = []
        for i in range(5):
            messages.append(HumanMessage(content=f"recent question {i}"))
            messages.append(AIMessage(content=f"recent answer {i}"))

        compacted, result = strategy.apply_compaction(messages)

        assert len(compacted) == len(messages)

    def test_apply_compaction_preserves_text_content(self):
        strategy = ContextReductionStrategy(
            config=ReductionConfig(compaction_age_threshold=2)
        )
        messages = [
            HumanMessage(content="old question"),
            AIMessage(
                content="old answer with important info",
                tool_calls=[{"id": "call_old", "name": "search", "args": {}}],
            ),
            ToolMessage(content="old result", tool_call_id="call_old"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]

        compacted, _ = strategy.apply_compaction(messages)

        text_contents = [str(m.content) for m in compacted]
        has_important_info = any("important info" in c for c in text_contents)

        assert has_important_info is True

    def test_apply_compaction_removes_tool_messages(self):
        strategy = ContextReductionStrategy(
            config=ReductionConfig(compaction_age_threshold=2)
        )
        messages = [
            HumanMessage(content="old question"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_old", "name": "search", "args": {}}],
            ),
            ToolMessage(content="old tool result", tool_call_id="call_old"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]

        compacted, _ = strategy.apply_compaction(messages)

        tool_message_count = sum(1 for m in compacted if isinstance(m, ToolMessage))

        assert tool_message_count == 0

    def test_reduce_context_no_reduction_needed(
        self, strategy: ContextReductionStrategy
    ):
        messages = [HumanMessage(content="short")]

        reduced, result = strategy.reduce_context(messages)

        assert result.was_reduced is False
        assert reduced == messages

    def test_reduce_context_with_very_large_messages(self):
        strategy = ContextReductionStrategy(
            config=ReductionConfig(
                context_threshold=0.01,
                model_context_window=100,
                compaction_age_threshold=5,
            )
        )
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"question {i} " * 100))
            messages.append(
                AIMessage(
                    content=f"answer {i} " * 100,
                    tool_calls=[{"id": f"c{i}", "name": "s", "args": {}}],
                )
            )
            messages.append(ToolMessage(content="result " * 100, tool_call_id=f"c{i}"))

        compacted, result = strategy.apply_compaction(messages)

        assert result.was_reduced is True
        assert len(compacted) < len(messages)

    def test_create_summary_prompt(self, strategy: ContextReductionStrategy):
        messages = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]

        prompt = strategy._create_summary_prompt(messages)

        assert "Python" in prompt
        assert "Human" in prompt
        assert "AI" in prompt

    def test_apply_summarization_no_model(self, strategy: ContextReductionStrategy):
        messages = [HumanMessage(content="test")]

        summarized, result = strategy.apply_summarization(messages)

        assert result.was_reduced is False
        assert summarized == messages

    def test_compaction_preserves_system_messages(self):
        strategy = ContextReductionStrategy(
            config=ReductionConfig(compaction_age_threshold=2)
        )
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]

        compacted, _ = strategy.apply_compaction(messages)

        system_messages = [m for m in compacted if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1
        assert "helpful assistant" in str(system_messages[0].content)
