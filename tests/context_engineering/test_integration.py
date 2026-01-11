import os

import pytest

from context_engineering_research_agent import (
    ContextCachingStrategy,
    ContextIsolationStrategy,
    ContextOffloadingStrategy,
    ContextReductionStrategy,
    ContextRetrievalStrategy,
    create_context_aware_agent,
)
from context_engineering_research_agent.context_strategies.caching import CachingConfig
from context_engineering_research_agent.context_strategies.offloading import (
    OffloadingConfig,
)
from context_engineering_research_agent.context_strategies.reduction import (
    ReductionConfig,
)

SKIP_OPENAI = not os.environ.get("OPENAI_API_KEY")


class TestModuleExports:
    def test_exports_all_strategies(self):
        assert ContextOffloadingStrategy is not None
        assert ContextReductionStrategy is not None
        assert ContextRetrievalStrategy is not None
        assert ContextIsolationStrategy is not None
        assert ContextCachingStrategy is not None

    def test_exports_create_context_aware_agent(self):
        assert create_context_aware_agent is not None
        assert callable(create_context_aware_agent)


class TestStrategyInstantiation:
    def test_offloading_strategy_instantiation(self):
        strategy = ContextOffloadingStrategy()
        assert strategy.config.token_limit_before_evict == 20000

        custom_strategy = ContextOffloadingStrategy(
            config=OffloadingConfig(token_limit_before_evict=10000)
        )
        assert custom_strategy.config.token_limit_before_evict == 10000

    def test_reduction_strategy_instantiation(self):
        strategy = ContextReductionStrategy()
        assert strategy.config.context_threshold == 0.85

        custom_strategy = ContextReductionStrategy(
            config=ReductionConfig(context_threshold=0.9)
        )
        assert custom_strategy.config.context_threshold == 0.9

    def test_retrieval_strategy_instantiation(self):
        strategy = ContextRetrievalStrategy()
        assert len(strategy.tools) == 3

    def test_isolation_strategy_instantiation(self):
        strategy = ContextIsolationStrategy()
        assert len(strategy.tools) == 1

    def test_caching_strategy_instantiation(self):
        strategy = ContextCachingStrategy()
        assert strategy.config.min_cacheable_tokens == 1024

        custom_strategy = ContextCachingStrategy(
            config=CachingConfig(min_cacheable_tokens=2048)
        )
        assert custom_strategy.config.min_cacheable_tokens == 2048


@pytest.mark.skipif(SKIP_OPENAI, reason="OPENAI_API_KEY not set")
class TestCreateContextAwareAgent:
    def test_create_agent_default_settings(self):
        agent = create_context_aware_agent(model_name="gpt-4.1")

        assert agent is not None

    def test_create_agent_all_disabled(self):
        agent = create_context_aware_agent(
            model_name="gpt-4.1",
            enable_offloading=False,
            enable_reduction=False,
            enable_caching=False,
        )

        assert agent is not None

    def test_create_agent_all_enabled(self):
        agent = create_context_aware_agent(
            model_name="gpt-4.1",
            enable_offloading=True,
            enable_reduction=True,
            enable_caching=True,
        )

        assert agent is not None

    def test_create_agent_custom_thresholds(self):
        agent = create_context_aware_agent(
            model_name="gpt-4.1",
            enable_offloading=True,
            enable_reduction=True,
            offloading_token_limit=10000,
            reduction_threshold=0.9,
        )

        assert agent is not None

    def test_create_agent_offloading_only(self):
        agent = create_context_aware_agent(
            model_name="gpt-4.1",
            enable_offloading=True,
            enable_reduction=False,
            enable_caching=False,
        )

        assert agent is not None

    def test_create_agent_reduction_only(self):
        agent = create_context_aware_agent(
            model_name="gpt-4.1",
            enable_offloading=False,
            enable_reduction=True,
            enable_caching=False,
        )

        assert agent is not None


class TestStrategyCombinations:
    def test_offloading_with_reduction(self):
        offloading = ContextOffloadingStrategy(
            config=OffloadingConfig(token_limit_before_evict=15000)
        )
        reduction = ContextReductionStrategy(
            config=ReductionConfig(context_threshold=0.8)
        )

        assert offloading.config.token_limit_before_evict == 15000
        assert reduction.config.context_threshold == 0.8

    def test_all_strategies_together(self):
        offloading = ContextOffloadingStrategy()
        reduction = ContextReductionStrategy()
        retrieval = ContextRetrievalStrategy()
        isolation = ContextIsolationStrategy()
        caching = ContextCachingStrategy()

        strategies = [offloading, reduction, retrieval, isolation, caching]

        assert len(strategies) == 5
        for strategy in strategies:
            assert hasattr(strategy, "config")
