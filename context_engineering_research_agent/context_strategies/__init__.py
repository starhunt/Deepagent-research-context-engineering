"""Context Engineering 전략 모듈.

DeepAgents에서 구현된 5가지 Context Engineering 핵심 전략을
명시적으로 분리하고 문서화한 모듈입니다.

각 전략은 독립적으로 사용하거나 조합하여 사용할 수 있습니다.
"""

from context_engineering_research_agent.context_strategies.caching import (
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
)
from context_engineering_research_agent.context_strategies.isolation import (
    ContextIsolationStrategy,
)
from context_engineering_research_agent.context_strategies.offloading import (
    ContextOffloadingStrategy,
)
from context_engineering_research_agent.context_strategies.reduction import (
    ContextReductionStrategy,
)
from context_engineering_research_agent.context_strategies.retrieval import (
    ContextRetrievalStrategy,
)

__all__ = [
    "ContextOffloadingStrategy",
    "ContextReductionStrategy",
    "ContextRetrievalStrategy",
    "ContextIsolationStrategy",
    "ContextCachingStrategy",
    "ProviderType",
    "OpenRouterSubProvider",
    "detect_provider",
    "detect_openrouter_sub_provider",
    "requires_cache_control_marker",
    "CacheTelemetry",
    "PromptCachingTelemetryMiddleware",
]
