"""자율적 연구 SubAgent 모듈.

이 모듈은 "넓게 탐색 → 깊게 파기" 방법론을 따르는
자체 계획 및 자체 반성 연구 에이전트를 제공한다.

사용법:
    from research_agent.researcher import get_researcher_subagent

    researcher = get_researcher_subagent(model=model, backend=backend)
    # create_deep_agent(subagents=[...])에 사용할 CompiledSubAgent 반환
"""

from research_agent.researcher.agent import (
    create_researcher_agent,
    get_researcher_subagent,
)
from research_agent.researcher.prompts import AUTONOMOUS_RESEARCHER_INSTRUCTIONS

__all__ = [
    "create_researcher_agent",
    "get_researcher_subagent",
    "AUTONOMOUS_RESEARCHER_INSTRUCTIONS",
]
