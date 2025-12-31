"""research_agent용 SubAgents 모듈.

이 모듈은 Claude Code에서 영감을 받은 SubAgent 시스템을 구현한다:
- 다중 전문화 에이전트 타입 (researcher, explorer, synthesizer)
- 동적 에이전트 관리를 위한 SubAgent 레지스트리
- task 도구를 통한 타입 기반 라우팅

아키텍처:
    Main Orchestrator Agent
        ├── researcher SubAgent (심층 웹 연구)
        ├── explorer SubAgent (빠른 코드베이스 탐색)
        └── synthesizer SubAgent (연구 결과 통합)

사용법:
    from research_agent.subagents import get_all_subagents

    agent = create_deep_agent(
        model=model,
        subagents=get_all_subagents(),
        ...
    )
"""

from research_agent.subagents.definitions import (
    EXPLORER_AGENT,
    RESEARCHER_AGENT,
    SYNTHESIZER_AGENT,
    get_all_subagents,
)
from research_agent.subagents.registry import SubAgentRegistry

__all__ = [
    "SubAgentRegistry",
    "RESEARCHER_AGENT",
    "EXPLORER_AGENT",
    "SYNTHESIZER_AGENT",
    "get_all_subagents",
]
