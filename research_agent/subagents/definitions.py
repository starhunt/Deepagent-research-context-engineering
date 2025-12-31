"""research_agent용 SubAgent 정의.

이 모듈은 Claude Code subagent_type 패턴을 따르는
전문화된 SubAgent 명세를 정의한다. 각 SubAgent는:
- 고유 이름 (subagent_type으로 사용)
- 위임 결정을 위한 명확한 설명
- 전문화된 시스템 프롬프트
- 엄선된 도구 세트
- 선택적 모델 오버라이드

SubAgent 타입:
    researcher: 반성을 포함한 심층 웹 연구
    explorer: 빠른 읽기 전용 코드베이스/문서 탐색
    synthesizer: 연구 통합 및 보고서 생성
"""

from datetime import datetime

from research_agent.prompts import (
    EXPLORER_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SYNTHESIZER_INSTRUCTIONS,
)

# 동적 프롬프트용 현재 날짜
_current_date = datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# EXPLORER SubAgent
# =============================================================================

EXPLORER_AGENT = {
    "name": "explorer",
    "description": "Fast read-only exploration of codebases and documents. Use for finding files, searching patterns, and quick information retrieval. Cannot modify files.",
    "system_prompt": EXPLORER_INSTRUCTIONS,
    "tools": [],  # 런타임에 읽기 전용 도구로 채워짐
    "capabilities": ["explore", "search", "read"],
}


# =============================================================================
# RESEARCHER SubAgent
# =============================================================================

RESEARCHER_AGENT = {
    "name": "researcher",
    "description": "Deep web research with reflection. Use for comprehensive topic research, gathering sources, and in-depth analysis. Includes tavily_search and think_tool.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=_current_date),
    "tools": [],  # 런타임에 tavily_search, think_tool로 채워짐
    "capabilities": ["research", "web", "analysis"],
}


# =============================================================================
# SYNTHESIZER SubAgent
# =============================================================================

SYNTHESIZER_AGENT = {
    "name": "synthesizer",
    "description": "Synthesize multiple research findings into coherent reports. Use for combining sub-agent results, creating summaries, and writing final reports.",
    "system_prompt": SYNTHESIZER_INSTRUCTIONS,
    "tools": [],  # 런타임에 read_file, write_file, think_tool로 채워짐
    "capabilities": ["synthesize", "write", "analysis"],
}


# =============================================================================
# 유틸리티 함수
# =============================================================================


def get_all_subagents() -> list[dict]:
    """모든 SubAgent 정의를 목록으로 반환한다.

    Returns:
        SubAgent 명세 딕셔너리 목록.

    Note:
        도구는 비어 있으며 런타임에서 사용 가능한 도구를 기반으로
        에이전트 생성 시 채워져야 한다.
    """
    return [
        RESEARCHER_AGENT,
        EXPLORER_AGENT,
        SYNTHESIZER_AGENT,
    ]


def get_subagent_by_name(name: str) -> dict | None:
    """이름으로 특정 SubAgent 정의를 가져온다.

    Args:
        name: SubAgent 이름 (예: "researcher", "explorer", "synthesizer")

    Returns:
        찾으면 SubAgent 명세 딕셔너리, 그렇지 않으면 None.
    """
    agents = {
        "researcher": RESEARCHER_AGENT,
        "explorer": EXPLORER_AGENT,
        "synthesizer": SYNTHESIZER_AGENT,
    }
    return agents.get(name)


def get_subagent_descriptions() -> str:
    """모든 SubAgent의 포맷된 설명을 가져온다.

    Returns:
        모든 SubAgent와 설명을 나열한 포맷된 문자열.
    """
    descriptions = []
    for agent in get_all_subagents():
        descriptions.append(f"- **{agent['name']}**: {agent['description']}")
    return "\n".join(descriptions)
