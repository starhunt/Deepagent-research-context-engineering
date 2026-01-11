"""Context Engineering 연구 에이전트용 스킬 시스템.

이 모듈은 Anthropic의 Agent Skills 패턴을 구현하여
에이전트에게 도메인별 전문 지식과 워크플로우를 제공합니다.

## Progressive Disclosure (점진적 공개)

스킬은 점진적 공개 패턴을 따릅니다:
1. 세션 시작 시 스킬 메타데이터(이름 + 설명)만 로드
2. 에이전트가 필요할 때 전체 SKILL.md 내용 읽기
3. 컨텍스트 윈도우 효율적 사용

## Context Engineering 관점에서의 스킬

스킬 시스템은 Context Engineering의 핵심 전략들을 활용합니다:

1. **Context Retrieval**: read_file로 필요할 때만 스킬 내용 로드
2. **Context Offloading**: 전체 스킬 내용 대신 메타데이터만 시스템 프롬프트에 포함
3. **Context Isolation**: 각 스킬은 독립적인 도메인 지식 캡슐화
"""

from context_engineering_research_agent.skills.load import (
    SkillMetadata,
    list_skills,
)
from context_engineering_research_agent.skills.middleware import (
    SkillsMiddleware,
    SkillsState,
)

__all__ = [
    "SkillMetadata",
    "list_skills",
    "SkillsMiddleware",
    "SkillsState",
]
