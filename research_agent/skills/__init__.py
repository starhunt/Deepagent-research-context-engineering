"""research_agent용 Skills 모듈.

이 모듈은 점진적 공개(Progressive Disclosure) 패턴으로 Agent Skills 패턴을 구현한다:
1. 세션 시작 시 SKILL.md 파일에서 YAML 프론트매터 파싱
2. 스킬 메타데이터(이름 + 설명)를 시스템 프롬프트에 주입
3. 스킬이 관련될 때 에이전트가 전체 SKILL.md 콘텐츠 읽기

공개 API:
- SkillsMiddleware: 에이전트 실행에 스킬을 통합하는 미들웨어
- list_skills: 디렉토리에서 스킬 메타데이터 로드
- SkillMetadata: 스킬 메타데이터 구조용 TypedDict
"""

from research_agent.skills.load import SkillMetadata, list_skills
from research_agent.skills.middleware import SkillsMiddleware

__all__ = [
    "SkillsMiddleware",
    "list_skills",
    "SkillMetadata",
]
