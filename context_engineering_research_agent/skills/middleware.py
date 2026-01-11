"""스킬 시스템 미들웨어.

Progressive Disclosure 패턴으로 스킬 메타데이터를 시스템 프롬프트에 주입합니다.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from context_engineering_research_agent.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    skills_metadata: NotRequired[list[SkillMetadata]]


class SkillsStateUpdate(TypedDict):
    skills_metadata: list[SkillMetadata]


SKILLS_SYSTEM_PROMPT = """

## 스킬 시스템

스킬 라이브러리를 통해 전문화된 기능과 도메인 지식을 사용할 수 있습니다.

{skills_locations}

**사용 가능한 스킬:**

{skills_list}

**스킬 사용법 (Progressive Disclosure):**

스킬은 점진적 공개 패턴을 따릅니다. 위에서 스킬의 존재(이름 + 설명)를 알 수 있지만,
필요할 때만 전체 지침을 읽습니다:

1. 스킬 적용 여부 판단: 사용자 요청이 스킬 설명과 일치하는지 확인
2. 전체 지침 읽기: read_file로 SKILL.md 경로 읽기
3. 지침 따르기: SKILL.md에는 단계별 워크플로우와 예시 포함
4. 지원 파일 활용: 스킬에 Python 스크립트나 설정 파일 포함 가능

**스킬 사용 시점:**
- 사용자 요청이 스킬 도메인과 일치할 때
- 전문 지식이나 구조화된 워크플로우가 도움될 때
- 복잡한 작업에 검증된 패턴이 필요할 때
"""


class SkillsMiddleware(AgentMiddleware):
    """Progressive Disclosure 패턴으로 스킬을 노출하는 미들웨어."""

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = (
            Path(project_skills_dir).expanduser() if project_skills_dir else None
        )
        self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        locations = [f"**사용자 스킬**: `{self.user_skills_display}`"]
        if self.project_skills_dir:
            locations.append(
                f"**프로젝트 스킬**: `{self.project_skills_dir}` (사용자 스킬 오버라이드)"
            )
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(사용 가능한 스킬 없음. {' 또는 '.join(locations)}에서 스킬 생성 가능)"

        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        if user_skills:
            lines.append("**사용자 스킬:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → 전체 지침: `{skill['path']}`")
            lines.append("")

        if project_skills:
            lines.append("**프로젝트 스킬:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → 전체 지침: `{skill['path']}`")

        return "\n".join(lines)

    def before_agent(
        self, state: SkillsState, runtime: Runtime
    ) -> SkillsStateUpdate | None:
        skills = list_skills(
            user_skills_dir=self.skills_dir,
            project_skills_dir=self.project_skills_dir,
        )
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skills_metadata = request.state.get("skills_metadata", [])

        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return handler(request.override(system_prompt=system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
