"""에이전트 스킬을 시스템 프롬프트에 로드하고 노출하기 위한 미들웨어.

이 미들웨어는 점진적 공개를 통해 Anthropic의 "Agent Skills" 패턴을 구현한다:
1. 세션 시작 시 SKILL.md 파일에서 YAML 프론트매터 파싱
2. 스킬 메타데이터(이름 + 설명)를 시스템 프롬프트에 주입
3. 스킬이 관련될 때 에이전트가 전체 SKILL.md 콘텐츠 읽기

스킬 디렉토리 구조 (프로젝트 레벨):
{PROJECT_ROOT}/skills/
├── web-research/
│   ├── SKILL.md        # 필수: YAML 프론트매터 + 지침
│   └── helper.py       # 선택: 지원 파일
├── code-review/
│   ├── SKILL.md
│   └── checklist.md

research_agent 프로젝트용으로 deepagents-cli에서 적응함.
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

from research_agent.skills.load import SkillMetadata, list_skills


class SkillsState(AgentState):
    """스킬 미들웨어용 상태."""

    skills_metadata: NotRequired[list[SkillMetadata]]
    """로드된 스킬 메타데이터 목록 (이름, 설명, 경로)."""


class SkillsStateUpdate(TypedDict):
    """스킬 미들웨어용 상태 업데이트."""

    skills_metadata: list[SkillMetadata]
    """로드된 스킬 메타데이터 목록 (이름, 설명, 경로)."""


# 스킬 시스템 문서 템플릿
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern. You know that skills exist (name + description above), but you only read full instructions when needed:

1. **Identify when a skill applies**: Check if the user's task matches a skill's description.
2. **Read the skill's full instructions**: The skill list above shows exact paths for use with read_file.
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, recommendations, and examples.
4. **Access supporting files**: Skills may include Python scripts, configs, or reference docs. Use absolute paths.

**When to use skills:**
- When the user's request matches a skill's domain (e.g., "research X" → web-research skill)
- When specialized knowledge or structured workflows would help
- When the skill provides proven patterns for complex tasks

**Skills are self-documenting:**
- Each SKILL.md tells you exactly what the skill does and how to use it.
- The skill list above shows full paths to each skill's SKILL.md file.

**Running skill scripts:**
Skills may include Python scripts or other executables. Always use the absolute paths from the skill list.

**Workflow example:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills above → find "web-research" skill with full path
2. Read the skill using the path shown in the list
3. Follow the skill's research workflow (plan → save → delegate → synthesize)
4. Use helper scripts with absolute paths

Note: Skills are tools that make you more capable and consistent. When in doubt, check if there's a skill for the task!
"""


class SkillsMiddleware(AgentMiddleware):
    """에이전트 스킬을 로드하고 노출하기 위한 미들웨어.

    이 미들웨어는 Anthropic의 Agent Skills 패턴을 구현한다:
    - 세션 시작 시: YAML 프론트매터에서 스킬 메타데이터(이름, 설명) 로드
    - 발견 가능성을 위해 시스템 프롬프트에 스킬 목록 주입
    - 스킬이 관련될 때 에이전트가 전체 SKILL.md 콘텐츠 읽기 (점진적 공개)

    사용자 레벨과 프로젝트 레벨 스킬 모두 지원:
    - 프로젝트 스킬: {PROJECT_ROOT}/skills/
    - 프로젝트 스킬이 같은 이름의 사용자 스킬을 오버라이드

    Args:
        skills_dir: 사용자 레벨 스킬 디렉토리 경로 (에이전트별).
        assistant_id: 프롬프트의 경로 참조용 에이전트 식별자.
        project_skills_dir: 프로젝트 레벨 스킬 디렉토리 경로 (선택).
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        skills_dir: str | Path,
        assistant_id: str,
        project_skills_dir: str | Path | None = None,
    ) -> None:
        """스킬 미들웨어를 초기화한다.

        Args:
            skills_dir: 사용자 레벨 스킬 디렉토리 경로.
            assistant_id: 에이전트 식별자.
            project_skills_dir: 프로젝트 레벨 스킬 디렉토리 경로 (선택).
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.assistant_id = assistant_id
        self.project_skills_dir = (
            Path(project_skills_dir).expanduser() if project_skills_dir else None
        )
        # 프롬프트 표시용 경로 저장
        self.user_skills_display = f"~/.deepagents/{assistant_id}/skills"
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _format_skills_locations(self) -> str:
        """시스템 프롬프트 표시용 스킬 위치를 포맷팅한다."""
        locations = [f"**User Skills**: `{self.user_skills_display}`"]
        if self.project_skills_dir:
            locations.append(
                f"**Project Skills**: `{self.project_skills_dir}` (overrides user skills)"
            )
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """시스템 프롬프트 표시용 스킬 메타데이터를 포맷팅한다."""
        if not skills:
            locations = [f"{self.user_skills_display}/"]
            if self.project_skills_dir:
                locations.append(f"{self.project_skills_dir}/")
            return f"(No skills available. You can create skills in {' or '.join(locations)})"

        # 출처별로 스킬 그룹화
        user_skills = [s for s in skills if s["source"] == "user"]
        project_skills = [s for s in skills if s["source"] == "project"]

        lines = []

        # 사용자 스킬 표시
        if user_skills:
            lines.append("**User Skills:**")
            for skill in user_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → To read full instructions: `{skill['path']}`")
            lines.append("")

        # 프로젝트 스킬 표시
        if project_skills:
            lines.append("**Project Skills:**")
            for skill in project_skills:
                lines.append(f"- **{skill['name']}**: {skill['description']}")
                lines.append(f"  → To read full instructions: `{skill['path']}`")

        return "\n".join(lines)

    def before_agent(
        self, state: SkillsState, runtime: Runtime
    ) -> SkillsStateUpdate | None:
        """에이전트 실행 전에 스킬 메타데이터를 로드한다.

        세션 시작 시 한 번 실행되어 사용자 레벨과 프로젝트 레벨
        디렉토리 모두에서 사용 가능한 스킬을 발견한다.

        Args:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.

        Returns:
            skills_metadata가 채워진 업데이트된 상태.
        """
        # 디렉토리 변경을 캐치하기 위해 각 상호작용마다 스킬 다시 로드
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
        """시스템 프롬프트에 스킬 문서를 주입한다.

        스킬 정보가 사용 가능하도록 모든 모델 호출 전에 실행된다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        # 상태에서 스킬 메타데이터 가져오기
        skills_metadata = request.state.get("skills_metadata", [])

        # 스킬 위치와 목록 포맷팅
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # 스킬 문서 포맷팅
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
        """(비동기) 시스템 프롬프트에 스킬 문서를 주입한다.

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            핸들러의 모델 응답.
        """
        # state_schema로 인해 상태가 SkillsState임이 보장됨
        state = cast("SkillsState", request.state)
        skills_metadata = state.get("skills_metadata", [])

        # 스킬 위치와 목록 포맷팅
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        # 스킬 문서 포맷팅
        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        # 시스템 프롬프트에 주입
        if request.system_prompt:
            system_prompt = request.system_prompt + "\n\n" + skills_section
        else:
            system_prompt = skills_section

        return await handler(request.override(system_prompt=system_prompt))
