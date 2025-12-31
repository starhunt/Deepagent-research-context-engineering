"""전문화된 에이전트 타입 관리를 위한 SubAgent 레지스트리.

이 모듈은 Claude Code의 subagent_type 시스템에서 영감을 받은 레지스트리 패턴을 제공하며,
SubAgent 명세의 동적 등록과 조회를 허용한다.

레지스트리 지원 기능:
- 타입 기반 에이전트 조회 (이름으로)
- 능력 기반 필터링 (태그로)
- 런타임 에이전트 발견

Example:
    registry = SubAgentRegistry()
    registry.register(RESEARCHER_AGENT)
    registry.register(EXPLORER_AGENT)

    # 특정 에이전트 가져오기
    researcher = registry.get("researcher")

    # "research" 능력을 가진 모든 에이전트 가져오기
    research_agents = registry.get_by_capability("research")
"""

from typing import Any, TypedDict

from typing_extensions import NotRequired


class SubAgentSpec(TypedDict):
    """SubAgent 명세.

    DeepAgents SubAgent TypedDict 패턴을 따르며
    능력 기반 라우팅을 위한 추가 필드를 포함한다.
    """

    name: str
    """SubAgent의 고유 식별자 (subagent_type으로 사용)."""

    description: str
    """위임 결정을 위해 메인 에이전트에게 표시되는 설명."""

    system_prompt: str
    """SubAgent 동작을 정의하는 시스템 프롬프트."""

    tools: list[Any]
    """이 SubAgent에서 사용 가능한 도구."""

    model: NotRequired[str]
    """선택적 모델 오버라이드 (기본값은 부모의 모델)."""

    capabilities: NotRequired[list[str]]
    """필터링용 능력 태그 (예: ['research', 'web'])."""


class SubAgentRegistry:
    """SubAgent 명세 관리를 위한 레지스트리.

    이 클래스는 Claude Code 스타일의 SubAgent 관리를 제공한다:
    - 에이전트 등록 및 등록 해제
    - 이름 기반 조회 (subagent_type 매칭)
    - 능력 기반 필터링

    Example:
        registry = SubAgentRegistry()

        # 에이전트 등록
        registry.register({
            "name": "researcher",
            "description": "Deep web research",
            "system_prompt": "...",
            "tools": [...],
            "capabilities": ["research", "web"],
        })

        # 이름으로 조회
        agent = registry.get("researcher")

        # 능력으로 필터링
        web_agents = registry.get_by_capability("web")
    """

    def __init__(self) -> None:
        """빈 레지스트리를 초기화한다."""
        self._agents: dict[str, SubAgentSpec] = {}

    def register(self, agent_spec: SubAgentSpec) -> None:
        """SubAgent 명세를 등록한다.

        Args:
            agent_spec: SubAgent 명세 딕셔너리.

        Raises:
            ValueError: 같은 이름의 에이전트가 이미 등록된 경우.
        """
        name = agent_spec["name"]
        if name in self._agents:
            msg = f"SubAgent '{name}'은(는) 이미 등록되어 있습니다"
            raise ValueError(msg)
        self._agents[name] = agent_spec

    def unregister(self, name: str) -> None:
        """레지스트리에서 SubAgent를 제거한다.

        Args:
            name: 제거할 SubAgent 이름.

        Raises:
            KeyError: 에이전트를 찾을 수 없는 경우.
        """
        if name not in self._agents:
            msg = f"SubAgent '{name}'을(를) 레지스트리에서 찾을 수 없습니다"
            raise KeyError(msg)
        del self._agents[name]

    def get(self, name: str) -> SubAgentSpec | None:
        """이름으로 SubAgent 명세를 가져온다.

        Args:
            name: SubAgent 이름 (subagent_type).

        Returns:
            찾으면 SubAgent 명세, 그렇지 않으면 None.
        """
        return self._agents.get(name)

    def list_all(self) -> list[SubAgentSpec]:
        """등록된 모든 SubAgent 명세를 나열한다.

        Returns:
            모든 SubAgent 명세 목록.
        """
        return list(self._agents.values())

    def list_names(self) -> list[str]:
        """등록된 모든 SubAgent 이름을 나열한다.

        Returns:
            SubAgent 이름 목록.
        """
        return list(self._agents.keys())

    def get_by_capability(self, capability: str) -> list[SubAgentSpec]:
        """특정 능력을 가진 SubAgent를 가져온다.

        Args:
            capability: 필터링할 능력 태그.

        Returns:
            지정된 능력을 가진 SubAgent 목록.
        """
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.get("capabilities", [])
        ]

    def get_descriptions(self) -> dict[str, str]:
        """에이전트 이름과 설명의 매핑을 가져온다.

        메인 오케스트레이터에 사용 가능한 에이전트를 표시할 때 유용하다.

        Returns:
            에이전트 이름을 설명에 매핑하는 딕셔너리.
        """
        return {name: agent["description"] for name, agent in self._agents.items()}

    def __contains__(self, name: str) -> bool:
        """SubAgent가 등록되어 있는지 확인한다.

        Args:
            name: 확인할 SubAgent 이름.

        Returns:
            에이전트가 등록되어 있으면 True.
        """
        return name in self._agents

    def __len__(self) -> int:
        """등록된 SubAgent 수를 반환한다."""
        return len(self._agents)
