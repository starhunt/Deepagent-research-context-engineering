"""자율적 연구 에이전트 팩토리.

이 모듈은 자체 계획, 반성, 컨텍스트 관리 기능을 갖춘
독립적인 연구 DeepAgent를 생성합니다.
"""

from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from research_agent.researcher.prompts import AUTONOMOUS_RESEARCHER_INSTRUCTIONS
from research_agent.tools import tavily_search, think_tool


def create_researcher_agent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> CompiledStateGraph:
    """자율적 연구 DeepAgent를 생성한다.

    이 에이전트는 다음 기능을 자체적으로 보유한다:
    - 계획 루프 (TodoListMiddleware를 통한 write_todos)
    - 연구 루프 (tavily_search + think_tool)
    - 컨텍스트 관리 (SummarizationMiddleware)
    - 중간 결과 저장을 위한 파일 접근 (FilesystemMiddleware)

    본질적으로 자율적으로 작동하는 "연구 SubGraph"이다.

    Args:
        model: 사용할 LLM. 기본값은 temperature=0인 gpt-4.1.
        backend: 파일 작업용 백엔드. 제공되면
                 연구자가 중간 결과를 파일시스템에 저장할 수 있다.

    Returns:
        CompiledStateGraph: 독립적으로 사용하거나 오케스트레이터의
        CompiledSubAgent로 사용할 수 있는 완전 자율적 연구 에이전트.

    Example:
        # 독립 사용
        researcher = create_researcher_agent()
        result = researcher.invoke({
            "messages": [HumanMessage("양자 컴퓨팅 트렌드 연구")]
        })

        # 오케스트레이터의 SubAgent로 사용
        subagent = get_researcher_subagent()
        orchestrator = create_deep_agent(subagents=[subagent, ...])
    """
    if model is None:
        model = ChatOpenAI(model="gpt-4.1", temperature=0.0)

    # 현재 날짜로 프롬프트 포맷팅
    current_date = datetime.now().strftime("%Y-%m-%d")
    formatted_prompt = AUTONOMOUS_RESEARCHER_INSTRUCTIONS.format(date=current_date)

    return create_deep_agent(
        model=model,
        tools=[tavily_search, think_tool],
        system_prompt=formatted_prompt,
        backend=backend,
    )


def get_researcher_subagent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> dict:
    """오케스트레이터에서 사용할 CompiledSubAgent로 연구자를 가져온다.

    이 함수는 자율적 연구 에이전트를 생성하고 SubAgentMiddleware가
    기대하는 CompiledSubAgent 형식으로 래핑한다.

    Args:
        model: 사용할 LLM. 기본값은 gpt-4.1.
        backend: 파일 작업용 백엔드.

    Returns:
        dict: 다음 키를 가진 CompiledSubAgent:
            - name: "researcher"
            - description: 오케스트레이터가 위임 결정 시 사용
            - runnable: 자율적 연구 에이전트

    Example:
        from research_agent.researcher import get_researcher_subagent

        researcher = get_researcher_subagent(model=model, backend=backend)

        agent = create_deep_agent(
            model=model,
            subagents=[researcher, explorer, synthesizer],
            ...
        )
    """
    researcher = create_researcher_agent(model=model, backend=backend)

    return {
        "name": "researcher",
        "description": (
            "Autonomous deep research agent with self-planning and "
            "'breadth-first, depth-second' methodology. Use for comprehensive "
            "topic research requiring multiple search iterations and synthesis. "
            "The agent plans its own research phases, reflects after each search, "
            "and synthesizes findings into structured output. "
            "Best for: complex topics, multi-faceted questions, trend analysis."
        ),
        "runnable": researcher,
    }
