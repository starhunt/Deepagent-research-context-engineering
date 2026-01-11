"""Context Isolation 전략 구현.

SubAgent를 통해 독립된 컨텍스트 윈도우에서 작업을 수행하여
메인 에이전트의 컨텍스트를 오염시키지 않는 전략입니다.

DeepAgents의 SubAgentMiddleware에서 task() 도구로 구현되어 있습니다.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command


class SubAgentSpec(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]


class CompiledSubAgentSpec(TypedDict):
    name: str
    description: str
    runnable: Runnable


@dataclass
class IsolationConfig:
    default_model: str | BaseChatModel = "gpt-4.1"
    include_general_purpose_agent: bool = True
    excluded_state_keys: tuple[str, ...] = ("messages", "todos", "structured_response")


@dataclass
class IsolationResult:
    subagent_name: str
    was_successful: bool
    result_length: int
    error: str | None = None


class ContextIsolationStrategy(AgentMiddleware):
    """SubAgent를 통한 Context Isolation 구현.

    Args:
        config: Isolation 설정
        subagents: SubAgent 명세 목록
        agent_factory: 에이전트 생성 팩토리 함수
    """

    def __init__(
        self,
        config: IsolationConfig | None = None,
        subagents: list[SubAgentSpec | CompiledSubAgentSpec] | None = None,
        agent_factory: Callable[..., Runnable] | None = None,
    ) -> None:
        self.config = config or IsolationConfig()
        self._subagents = subagents or []
        self._agent_factory = agent_factory
        self._compiled_agents: dict[str, Runnable] = {}
        self.tools = [self._create_task_tool()]

    def _compile_subagents(self) -> dict[str, Runnable]:
        if self._compiled_agents:
            return self._compiled_agents

        agents: dict[str, Runnable] = {}

        for spec in self._subagents:
            if "runnable" in spec:
                compiled = spec  # type: ignore
                agents[compiled["name"]] = compiled["runnable"]
            elif self._agent_factory:
                simple = spec  # type: ignore
                agents[simple["name"]] = self._agent_factory(
                    model=simple.get("model", self.config.default_model),
                    system_prompt=simple["system_prompt"],
                    tools=simple["tools"],
                    middleware=simple.get("middleware", []),
                )

        self._compiled_agents = agents
        return agents

    def _get_subagent_descriptions(self) -> str:
        descriptions = []
        for spec in self._subagents:
            descriptions.append(f"- {spec['name']}: {spec['description']}")
        return "\n".join(descriptions)

    def _prepare_subagent_state(
        self, state: dict[str, Any], task_description: str
    ) -> dict[str, Any]:
        filtered = {
            k: v for k, v in state.items() if k not in self.config.excluded_state_keys
        }
        filtered["messages"] = [HumanMessage(content=task_description)]
        return filtered

    def _create_task_tool(self) -> BaseTool:
        strategy = self

        def task(
            description: str,
            subagent_type: str,
            runtime: ToolRuntime,
        ) -> str | Command:
            agents = strategy._compile_subagents()

            if subagent_type not in agents:
                allowed = ", ".join(f"`{k}`" for k in agents)
                return f"SubAgent '{subagent_type}'가 존재하지 않습니다. 사용 가능: {allowed}"

            subagent = agents[subagent_type]
            subagent_state = strategy._prepare_subagent_state(
                runtime.state, description
            )

            result = subagent.invoke(subagent_state, runtime.config)

            final_message = result["messages"][-1].text.rstrip()

            state_update = {
                k: v
                for k, v in result.items()
                if k not in strategy.config.excluded_state_keys
            }

            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(final_message, tool_call_id=runtime.tool_call_id)
                    ],
                }
            )

        async def atask(
            description: str,
            subagent_type: str,
            runtime: ToolRuntime,
        ) -> str | Command:
            agents = strategy._compile_subagents()

            if subagent_type not in agents:
                allowed = ", ".join(f"`{k}`" for k in agents)
                return f"SubAgent '{subagent_type}'가 존재하지 않습니다. 사용 가능: {allowed}"

            subagent = agents[subagent_type]
            subagent_state = strategy._prepare_subagent_state(
                runtime.state, description
            )

            result = await subagent.ainvoke(subagent_state, runtime.config)

            final_message = result["messages"][-1].text.rstrip()

            state_update = {
                k: v
                for k, v in result.items()
                if k not in strategy.config.excluded_state_keys
            }

            return Command(
                update={
                    **state_update,
                    "messages": [
                        ToolMessage(final_message, tool_call_id=runtime.tool_call_id)
                    ],
                }
            )

        subagent_list = self._get_subagent_descriptions()

        return StructuredTool.from_function(
            name="task",
            func=task,
            coroutine=atask,
            description=f"""SubAgent에게 작업을 위임합니다.

사용 가능한 SubAgent:
{subagent_list}

사용법:
- description: 위임할 작업 상세 설명
- subagent_type: 사용할 SubAgent 이름

SubAgent는 독립된 컨텍스트에서 실행되어 메인 에이전트의 컨텍스트를 오염시키지 않습니다.
복잡하고 다단계 작업에 적합합니다.""",
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request)


ISOLATION_SYSTEM_PROMPT = """## Context Isolation (task 도구)

task 도구로 SubAgent에게 작업을 위임할 수 있습니다.

장점:
- 독립된 컨텍스트 윈도우
- 메인 컨텍스트 오염 방지
- 복잡한 작업의 격리 처리

사용 시점:
- 다단계 복잡한 작업
- 대량의 컨텍스트가 필요한 연구
- 병렬 처리가 가능한 독립 작업
"""
