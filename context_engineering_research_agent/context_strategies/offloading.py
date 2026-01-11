"""Context Offloading 전략 구현.

## 개요

Context Offloading은 대용량 도구 결과를 파일시스템으로 축출하여
컨텍스트 윈도우 오버플로우를 방지하는 전략입니다.

## 핵심 원리

1. 도구 실행 결과가 특정 토큰 임계값을 초과하면 자동으로 파일로 저장
2. 원본 메시지는 파일 경로 참조로 대체
3. 에이전트가 필요할 때 read_file로 데이터 로드

## DeepAgents 구현

FilesystemMiddleware의 `_intercept_large_tool_result` 메서드에서 구현:
- `tool_token_limit_before_evict`: 축출 임계값 (기본 20,000 토큰)
- `/large_tool_results/{tool_call_id}` 경로에 저장
- 처음 10줄 미리보기 제공

## 장점

- 컨텍스트 윈도우 절약
- 대용량 데이터 처리 가능
- 선택적 로딩으로 효율성 증가

## 사용 예시

```python
from deepagents.middleware.filesystem import FilesystemMiddleware

middleware = FilesystemMiddleware(
    tool_token_limit_before_evict=15000  # 15,000 토큰 초과 시 축출
)
```
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@dataclass
class OffloadingConfig:
    """Context Offloading 설정."""

    token_limit_before_evict: int = 20000
    """도구 결과를 파일로 축출하기 전 토큰 임계값. 기본값 20,000."""

    eviction_path_prefix: str = "/large_tool_results"
    """축출된 파일이 저장될 경로 접두사."""

    preview_lines: int = 10
    """축출 시 포함할 미리보기 줄 수."""

    chars_per_token: int = 4
    """토큰당 문자 수 근사값 (보수적 추정)."""


@dataclass
class OffloadingResult:
    """Offloading 처리 결과."""

    was_offloaded: bool
    """실제로 축출이 발생했는지 여부."""

    original_size: int
    """원본 콘텐츠 크기 (문자 수)."""

    file_path: str | None = None
    """축출된 파일 경로 (축출된 경우)."""

    preview: str | None = None
    """축출 시 제공되는 미리보기."""


class ContextOffloadingStrategy(AgentMiddleware):
    """Context Offloading 전략 구현.

    대용량 도구 결과를 파일시스템으로 자동 축출하여
    컨텍스트 윈도우 오버플로우를 방지합니다.

    ## 동작 원리

    1. wrap_tool_call에서 도구 실행 결과를 가로챔
    2. 결과 크기가 임계값을 초과하면 파일로 저장
    3. 원본 메시지를 파일 경로 참조로 대체
    4. 에이전트는 필요시 read_file로 데이터 로드

    ## DeepAgents FilesystemMiddleware와의 관계

    이 클래스는 FilesystemMiddleware의 offloading 로직을
    명시적으로 분리하여 전략 패턴으로 구현한 것입니다.

    Args:
        config: Offloading 설정. None이면 기본값 사용.
        backend_factory: 파일 저장용 백엔드 팩토리 함수.
    """

    def __init__(
        self,
        config: OffloadingConfig | None = None,
        backend_factory: Callable[[ToolRuntime], Any] | None = None,
    ) -> None:
        self.config = config or OffloadingConfig()
        self._backend_factory = backend_factory

    def _estimate_tokens(self, content: str) -> int:
        """콘텐츠의 토큰 수를 추정합니다.

        보수적인 추정값을 사용하여 조기 축출을 방지합니다.
        실제 토큰 수는 모델과 콘텐츠에 따라 다릅니다.
        """
        return len(content) // self.config.chars_per_token

    def _should_offload(self, content: str) -> bool:
        """주어진 콘텐츠가 축출 대상인지 판단합니다."""
        estimated_tokens = self._estimate_tokens(content)
        return estimated_tokens > self.config.token_limit_before_evict

    def _create_preview(self, content: str) -> str:
        """축출될 콘텐츠의 미리보기를 생성합니다."""
        lines = content.splitlines()[: self.config.preview_lines]
        truncated_lines = [line[:1000] for line in lines]
        return "\n".join(f"{i + 1:5}\t{line}" for i, line in enumerate(truncated_lines))

    def _create_offload_message(
        self, tool_call_id: str, file_path: str, preview: str
    ) -> str:
        """축출 후 대체 메시지를 생성합니다."""
        return f"""도구 결과가 너무 커서 파일시스템에 저장되었습니다.

경로: {file_path}

read_file 도구로 결과를 읽을 수 있습니다.
대용량 결과의 경우 offset과 limit 파라미터로 부분 읽기를 권장합니다.

처음 {self.config.preview_lines}줄 미리보기:
{preview}
"""

    def process_tool_result(
        self,
        tool_result: ToolMessage,
        runtime: ToolRuntime,
    ) -> tuple[ToolMessage | Command, OffloadingResult]:
        """도구 결과를 처리하고 필요시 축출합니다.

        Args:
            tool_result: 원본 도구 실행 결과.
            runtime: 도구 런타임 컨텍스트.

        Returns:
            처리된 메시지와 Offloading 결과 튜플.
        """
        content = (
            tool_result.content
            if isinstance(tool_result.content, str)
            else str(tool_result.content)
        )

        result = OffloadingResult(
            was_offloaded=False,
            original_size=len(content),
        )

        if not self._should_offload(content):
            return tool_result, result

        if self._backend_factory is None:
            return tool_result, result

        backend = self._backend_factory(runtime)

        sanitized_id = self._sanitize_tool_call_id(tool_result.tool_call_id)
        file_path = f"{self.config.eviction_path_prefix}/{sanitized_id}"

        write_result = backend.write(file_path, content)
        if write_result.error:
            return tool_result, result

        preview = self._create_preview(content)
        replacement_text = self._create_offload_message(
            tool_result.tool_call_id, file_path, preview
        )

        result.was_offloaded = True
        result.file_path = file_path
        result.preview = preview

        if write_result.files_update is not None:
            return Command(
                update={
                    "files": write_result.files_update,
                    "messages": [
                        ToolMessage(
                            content=replacement_text,
                            tool_call_id=tool_result.tool_call_id,
                        )
                    ],
                }
            ), result

        return ToolMessage(
            content=replacement_text,
            tool_call_id=tool_result.tool_call_id,
        ), result

    def _sanitize_tool_call_id(self, tool_call_id: str) -> str:
        """파일명에 안전한 tool_call_id로 변환합니다."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_call_id)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 호출을 래핑하여 결과를 가로채고 필요시 축출합니다."""
        tool_result = handler(request)

        if isinstance(tool_result, ToolMessage):
            processed, _ = self.process_tool_result(tool_result, request.runtime)
            return processed

        return tool_result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """비동기 도구 호출을 래핑합니다."""
        tool_result = await handler(request)

        if isinstance(tool_result, ToolMessage):
            processed, _ = self.process_tool_result(tool_result, request.runtime)
            return processed

        return tool_result


OFFLOADING_SYSTEM_PROMPT = """## Context Offloading 안내

대용량 도구 결과는 자동으로 파일시스템에 저장됩니다.

결과가 축출된 경우:
1. 파일 경로와 미리보기가 제공됩니다
2. read_file(path, offset=0, limit=100)으로 부분 읽기하세요
3. 전체 내용이 필요한 경우에만 전체 파일을 읽으세요

이 방식으로 컨텍스트 윈도우를 효율적으로 관리할 수 있습니다.
"""
