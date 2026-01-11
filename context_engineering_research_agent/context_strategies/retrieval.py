"""Context Retrieval 전략 구현.

## 개요

Context Retrieval은 필요한 정보를 선택적으로 로드하여
컨텍스트 윈도우를 효율적으로 사용하는 전략입니다.

## 핵심 원리

1. 벡터 DB나 복잡한 인덱싱 없이 직접 파일 검색
2. grep/glob 기반의 단순하고 빠른 패턴 매칭
3. 필요한 파일/내용만 선택적으로 로드

## DeepAgents 구현

FilesystemMiddleware에서 제공하는 도구들:
- `read_file`: 파일 내용 읽기 (offset/limit으로 부분 읽기 지원)
- `grep`: 텍스트 패턴 검색
- `glob`: 파일명 패턴 매칭
- `ls`: 디렉토리 목록 조회

## 벡터 검색을 사용하지 않는 이유

1. **단순성**: 추가 인프라 불필요
2. **결정성**: 정확한 매칭, 모호함 없음
3. **속도**: 인덱싱 오버헤드 없음
4. **디버깅 용이**: 검색 결과 예측 가능

## 사용 예시

```python
# 파일 검색
grep(pattern="TODO", glob="*.py")

# 파일 목록
glob(pattern="**/*.md")

# 부분 읽기
read_file(path="/data.txt", offset=100, limit=50)
```
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool


@dataclass
class RetrievalConfig:
    """Context Retrieval 설정."""

    default_read_limit: int = 500
    """read_file의 기본 줄 수 제한."""

    max_grep_results: int = 100
    """grep 결과 최대 개수."""

    max_glob_results: int = 100
    """glob 결과 최대 개수."""

    truncate_line_length: int = 2000
    """줄 길이 제한 (초과 시 자름)."""


@dataclass
class RetrievalResult:
    """검색 결과 메타데이터."""

    tool_used: str
    """사용된 도구 이름."""

    query: str
    """검색 쿼리/패턴."""

    result_count: int
    """결과 개수."""

    was_truncated: bool = False
    """결과가 잘렸는지 여부."""


class ContextRetrievalStrategy(AgentMiddleware):
    """Context Retrieval 전략 구현.

    grep/glob 기반의 단순하고 빠른 검색으로
    필요한 정보만 선택적으로 로드합니다.

    ## 동작 원리

    1. 파일시스템에서 직접 패턴 매칭
    2. 결과 개수 제한으로 컨텍스트 오버플로우 방지
    3. 부분 읽기로 대용량 파일 효율적 처리

    ## 제공 도구

    - read_file: 파일 읽기 (offset/limit 지원)
    - grep: 텍스트 패턴 검색
    - glob: 파일명 패턴 매칭

    ## 벡터 DB를 사용하지 않는 이유

    DeepAgents는 의도적으로 벡터 검색 대신 직접 파일 검색을 선택했습니다:
    - 결정적이고 예측 가능한 결과
    - 추가 인프라 불필요
    - 디버깅 용이

    Args:
        config: Retrieval 설정. None이면 기본값 사용.
        backend_factory: 파일 작업용 백엔드 팩토리 함수.
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        backend_factory: Callable[[ToolRuntime], Any] | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._backend_factory = backend_factory
        self.tools = self._create_tools()

    def _create_tools(self) -> list[BaseTool]:
        """검색 도구들을 생성합니다."""
        return [
            self._create_read_file_tool(),
            self._create_grep_tool(),
            self._create_glob_tool(),
        ]

    def _create_read_file_tool(self) -> BaseTool:
        """read_file 도구를 생성합니다."""
        config = self.config
        backend_factory = self._backend_factory

        def read_file(
            file_path: str,
            runtime: ToolRuntime,
            offset: int = 0,
            limit: int | None = None,
        ) -> str:
            """파일을 읽습니다.

            Args:
                file_path: 읽을 파일의 절대 경로.
                offset: 시작 줄 번호 (0부터 시작).
                limit: 읽을 최대 줄 수. 기본값은 설정에 따름.

            Returns:
                줄 번호가 포함된 파일 내용.
            """
            if backend_factory is None:
                return "백엔드가 설정되지 않았습니다."

            backend = backend_factory(runtime)
            actual_limit = limit or config.default_read_limit
            return backend.read(file_path, offset=offset, limit=actual_limit)

        return StructuredTool.from_function(
            name="read_file",
            description=f"""파일을 읽습니다.

사용법:
- file_path: 절대 경로 필수
- offset: 시작 줄 (기본 0)
- limit: 읽을 줄 수 (기본 {config.default_read_limit})

대용량 파일은 offset/limit으로 부분 읽기를 권장합니다.""",
            func=read_file,
        )

    def _create_grep_tool(self) -> BaseTool:
        """Grep 도구를 생성합니다."""
        config = self.config
        backend_factory = self._backend_factory

        def grep(
            pattern: str,
            runtime: ToolRuntime,
            path: str | None = None,
            glob_pattern: str | None = None,
            output_mode: Literal[
                "files_with_matches", "content", "count"
            ] = "files_with_matches",
        ) -> str:
            """텍스트 패턴을 검색합니다.

            Args:
                pattern: 검색할 텍스트 (정규식 아님).
                path: 검색 시작 디렉토리.
                glob_pattern: 파일 필터 (예: "*.py").
                output_mode: 출력 형식.

            Returns:
                검색 결과.
            """
            if backend_factory is None:
                return "백엔드가 설정되지 않았습니다."

            backend = backend_factory(runtime)
            raw_results = backend.grep_raw(pattern, path=path, glob=glob_pattern)

            if isinstance(raw_results, str):
                return raw_results

            truncated = raw_results[: config.max_grep_results]

            if output_mode == "files_with_matches":
                files = list(set(r.get("path", "") for r in truncated))
                return "\n".join(files)
            elif output_mode == "count":
                from collections import Counter

                counts = Counter(r.get("path", "") for r in truncated)
                return "\n".join(f"{path}: {count}" for path, count in counts.items())
            else:
                lines = []
                for r in truncated:
                    path = r.get("path", "")
                    line_num = r.get("line_number", 0)
                    content = r.get("content", "")[: config.truncate_line_length]
                    lines.append(f"{path}:{line_num}: {content}")
                return "\n".join(lines)

        return StructuredTool.from_function(
            name="grep",
            description=f"""텍스트 패턴을 검색합니다.

사용법:
- pattern: 검색할 텍스트 (리터럴 문자열)
- path: 검색 디렉토리 (선택)
- glob_pattern: 파일 필터 예: "*.py" (선택)
- output_mode: files_with_matches | content | count

최대 {config.max_grep_results}개 결과를 반환합니다.""",
            func=grep,
        )

    def _create_glob_tool(self) -> BaseTool:
        """Glob 도구를 생성합니다."""
        config = self.config
        backend_factory = self._backend_factory

        def glob(
            pattern: str,
            runtime: ToolRuntime,
            path: str = "/",
        ) -> str:
            """파일명 패턴으로 파일을 찾습니다.

            Args:
                pattern: glob 패턴 (예: "**/*.py").
                path: 검색 시작 경로.

            Returns:
                매칭된 파일 경로 목록.
            """
            if backend_factory is None:
                return "백엔드가 설정되지 않았습니다."

            backend = backend_factory(runtime)
            infos = backend.glob_info(pattern, path=path)

            paths = [fi.get("path", "") for fi in infos[: config.max_glob_results]]
            return "\n".join(paths)

        return StructuredTool.from_function(
            name="glob",
            description=f"""파일명 패턴으로 파일을 찾습니다.

사용법:
- pattern: glob 패턴 (*, **, ? 지원)
- path: 검색 시작 경로 (기본 "/")

예시:
- "**/*.py": 모든 Python 파일
- "src/**/*.ts": src 아래 모든 TypeScript 파일

최대 {config.max_glob_results}개 결과를 반환합니다.""",
            func=glob,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출을 래핑합니다 (기본 동작)."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """비동기 모델 호출을 래핑합니다 (기본 동작)."""
        return await handler(request)


RETRIEVAL_SYSTEM_PROMPT = """## Context Retrieval 도구

파일시스템에서 정보를 검색할 수 있습니다.

도구:
- read_file: 파일 읽기 (offset/limit으로 부분 읽기)
- grep: 텍스트 패턴 검색
- glob: 파일명 패턴 매칭

사용 팁:
1. 대용량 파일은 먼저 구조 파악 (limit=100)
2. grep으로 관련 파일 찾은 후 read_file
3. glob으로 파일 위치 확인 후 탐색
"""
