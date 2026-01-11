"""Docker 공유 작업공간 백엔드.

여러 SubAgent가 동일한 Docker 컨테이너 작업공간을 공유하는 백엔드입니다.

## 설계 배경

DeepAgents의 기본 구조에서 SubAgent들은 독립된 컨텍스트를 가지지만,
파일시스템을 공유해야 하는 경우가 있습니다:
- 연구 결과물 공유
- 중간 생성물 활용
- 협업 워크플로우

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                   Main Agent (Orchestrator)                 │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │SubAgent A │  │SubAgent B │  │SubAgent C │               │
│  │(Research) │  │(Analysis) │  │(Synthesis)│               │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │
│        │              │              │                      │
│        └──────────────┼──────────────┘                      │
│                       ▼                                     │
│              SharedDockerBackend                            │
│                       │                                     │
└───────────────────────┼─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Docker Container                          │
│  ┌─────────────────────────────────────────────────────────│
│  │  /workspace (공유 작업 디렉토리)                         │
│  │   ├── research/     (SubAgent A 출력)                   │
│  │   ├── analysis/     (SubAgent B 출력)                   │
│  │   └── synthesis/    (SubAgent C 출력)                   │
│  └─────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────┘
```

## 보안 고려사항

1. **컨테이너 격리**: 호스트 시스템과 격리
2. **볼륨 마운트**: 필요한 디렉토리만 마운트
3. **네트워크 정책**: 필요시 네트워크 격리
4. **리소스 제한**: CPU/메모리 제한 설정
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DockerConfig:
    image: str = "python:3.11-slim"
    workspace_path: str = "/workspace"
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    network_mode: str = "none"
    auto_remove: bool = True
    timeout_seconds: int = 300


@dataclass
class ExecuteResponse:
    output: str
    exit_code: int | None = None
    truncated: bool = False
    error: str | None = None


@dataclass
class WriteResult:
    path: str
    error: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    path: str
    occurrences: int = 0
    error: str | None = None
    files_update: dict[str, Any] | None = None


class SharedDockerBackend:
    """여러 SubAgent가 공유하는 Docker 작업공간 백엔드.

    Args:
        config: Docker 설정
        container_id: 기존 컨테이너 ID (재사용 시)
    """

    def __init__(
        self,
        config: DockerConfig | None = None,
        container_id: str | None = None,
    ) -> None:
        self.config = config or DockerConfig()
        self._container_id = container_id
        self._docker_client: Any = None

    def _get_docker_client(self) -> Any:
        if self._docker_client is None:
            try:
                import docker

                self._docker_client = docker.from_env()
            except ImportError:
                raise RuntimeError(
                    "docker 패키지가 설치되지 않았습니다: pip install docker"
                )
        return self._docker_client

    def _ensure_container(self) -> str:
        if self._container_id:
            return self._container_id

        client = self._get_docker_client()
        container = client.containers.run(
            self.config.image,
            command="tail -f /dev/null",
            detach=True,
            mem_limit=self.config.memory_limit,
            nano_cpus=int(self.config.cpu_limit * 1e9),
            network_mode=self.config.network_mode,
            auto_remove=self.config.auto_remove,
        )
        self._container_id = container.id
        return container.id

    def execute(self, command: str) -> ExecuteResponse:
        try:
            container_id = self._ensure_container()
            client = self._get_docker_client()
            container = client.containers.get(container_id)

            exec_result = container.exec_run(
                command,
                workdir=self.config.workspace_path,
            )

            output = exec_result.output.decode("utf-8", errors="replace")
            truncated = len(output) > 100000
            if truncated:
                output = output[:100000] + "\n[출력이 잘렸습니다...]"

            return ExecuteResponse(
                output=output,
                exit_code=exec_result.exit_code,
                truncated=truncated,
            )
        except Exception as e:
            return ExecuteResponse(
                output="",
                exit_code=1,
                error=str(e),
            )

    async def aexecute(self, command: str) -> ExecuteResponse:
        return self.execute(command)

    def read(self, path: str, offset: int = 0, limit: int = 500) -> str:
        full_path = f"{self.config.workspace_path}{path}"
        result = self.execute(f"sed -n '{offset + 1},{offset + limit}p' {full_path}")

        if result.error:
            return f"파일 읽기 오류: {result.error}"

        return result.output

    async def aread(self, path: str, offset: int = 0, limit: int = 500) -> str:
        return self.read(path, offset, limit)

    def write(self, path: str, content: str) -> WriteResult:
        full_path = f"{self.config.workspace_path}{path}"

        dir_path = "/".join(full_path.split("/")[:-1])
        self.execute(f"mkdir -p {dir_path}")

        escaped_content = content.replace("'", "'\\''")
        result = self.execute(f"echo '{escaped_content}' > {full_path}")

        if result.exit_code != 0:
            return WriteResult(path=path, error=result.output or result.error)

        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        return self.write(path, content)

    def ls_info(self, path: str) -> list[dict[str, Any]]:
        full_path = f"{self.config.workspace_path}{path}"
        result = self.execute(f"ls -la {full_path}")

        if result.error:
            return []

        files = []
        for line in result.output.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 9:
                name = " ".join(parts[8:])
                files.append(
                    {
                        "path": f"{path}/{name}".replace("//", "/"),
                        "is_dir": line.startswith("d"),
                    }
                )

        return files

    async def als_info(self, path: str) -> list[dict[str, Any]]:
        return self.ls_info(path)

    def cleanup(self) -> None:
        if self._container_id and self._docker_client:
            try:
                container = self._docker_client.containers.get(self._container_id)
                container.stop()
            except Exception:
                pass
            self._container_id = None

    def __enter__(self) -> "SharedDockerBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()


SHARED_WORKSPACE_DESIGN_DOC = """
# 공유 작업공간 설계

## 문제점

DeepAgents의 SubAgent들은 독립된 컨텍스트를 가지지만,
연구 워크플로우에서는 파일시스템 공유가 필요한 경우가 많습니다.

예시:
1. Research Agent가 수집한 데이터를 Analysis Agent가 처리
2. 여러 Agent가 생성한 결과물을 Synthesis Agent가 통합
3. 중간 체크포인트를 다른 Agent가 이어서 작업

## 해결책: SharedDockerBackend

1. **단일 컨테이너**: 모든 SubAgent가 동일한 Docker 컨테이너 사용
2. **격리된 디렉토리**: 각 SubAgent는 자신의 디렉토리에서 작업
3. **공유 영역**: /workspace/shared 같은 공유 디렉토리 운영

## 장점

- 파일 복사 오버헤드 없음
- 실시간 결과 공유
- 일관된 실행 환경

## 단점

- 컨테이너 수명 관리 필요
- 동시 접근 충돌 가능성
- 보안 경계가 SubAgent 간에는 없음
"""
