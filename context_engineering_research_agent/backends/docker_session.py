"""Docker 샌드박스 세션 관리 모듈.

요청 단위로 Docker 컨테이너를 생성하고, 모든 subagent가 동일한 /workspace를 공유합니다.
"""

from __future__ import annotations

import asyncio
from typing import Any

from context_engineering_research_agent.backends.docker_sandbox import (
    DockerSandboxBackend,
)
from context_engineering_research_agent.backends.workspace_protocol import (
    META_DIR,
    SHARED_DIR,
    WORKSPACE_ROOT,
)


class DockerSandboxSession:
    """Docker 컨테이너 라이프사이클을 관리하는 세션.

    컨테이너는 요청 단위로 생성되며, SubAgent는 동일한 컨테이너를 공유합니다.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        workspace_root: str = WORKSPACE_ROOT,
    ) -> None:
        self.image = image
        self.workspace_root = workspace_root
        self._docker_client: Any | None = None
        self._container: Any | None = None
        self._backend: DockerSandboxBackend | None = None

    def _get_docker_client(self) -> Any:
        if self._docker_client is None:
            try:
                import docker
            except ImportError as exc:
                raise RuntimeError(
                    "docker 패키지가 설치되지 않았습니다: pip install docker"
                ) from exc
            try:
                self._docker_client = docker.from_env()
            except Exception as exc:
                docker_exception = getattr(
                    getattr(docker, "errors", None), "DockerException", None
                )
                if docker_exception and isinstance(exc, docker_exception):
                    raise RuntimeError(f"Docker 클라이언트 초기화 실패: {exc}") from exc
                raise RuntimeError(f"Docker 클라이언트 초기화 실패: {exc}") from exc
        return self._docker_client

    async def start(self) -> None:
        """보안 옵션이 적용된 컨테이너를 생성합니다."""
        if self._container is not None:
            return

        client = self._get_docker_client()
        try:
            self._container = await asyncio.to_thread(
                client.containers.run,
                self.image,
                command="tail -f /dev/null",
                detach=True,
                network_mode="none",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges=true"],
                mem_limit="512m",
                pids_limit=128,
                working_dir=self.workspace_root,
            )
            await asyncio.to_thread(
                self._container.exec_run,
                f"mkdir -p {self.workspace_root}/{META_DIR} {self.workspace_root}/{SHARED_DIR}",
            )
        except Exception as exc:
            raise RuntimeError(f"Docker 컨테이너 생성 실패: {exc}") from exc

    async def stop(self) -> None:
        """컨테이너를 중지하고 제거합니다."""
        container = self._container
        self._container = None
        self._backend = None
        if container is None:
            return

        try:
            await asyncio.to_thread(container.stop)
        except Exception:
            pass

        try:
            await asyncio.to_thread(container.remove)
        except Exception:
            pass

    def get_backend(self) -> DockerSandboxBackend:
        """현재 세션용 백엔드를 반환합니다."""
        if self._container is None:
            raise RuntimeError("DockerSandboxSession이 시작되지 않았습니다")
        if self._backend is None:
            self._backend = DockerSandboxBackend(
                container_id=self._container.id,
                workspace_root=self.workspace_root,
                docker_client=self._get_docker_client(),
            )
        return self._backend

    async def __aenter__(self) -> DockerSandboxSession:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()
