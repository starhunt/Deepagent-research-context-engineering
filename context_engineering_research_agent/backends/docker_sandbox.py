"""Docker 샌드박스 백엔드 구현.

BaseSandbox를 상속하여 Docker 컨테이너 내에서 파일 작업 및 코드 실행을 수행합니다.
"""

from __future__ import annotations

import asyncio
import io
import posixpath
import tarfile
import time
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from context_engineering_research_agent.backends.workspace_protocol import (
    WORKSPACE_ROOT,
)


class DockerSandboxBackend(BaseSandbox):
    """Docker 컨테이너 기반 샌드박스 백엔드.

    실행과 파일 작업을 모두 컨테이너 내부에서 수행하여
    SubAgent 간 공유 작업공간을 제공합니다.
    """

    def __init__(
        self,
        container_id: str,
        *,
        workspace_root: str = WORKSPACE_ROOT,
        docker_client: Any | None = None,
    ) -> None:
        self._container_id = container_id
        self._workspace_root = workspace_root
        self._docker_client = docker_client

    @property
    def id(self) -> str:
        """컨테이너 식별자를 반환합니다."""
        return self._container_id

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

    def _get_container(self) -> Any:
        client = self._get_docker_client()
        try:
            return client.containers.get(self._container_id)
        except Exception as exc:
            raise RuntimeError(f"컨테이너 조회 실패: {exc}") from exc

    def _resolve_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return posixpath.join(self._workspace_root, path)

    def _ensure_parent_dir(self, parent_dir: str) -> None:
        if not parent_dir:
            return
        result = self.execute(f"mkdir -p {parent_dir}")
        if result.exit_code not in (0, None):
            raise RuntimeError(result.output or "워크스페이스 디렉토리 생성 실패")

    def _truncate_output(self, output: str) -> tuple[str, bool]:
        if len(output) <= 100000:
            return output, False
        return output[:100000] + "\n[출력이 잘렸습니다...]", True

    def execute(self, command: str) -> ExecuteResponse:
        """컨테이너 내부에서 명령을 실행합니다.

        shell을 통해 실행하므로 리다이렉션(>), 파이프(|), &&, || 등을 사용할 수 있습니다.
        """
        try:
            container = self._get_container()
            exec_result = container.exec_run(
                ["sh", "-c", command],
                workdir=self._workspace_root,
            )
            raw_output = exec_result.output
            if isinstance(raw_output, bytes):
                output = raw_output.decode("utf-8", errors="replace")
            else:
                output = str(raw_output)
            output, truncated = self._truncate_output(output)
            return ExecuteResponse(
                output=output,
                exit_code=exec_result.exit_code,
                truncated=truncated,
            )
        except Exception as exc:
            return ExecuteResponse(output=f"Docker 실행 오류: {exc}", exit_code=1)

    async def aexecute(self, command: str) -> ExecuteResponse:
        """비동기 실행 래퍼."""
        return await asyncio.to_thread(self.execute, command)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """파일을 컨테이너로 업로드합니다."""
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                full_path = self._resolve_path(path)
                parent_dir = posixpath.dirname(full_path)
                file_name = posixpath.basename(full_path)
                if not file_name:
                    raise ValueError("업로드 경로에 파일명이 필요합니다")
                self._ensure_parent_dir(parent_dir)

                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                    info = tarfile.TarInfo(name=file_name)
                    info.size = len(content)
                    info.mtime = time.time()
                    tar.addfile(info, io.BytesIO(content))
                tar_stream.seek(0)

                container = self._get_container()
                container.put_archive(parent_dir or "/", tar_stream.getvalue())
                responses.append(FileUploadResponse(path=path))
            except Exception as exc:
                responses.append(
                    FileUploadResponse(path=path, error=self._map_upload_error(exc))
                )
        return responses

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """비동기 업로드 래퍼."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """컨테이너에서 파일을 다운로드합니다."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                full_path = self._resolve_path(path)
                container = self._get_container()
                stream, _ = container.get_archive(full_path)
                raw = b"".join(chunk for chunk in stream)
                content = self._extract_tar_content(raw)
                if content is None:
                    responses.append(
                        FileDownloadResponse(path=path, error="is_directory")
                    )
                    continue
                responses.append(FileDownloadResponse(path=path, content=content))
            except Exception as exc:
                responses.append(
                    FileDownloadResponse(path=path, error=self._map_download_error(exc))
                )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """비동기 다운로드 래퍼."""
        return await asyncio.to_thread(self.download_files, paths)

    def _extract_tar_content(self, raw: bytes) -> bytes | None:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if not members:
                return None
            target = members[0]
            file_obj = tar.extractfile(target)
            if file_obj is None:
                return None
            return file_obj.read()

    def _map_upload_error(self, exc: Exception) -> FileOperationError:
        message = str(exc).lower()
        if "permission" in message:
            return "permission_denied"
        if "is a directory" in message:
            return "is_directory"
        return "invalid_path"

    def _map_download_error(self, exc: Exception) -> FileOperationError:
        message = str(exc).lower()
        if "permission" in message:
            return "permission_denied"
        if "no such file" in message or "not found" in message:
            return "file_not_found"
        if "is a directory" in message:
            return "is_directory"
        return "invalid_path"
