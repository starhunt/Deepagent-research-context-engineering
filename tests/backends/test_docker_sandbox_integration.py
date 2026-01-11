"""DockerSandboxBackend/DockerSandboxSession 실환경 통합 테스트.

주의: 이 테스트는 실제 Docker 데몬과 컨테이너 실행이 필요합니다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from langchain_core.messages import ToolMessage

from context_engineering_research_agent.backends.docker_sandbox import (
    DockerSandboxBackend,
)
from context_engineering_research_agent.backends.docker_session import (
    DockerSandboxSession,
)
from context_engineering_research_agent.context_strategies.offloading import (
    ContextOffloadingStrategy,
    OffloadingConfig,
)


def _docker_available() -> bool:
    """Docker 사용 가능 여부를 확인합니다."""
    try:
        import docker

        client = docker.from_env()
        # ping은 Docker 데몬 연결 여부를 가장 빠르게 확인합니다.
        client.ping()
        return True
    except Exception as e:
        print(f"DEBUG: Docker not available: {type(e).__name__}: {e}")
        return False


DOCKER_AVAILABLE = _docker_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DOCKER_AVAILABLE,
        reason="Docker 데몬 또는 python docker SDK를 사용할 수 없습니다.",
    ),
]


@pytest.fixture(scope="module")
def docker_backend() -> Iterator[DockerSandboxBackend]:
    """테스트용 Docker 샌드박스 백엔드를 제공합니다."""
    session = DockerSandboxSession()
    try:
        import asyncio

        asyncio.run(session.start())
        backend = session.get_backend()
        yield backend
    finally:
        import asyncio

        asyncio.run(session.stop())


@pytest.fixture(autouse=True)
def _reset_workspace(docker_backend: DockerSandboxBackend) -> None:
    """각 테스트마다 워크스페이스 내 테스트 디렉토리를 초기화합니다."""
    rm_result = docker_backend.execute("rm -rf test_docker_sandbox")
    assert rm_result.exit_code == 0, f"rm failed: {rm_result.output}"
    mkdir_result = docker_backend.execute("mkdir -p test_docker_sandbox")
    assert mkdir_result.exit_code == 0, f"mkdir failed: {mkdir_result.output}"


# ---------------------------------------------------------------------------
# 1) Code Execution Tests
# ---------------------------------------------------------------------------


def test_execute_basic_commands(docker_backend: DockerSandboxBackend) -> None:
    """기본 명령( echo/ls/pwd )이 컨테이너 내부에서 정상 동작하는지 확인합니다."""
    echo = docker_backend.execute("echo 'hello'")
    assert echo.exit_code == 0
    assert echo.output.strip() == "hello"

    pwd = docker_backend.execute("pwd")
    assert pwd.exit_code == 0
    # DockerSandboxBackend는 workdir=/workspace로 실행합니다.
    assert pwd.output.strip() == "/workspace"

    docker_backend.execute("echo 'x' > test_docker_sandbox/file.txt")
    ls = docker_backend.execute("ls -la test_docker_sandbox")
    assert ls.exit_code == 0
    assert "file.txt" in ls.output


def test_execute_python_and_exit_codes(docker_backend: DockerSandboxBackend) -> None:
    """파이썬 실행 및 exit code(성공/실패)가 정확히 전달되는지 확인합니다."""
    py = docker_backend.execute('python3 -c "print(2 + 2)"')
    assert py.exit_code == 0
    assert py.output.strip() == "4"

    fail = docker_backend.execute('python3 -c "import sys; sys.exit(42)"')
    assert fail.exit_code == 42


def test_execute_truncates_large_output(docker_backend: DockerSandboxBackend) -> None:
    """대용량 출력이 100,000자 기준으로 잘리는지(truncated) 확인합니다."""
    # 110k 이상 출력 생성
    big = docker_backend.execute("python3 -c \"print('x' * 110500)\"")
    assert big.exit_code == 0
    assert big.truncated is True
    assert "[출력이 잘렸습니다" in big.output
    assert len(big.output) <= 100000 + 200  # 안내 문구 포함 여유


def test_execute_timeout_handling_via_alarm(
    docker_backend: DockerSandboxBackend,
) -> None:
    """장시간 작업이 자체 타임아웃(알람)으로 빠르게 종료되는지 확인합니다."""
    start = time.monotonic()
    res = docker_backend.execute(
        'python3 -c "\n'
        "import signal, time\n"
        "def _handler(signum, frame):\n"
        "    raise TimeoutError('alarm')\n"
        "signal.signal(signal.SIGALRM, _handler)\n"
        "signal.alarm(1)\n"
        "time.sleep(10)\n"
        '"'
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"예상보다 오래 걸렸습니다: {elapsed:.2f}s"
    assert res.exit_code != 0
    assert "TimeoutError" in res.output or "alarm" in res.output


# ---------------------------------------------------------------------------
# 2) File Operations Tests
# ---------------------------------------------------------------------------


def test_upload_files_single_and_nested(docker_backend: DockerSandboxBackend) -> None:
    """upload_files가 단일 파일 및 중첩 디렉토리 업로드를 지원하는지 확인합니다."""
    files = [
        ("test_docker_sandbox/one.txt", b"one"),
        ("test_docker_sandbox/nested/dir/two.bin", b"\x00\x01\x02"),
    ]
    responses = docker_backend.upload_files(files)

    assert [r.path for r in responses] == [p for p, _ in files]
    assert all(r.error is None for r in responses)

    # 컨테이너 내 파일 존재/내용 확인
    cat_one = docker_backend.execute("cat test_docker_sandbox/one.txt")
    assert cat_one.exit_code == 0
    assert cat_one.output.strip() == "one"

    # 이진 파일은 다운로드로 검증
    dl = docker_backend.download_files(["test_docker_sandbox/nested/dir/two.bin"])
    assert dl[0].error is None
    assert dl[0].content == b"\x00\x01\x02"


def test_upload_download_multiple_roundtrip(
    docker_backend: DockerSandboxBackend,
) -> None:
    """여러 파일을 업로드한 뒤 다운로드하여 내용이 동일한지 확인합니다."""
    files = [
        ("test_docker_sandbox/a.txt", b"A"),
        ("test_docker_sandbox/b.txt", b"B"),
        ("test_docker_sandbox/sub/c.txt", b"C"),
    ]

    up = docker_backend.upload_files(files)
    assert len(up) == 3
    assert all(r.error is None for r in up)

    paths = [p for p, _ in files]
    dl = docker_backend.download_files(paths)
    assert [r.path for r in dl] == paths
    assert all(r.error is None for r in dl)

    got = {r.path: r.content for r in dl}
    expected = {p: c for p, c in files}
    assert got == expected


def test_download_files_nonexistent_and_directory(
    docker_backend: DockerSandboxBackend,
) -> None:
    """download_files가 없는 파일/디렉토리 대상에서 올바른 에러를 반환하는지 확인합니다."""
    docker_backend.execute("mkdir -p test_docker_sandbox/dir_only")

    responses = docker_backend.download_files(
        [
            "test_docker_sandbox/does_not_exist.txt",
            "test_docker_sandbox/dir_only",
        ]
    )

    assert responses[0].error == "file_not_found"
    assert responses[0].content is None

    assert responses[1].error == "is_directory"
    assert responses[1].content is None


# ---------------------------------------------------------------------------
# 3) Context Offloading Tests (WITHOUT Agent)
# ---------------------------------------------------------------------------


def test_context_offloading_writes_large_tool_result_to_docker_filesystem(
    docker_backend: DockerSandboxBackend,
) -> None:
    """ContextOffloadingStrategy가 대용량 결과를 Docker 파일시스템에 저장하는지 확인합니다."""

    # backend_factory는 runtime을 무시하고 현재 DockerSandboxBackend를 반환합니다.
    strategy = ContextOffloadingStrategy(
        config=OffloadingConfig(token_limit_before_evict=10, chars_per_token=1),
        backend_factory=lambda _runtime: docker_backend,
    )

    content_lines = [f"line_{i:03d}: {'x' * 20}" for i in range(50)]
    large_content = "\n".join(content_lines)

    tool_result = ToolMessage(
        content=large_content,
        tool_call_id="call/with:special@chars!",
    )

    class MinimalRuntime:
        """ToolRuntime 대체용 최소 객체입니다(backend_factory 호출을 위해서만 사용)."""

        state: dict = {}
        config: dict = {}

    processed, offload = strategy.process_tool_result(tool_result, MinimalRuntime())  # type: ignore[arg-type]

    assert offload.was_offloaded is True
    assert offload.file_path is not None
    assert offload.file_path.startswith("/large_tool_results/")

    # 반환 메시지는 원문 전체가 아니라 경로 참조를 포함해야 합니다.
    if hasattr(processed, "content"):
        replacement_text = processed.content  # ToolMessage
    else:
        # Command(update={messages:[ToolMessage...]}) 형태
        update = processed.update  # type: ignore[attr-defined]
        replacement_text = update["messages"][0].content

    assert offload.file_path in replacement_text
    assert "read_file" in replacement_text
    assert len(replacement_text) < len(large_content)

    # 실제 파일이 컨테이너에 저장되었는지 다운로드로 검증합니다.
    downloaded = docker_backend.download_files([offload.file_path])
    assert downloaded[0].error is None
    assert downloaded[0].content is not None
    assert downloaded[0].content.decode("utf-8") == large_content


# ---------------------------------------------------------------------------
# 4) Session Lifecycle Tests
# ---------------------------------------------------------------------------


def test_session_initializes_workspace_dirs() -> None:
    """세션 시작 시 /workspace/_meta 및 /workspace/shared 디렉토리가 생성되는지 확인합니다."""
    session = DockerSandboxSession()
    try:
        import asyncio

        asyncio.run(session.start())
        backend = session.get_backend()

        meta = backend.execute("test -d /workspace/_meta && echo ok")
        shared = backend.execute("test -d /workspace/shared && echo ok")
        assert meta.exit_code == 0
        assert shared.exit_code == 0
        assert meta.output.strip() == "ok"
        assert shared.output.strip() == "ok"
    finally:
        import asyncio

        asyncio.run(session.stop())


def test_multiple_backends_share_same_container_workspace() -> None:
    """동일 컨테이너 ID를 사용하는 여러 백엔드가 파일을 공유하는지 확인합니다."""
    try:
        import docker
    except Exception:
        pytest.skip("python docker SDK가 필요합니다")

    session = DockerSandboxSession()
    try:
        import asyncio

        asyncio.run(session.start())
        backend1 = session.get_backend()
        backend2 = DockerSandboxBackend(
            container_id=backend1.id,
            docker_client=docker.from_env(),
        )

        backend1.execute("mkdir -p test_docker_sandbox")
        backend1.write("/workspace/test_docker_sandbox/shared.txt", "shared")

        read_back = backend2.execute("cat /workspace/test_docker_sandbox/shared.txt")
        assert read_back.exit_code == 0
        assert read_back.output.strip() == "shared"
    finally:
        import asyncio

        asyncio.run(session.stop())


def test_session_stop_removes_container() -> None:
    """세션 종료 시 컨테이너가 중지/삭제되는지 확인합니다."""
    try:
        import docker
    except Exception:
        pytest.skip("python docker SDK가 필요합니다")

    client = docker.from_env()
    session = DockerSandboxSession()

    import asyncio

    asyncio.run(session.start())
    backend = session.get_backend()
    container_id = backend.id

    # 실제로 컨테이너가 존재하는지 확인
    client.containers.get(container_id)

    asyncio.run(session.stop())

    # stop()이 swallow하므로, 실제 삭제 여부는 inspect로 확인
    with pytest.raises(Exception):
        client.containers.get(container_id)


# ---------------------------------------------------------------------------
# 5) Security Verification Tests
# ---------------------------------------------------------------------------


def test_container_security_options_applied(
    docker_backend: DockerSandboxBackend,
) -> None:
    """컨테이너 생성 시 네트워크/권한/메모리 제한 옵션이 적용되는지 확인합니다."""
    try:
        import docker
    except Exception:
        pytest.skip("python docker SDK가 필요합니다")

    client = docker.from_env()
    container = client.containers.get(docker_backend.id)
    container.reload()
    host_cfg = container.attrs.get("HostConfig", {})

    assert host_cfg.get("NetworkMode") == "none"

    cap_drop = host_cfg.get("CapDrop") or []
    assert "ALL" in cap_drop

    security_opt = host_cfg.get("SecurityOpt") or []
    assert any("no-new-privileges" in opt for opt in security_opt)

    # Docker가 바이트 단위로 변환합니다(512m ≈ 536,870,912 bytes)
    memory = host_cfg.get("Memory")
    assert memory is not None
    assert memory >= 512 * 1024 * 1024

    pids_limit = host_cfg.get("PidsLimit")
    assert pids_limit == 128


def test_network_isolation_blocks_outbound(
    docker_backend: DockerSandboxBackend,
) -> None:
    """network_mode='none' 설정으로 외부 네트워크 연결이 차단되는지 확인합니다."""
    res = docker_backend.execute(
        'python3 -c "\n'
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(1.0)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53))\n"
        "    print('UNEXPECTED_CONNECTED')\n"
        "    raise SystemExit(1)\n"
        "except OSError as e:\n"
        "    print('blocked', type(e).__name__)\n"
        "    raise SystemExit(0)\n"
        "finally:\n"
        "    s.close()\n"
        '"'
    )

    assert res.exit_code == 0
    assert "blocked" in res.output
    assert "UNEXPECTED_CONNECTED" not in res.output


# ---------------------------------------------------------------------------
# 6) LLM Output Formatting Tests (코드 실행 결과가 LLM에 전달되는지 검증)
# ---------------------------------------------------------------------------


def _format_execute_result_for_llm(result) -> str:
    """DeepAgents _execute_tool_generator와 동일한 포맷팅 로직.

    FilesystemMiddleware의 execute tool이 LLM에 반환하는 형식을 재현합니다.
    """
    parts = [result.output]

    if result.exit_code is not None:
        status = "succeeded" if result.exit_code == 0 else "failed"
        parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

    if result.truncated:
        parts.append("\n[Output was truncated due to size limits]")

    return "".join(parts)


def test_execute_result_formatted_for_llm_success(
    docker_backend: DockerSandboxBackend,
) -> None:
    """성공한 코드 실행 결과가 LLM이 인지할 수 있는 형태로 포맷팅되는지 확인합니다."""
    result = docker_backend.execute('python3 -c "print(42 * 2)"')

    llm_output = _format_execute_result_for_llm(result)

    assert "84" in llm_output
    assert "[Command succeeded with exit code 0]" in llm_output
    assert "truncated" not in llm_output.lower()


def test_execute_result_formatted_for_llm_failure(
    docker_backend: DockerSandboxBackend,
) -> None:
    """실패한 코드 실행 결과가 LLM이 인지할 수 있는 형태로 포맷팅되는지 확인합니다."""
    result = docker_backend.execute("python3 -c \"raise ValueError('test error')\"")

    llm_output = _format_execute_result_for_llm(result)

    assert "ValueError" in llm_output
    assert "test error" in llm_output
    assert "[Command failed with exit code 1]" in llm_output


def test_execute_result_formatted_for_llm_multiline_output(
    docker_backend: DockerSandboxBackend,
) -> None:
    """여러 줄 출력이 LLM에 그대로 전달되는지 확인합니다."""
    result = docker_backend.execute(
        "python3 -c \"for i in range(5): print(f'line {i}')\""
    )

    llm_output = _format_execute_result_for_llm(result)

    for i in range(5):
        assert f"line {i}" in llm_output
    assert "[Command succeeded with exit code 0]" in llm_output


def test_execute_result_formatted_for_llm_truncation_notice(
    docker_backend: DockerSandboxBackend,
) -> None:
    """대용량 출력이 잘릴 때 LLM에 truncation 알림이 포함되는지 확인합니다."""
    result = docker_backend.execute("python3 -c \"print('x' * 110500)\"")

    llm_output = _format_execute_result_for_llm(result)

    assert result.truncated is True
    assert "[Output was truncated due to size limits]" in llm_output
    assert "[Command succeeded with exit code 0]" in llm_output


def test_execute_result_contains_stderr_for_llm(
    docker_backend: DockerSandboxBackend,
) -> None:
    """stderr 출력이 LLM에 전달되는지 확인합니다."""
    result = docker_backend.execute(
        "python3 -c \"import sys; sys.stderr.write('error message\\n')\""
    )

    llm_output = _format_execute_result_for_llm(result)

    assert "error message" in llm_output
