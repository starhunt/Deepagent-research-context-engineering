"""워크스페이스 경로 및 파일 통신 프로토콜 유틸리티."""

from __future__ import annotations

import posixpath

WORKSPACE_ROOT = "/workspace"
META_DIR = "_meta"
SHARED_DIR = "shared"


def _sanitize_segment(segment: str) -> str:
    return segment.strip().strip("/")


def get_subagent_dir(subagent_type: str) -> str:
    """SubAgent별 전용 작업 디렉토리를 반환합니다."""
    safe_segment = _sanitize_segment(subagent_type)
    return posixpath.join(WORKSPACE_ROOT, safe_segment)


def get_result_path(subagent_type: str) -> str:
    """SubAgent 결과 파일 경로를 반환합니다."""
    safe_segment = _sanitize_segment(subagent_type)
    return posixpath.join(WORKSPACE_ROOT, META_DIR, safe_segment, "result.json")
