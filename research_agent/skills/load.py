"""SKILL.md 파일에서 에이전트 스킬을 파싱하고 로드하는 스킬 로더.

이 모듈은 YAML 프론트매터 파싱을 통해 Anthropic Agent Skills 패턴을 구현한다.
각 스킬은 다음을 포함하는 SKILL.md 파일이 있는 디렉토리이다:
- YAML 프론트매터 (name, description 필수)
- 에이전트용 마크다운 지침
- 선택적 지원 파일 (스크립트, 설정 등)

SKILL.md 구조 예시:
```markdown
---
name: web-research
description: 철저한 웹 리서치를 수행하기 위한 구조화된 접근법
---

# 웹 리서치 스킬

## 사용 시점
- 사용자가 주제 연구를 요청할 때
...
```

research_agent 프로젝트용으로 deepagents-cli에서 적응함.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

logger = logging.getLogger(__name__)

# SKILL.md 파일 최대 크기 (10MB) - DoS 방지
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills 명세 제약 조건 (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Agent Skills 명세를 따르는 스킬 메타데이터."""

    name: str
    """스킬 이름 (최대 64자, 소문자 영숫자와 하이픈만)."""

    description: str
    """스킬이 하는 일에 대한 설명 (최대 1024자)."""

    path: str
    """SKILL.md 파일 경로."""

    source: str
    """스킬 출처 ('user' 또는 'project')."""

    # Agent Skills 명세에 따른 선택적 필드
    license: NotRequired[str | None]
    """라이선스 이름 또는 번들된 라이선스 파일 참조."""

    compatibility: NotRequired[str | None]
    """환경 요구사항 (최대 500자)."""

    metadata: NotRequired[dict[str, str] | None]
    """추가 메타데이터용 임의 키-값 매핑."""

    allowed_tools: NotRequired[str | None]
    """사전 승인된 도구의 공백 구분 목록."""


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """경로가 base_dir 내에 안전하게 포함되어 있는지 확인한다.

    심볼릭 링크나 경로 조작을 통한 디렉토리 탐색 공격을 방지한다.
    두 경로 모두 정규 형식으로 변환(심볼릭 링크 따라감)하고
    대상 경로가 기본 디렉토리 내에 있는지 확인한다.

    Args:
        path: 검증할 경로
        base_dir: 경로가 포함되어야 하는 기본 디렉토리

    Returns:
        경로가 base_dir 내에 안전하게 있으면 True, 그렇지 않으면 False
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        # 경로가 base_dir의 하위 디렉토리가 아님
        return False
    except (OSError, RuntimeError):
        # 경로 해석 오류 (예: 순환 심볼릭 링크)
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Agent Skills 명세에 따라 스킬 이름을 검증한다.

    요구사항:
    - 최대 64자
    - 소문자 영숫자와 하이픈만 (a-z, 0-9, -)
    - 하이픈으로 시작하거나 끝날 수 없음
    - 연속 하이픈 없음
    - 부모 디렉토리 이름과 일치해야 함

    Args:
        name: YAML 프론트매터의 스킬 이름
        directory_name: 부모 디렉토리 이름

    Returns:
        (is_valid, error_message) 튜플. 유효하면 에러 메시지는 빈 문자열.
    """
    if not name:
        return False, "이름은 필수입니다"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "이름이 64자를 초과합니다"
    # 패턴: 소문자 영숫자, 세그먼트 사이에 단일 하이픈
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "이름은 소문자 영숫자와 단일 하이픈만 사용해야 합니다"
    if name != directory_name:
        return (
            False,
            f"이름 '{name}'은 디렉토리 이름 '{directory_name}'과 일치해야 합니다",
        )
    return True, ""


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """Agent Skills 명세에 따라 SKILL.md 파일에서 YAML 프론트매터를 파싱한다.

    Args:
        skill_md_path: SKILL.md 파일 경로
        source: 스킬 출처 ('user' 또는 'project')

    Returns:
        모든 필드가 있는 SkillMetadata, 파싱 실패 시 None
    """
    try:
        # 보안: DoS 방지를 위한 파일 크기 확인
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            logger.warning(
                "%s 건너뜀: 파일이 너무 큼 (%d 바이트)", skill_md_path, file_size
            )
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        # --- 구분자 사이의 YAML 프론트매터 매칭
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            logger.warning(
                "%s 건너뜀: 유효한 YAML 프론트매터를 찾을 수 없음", skill_md_path
            )
            return None

        frontmatter_str = match.group(1)

        # 적절한 중첩 구조 지원을 위해 safe_load로 YAML 파싱
        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.warning("%s의 YAML이 유효하지 않음: %s", skill_md_path, e)
            return None

        if not isinstance(frontmatter_data, dict):
            logger.warning("%s 건너뜀: 프론트매터가 매핑이 아님", skill_md_path)
            return None

        # 필수 필드 검증
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            logger.warning(
                "%s 건너뜀: 필수 'name' 또는 'description' 누락", skill_md_path
            )
            return None

        # 명세에 따른 이름 형식 검증 (역호환성을 위해 경고하지만 로드)
        directory_name = skill_md_path.parent.name
        is_valid, error = _validate_skill_name(str(name), directory_name)
        if not is_valid:
            logger.warning(
                "'%s' 스킬 (%s)이 Agent Skills 명세를 따르지 않음: %s. "
                "명세 준수를 위해 이름 변경을 고려하세요.",
                name,
                skill_md_path,
                error,
            )

        # 설명 길이 검증 (명세: 최대 1024자)
        description_str = str(description)
        if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
            logger.warning(
                "%s의 설명이 %d자를 초과하여 잘림",
                skill_md_path,
                MAX_SKILL_DESCRIPTION_LENGTH,
            )
            description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

        return SkillMetadata(
            name=str(name),
            description=description_str,
            path=str(skill_md_path),
            source=source,
            license=frontmatter_data.get("license"),
            compatibility=frontmatter_data.get("compatibility"),
            metadata=frontmatter_data.get("metadata"),
            allowed_tools=frontmatter_data.get("allowed-tools"),
        )

    except (OSError, UnicodeDecodeError) as e:
        logger.warning("%s 읽기 오류: %s", skill_md_path, e)
        return None


def _list_skills_from_dir(skills_dir: Path, source: str) -> list[SkillMetadata]:
    """단일 스킬 디렉토리에서 모든 스킬을 나열한다 (내부 헬퍼).

    스킬 디렉토리를 스캔하여 SKILL.md 파일을 포함하는 하위 디렉토리를 찾고,
    YAML 프론트매터를 파싱하여 스킬 메타데이터를 반환한다.

    스킬 구조:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # 필수: YAML 프론트매터가 있는 지침
    │   ├── script.py       # 선택: 지원 파일
    │   └── config.json     # 선택: 지원 파일

    Args:
        skills_dir: 스킬 디렉토리 경로
        source: 스킬 출처 ('user' 또는 'project')

    Returns:
        name, description, path, source가 있는 스킬 메타데이터 딕셔너리 목록
    """
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    # 보안 검사를 위한 기본 디렉토리 해석
    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        return []

    skills: list[SkillMetadata] = []

    # 하위 디렉토리 순회
    for skill_dir in skills_dir.iterdir():
        # 보안: 스킬 디렉토리 외부를 가리키는 심볼릭 링크 포착
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        # SKILL.md 파일 찾기
        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        # 보안: 읽기 전에 SKILL.md 경로 검증
        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        # 메타데이터 파싱
        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(
    *,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
) -> list[SkillMetadata]:
    """사용자 및/또는 프로젝트 디렉토리에서 스킬을 나열한다.

    두 디렉토리가 모두 제공되면, 사용자 스킬과 동일한 이름의 프로젝트 스킬이
    사용자 스킬을 오버라이드한다.

    Args:
        user_skills_dir: 사용자 레벨 스킬 디렉토리 경로
        project_skills_dir: 프로젝트 레벨 스킬 디렉토리 경로

    Returns:
        두 출처의 스킬 메타데이터를 병합한 목록.
        이름이 충돌할 때 프로젝트 스킬이 우선됨
    """
    all_skills: dict[str, SkillMetadata] = {}

    # 사용자 스킬을 먼저 로드 (기본)
    if user_skills_dir:
        user_skills = _list_skills_from_dir(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    # 프로젝트 스킬을 두 번째로 로드 (오버라이드/확장)
    if project_skills_dir:
        project_skills = _list_skills_from_dir(project_skills_dir, source="project")
        for skill in project_skills:
            # 프로젝트 스킬이 같은 이름의 사용자 스킬을 오버라이드
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
