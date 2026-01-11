"""SKILL.md 파일에서 에이전트 스킬을 파싱하고 로드하는 스킬 로더.

YAML 프론트매터 파싱을 통해 Anthropic Agent Skills 패턴을 구현합니다.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml

logger = logging.getLogger(__name__)

MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024  # 10MB - DoS 방지
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """Agent Skills 명세를 따르는 스킬 메타데이터."""

    name: str
    description: str
    path: str
    source: str
    license: NotRequired[str | None]
    compatibility: NotRequired[str | None]
    metadata: NotRequired[dict[str, str] | None]
    allowed_tools: NotRequired[str | None]


def _is_safe_path(path: Path, base_dir: Path) -> bool:
    """경로가 base_dir 내에 안전하게 포함되어 있는지 확인합니다."""
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        return False
    except (OSError, RuntimeError):
        return False


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Agent Skills 명세에 따라 스킬 이름을 검증합니다."""
    if not name:
        return False, "이름은 필수입니다"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "이름이 64자를 초과합니다"
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "이름은 소문자 영숫자와 단일 하이픈만 사용해야 합니다"
    if name != directory_name:
        return (
            False,
            f"이름 '{name}'은 디렉토리 이름 '{directory_name}'과 일치해야 합니다",
        )
    return True, ""


def _parse_skill_metadata(skill_md_path: Path, source: str) -> SkillMetadata | None:
    """SKILL.md 파일에서 YAML 프론트매터를 파싱합니다."""
    try:
        file_size = skill_md_path.stat().st_size
        if file_size > MAX_SKILL_FILE_SIZE:
            logger.warning(
                "%s 건너뜀: 파일이 너무 큼 (%d 바이트)", skill_md_path, file_size
            )
            return None

        content = skill_md_path.read_text(encoding="utf-8")

        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if not match:
            logger.warning(
                "%s 건너뜀: 유효한 YAML 프론트매터를 찾을 수 없음", skill_md_path
            )
            return None

        frontmatter_str = match.group(1)

        try:
            frontmatter_data = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.warning("%s의 YAML이 유효하지 않음: %s", skill_md_path, e)
            return None

        if not isinstance(frontmatter_data, dict):
            logger.warning("%s 건너뜀: 프론트매터가 매핑이 아님", skill_md_path)
            return None

        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")

        if not name or not description:
            logger.warning(
                "%s 건너뜀: 필수 'name' 또는 'description' 누락", skill_md_path
            )
            return None

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
    """단일 스킬 디렉토리에서 모든 스킬을 나열합니다."""
    skills_dir = skills_dir.expanduser()
    if not skills_dir.exists():
        return []

    try:
        resolved_base = skills_dir.resolve()
    except (OSError, RuntimeError):
        return []

    skills: list[SkillMetadata] = []

    for skill_dir in skills_dir.iterdir():
        if not _is_safe_path(skill_dir, resolved_base):
            continue

        if not skill_dir.is_dir():
            continue

        skill_md_path = skill_dir / "SKILL.md"
        if not skill_md_path.exists():
            continue

        if not _is_safe_path(skill_md_path, resolved_base):
            continue

        metadata = _parse_skill_metadata(skill_md_path, source=source)
        if metadata:
            skills.append(metadata)

    return skills


def list_skills(
    *,
    user_skills_dir: Path | None = None,
    project_skills_dir: Path | None = None,
) -> list[SkillMetadata]:
    """사용자 및/또는 프로젝트 디렉토리에서 스킬을 나열합니다.

    프로젝트 스킬이 같은 이름의 사용자 스킬을 오버라이드합니다.
    """
    all_skills: dict[str, SkillMetadata] = {}

    if user_skills_dir:
        user_skills = _list_skills_from_dir(user_skills_dir, source="user")
        for skill in user_skills:
            all_skills[skill["name"]] = skill

    if project_skills_dir:
        project_skills = _list_skills_from_dir(project_skills_dir, source="project")
        for skill in project_skills:
            all_skills[skill["name"]] = skill

    return list(all_skills.values())
