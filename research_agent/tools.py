"""리서치 도구 모듈.

이 모듈은 리서치 에이전트를 위한 검색 및 콘텐츠 처리 유틸리티를 제공하며,
Tavily 를 사용해 URL 을 찾고 전체 웹페이지 콘텐츠를 가져와 마크다운으로 변환한다.
"""

from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """웹페이지 콘텐츠를 가져와 마크다운으로 변환한다.

    Args:
        url: 가져올 URL
        timeout: 요청 타임아웃 (초 단위)

    Returns:
        마크다운 형식의 웹페이지 콘텐츠
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool()
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """주어진 쿼리로 웹을 검색한다.

    Tavily를 사용해 관련 URL을 찾고, 전체 웹페이지 콘텐츠를 마크다운으로 가져와 반환한다.

    Args:
        query: 실행할 검색 쿼리
        max_results: 반환할 최대 결과 수 (기본값: 1)
        topic: 주제 필터 - 'general', 'news', 또는 'finance' (기본값: 'general')

    Returns:
        전체 웹페이지 콘텐츠가 포함된 포맷팅된 검색 결과
    """
    # Tavily 를 사용해 관련 URL 목록을 조회한다
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )

    # 각 URL 에 대해 전체 콘텐츠를 가져온다
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]

        # 웹페이지 콘텐츠를 가져온다
        content = fetch_webpage_content(url)

        result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
        result_texts.append(result_text)

    # 최종 응답 형식으로 정리한다
    response = f"""Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""

    return response


@tool()
def think_tool(reflection: str) -> str:
    """연구 진행 상황과 의사결정을 위한 전략적 성찰 도구.

    각 검색 후 결과를 분석하고 다음 단계를 체계적으로 계획하기 위해 이 도구를 사용한다.
    이는 품질 높은 의사결정을 위해 연구 워크플로우에 의도적인 멈춤을 만든다.

    사용 시점:
    - 검색 결과를 받은 후: 어떤 핵심 정보를 찾았는가?
    - 다음 단계를 결정하기 전: 포괄적으로 답변할 수 있을 만큼 충분한가?
    - 연구 공백을 평가할 때: 아직 누락된 구체적인 정보는 무엇인가?
    - 연구를 마무리하기 전: 지금 완전한 답변을 제공할 수 있는가?

    성찰에 포함해야 할 내용:
    1. 현재 발견의 분석 - 어떤 구체적인 정보를 수집했는가?
    2. 공백 평가 - 어떤 중요한 정보가 아직 누락되어 있는가?
    3. 품질 평가 - 좋은 답변을 위한 충분한 증거/예시가 있는가?
    4. 전략적 결정 - 검색을 계속해야 하는가, 답변을 제공해야 하는가?

    Args:
        reflection: 연구 진행 상황, 발견, 공백, 다음 단계에 대한 상세한 성찰

    Returns:
        의사결정을 위해 성찰이 기록되었다는 확인
    """
    return f"성찰 기록됨: {reflection}"
