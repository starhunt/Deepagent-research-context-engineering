"""자율적 연구 에이전트."""

from datetime import datetime
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

AUTONOMOUS_RESEARCHER_INSTRUCTIONS = """당신은 자율적 연구 에이전트입니다. 
"넓게 탐색 → 깊게 파기" 방법론을 따라 주제를 철저히 연구합니다.

오늘 날짜: {date}

## 연구 워크플로우

### Phase 1: 탐색적 검색 (1-2회)
- 넓은 검색으로 분야 전체 파악
- 핵심 개념, 주요 플레이어, 최근 트렌드 확인
- think_tool로 유망한 방향 2-3개 식별

### Phase 2: 심층 연구 (방향당 1-2회)
- 식별된 방향별 집중 검색
- 각 검색 후 think_tool로 평가
- 가치 있는 정보 획득 여부 판단

### Phase 3: 종합
- 모든 발견 사항 검토
- 패턴과 연결점 식별
- 출처 일치/불일치 기록

## 도구 제한
- 탐색: 최대 2회 검색
- 심층: 최대 3-4회 검색
- **총합: 5-6회**

## 종료 조건
- 포괄적 답변 가능
- 최근 2회 검색이 유사 정보 반환
- 최대 검색 횟수 도달

## 응답 형식

```markdown
## 핵심 발견

### 발견 1: [제목]
[인용 포함 상세 설명 [1], [2]]

### 발견 2: [제목]
[인용 포함 상세 설명]

## 출처 일치 분석
- **높은 일치**: [출처들이 동의하는 주제]
- **불일치/불확실**: [충돌 정보]

## 출처
[1] 출처 제목: URL
[2] 출처 제목: URL
```
"""


def create_researcher_agent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> CompiledStateGraph:
    if model is None:
        model = ChatOpenAI(model="gpt-4.1", temperature=0.0)

    current_date = datetime.now().strftime("%Y-%m-%d")
    formatted_prompt = AUTONOMOUS_RESEARCHER_INSTRUCTIONS.format(date=current_date)

    return create_deep_agent(
        model=model,
        system_prompt=formatted_prompt,
        backend=backend,
    )


def get_researcher_subagent(
    model: str | BaseChatModel | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
) -> dict[str, Any]:
    researcher = create_researcher_agent(model=model, backend=backend)

    return {
        "name": "researcher",
        "description": (
            "자율적 심층 연구 에이전트. '넓게 탐색 → 깊게 파기' 방법론 사용. "
            "복잡한 주제 연구, 다각적 질문, 트렌드 분석에 적합."
        ),
        "runnable": researcher,
    }
