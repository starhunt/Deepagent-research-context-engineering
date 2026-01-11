"""Context Engineering 연구용 메인 에이전트.

5가지 Context Engineering 전략을 명시적으로 통합한 에이전트입니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from context_engineering_research_agent.context_strategies import (
    ContextCachingStrategy,
    ContextOffloadingStrategy,
    ContextReductionStrategy,
    PromptCachingTelemetryMiddleware,
    ProviderType,
    detect_provider,
)

CONTEXT_ENGINEERING_SYSTEM_PROMPT = """# Context Engineering 연구 에이전트

당신은 Context Engineering 전략을 연구하고 실험하는 에이전트입니다.

## Context Engineering 5가지 핵심 전략

### 1. Context Offloading (컨텍스트 오프로딩)
대용량 도구 결과를 파일시스템으로 축출합니다.
- 도구 결과가 20,000 토큰 초과 시 자동 축출
- /large_tool_results/ 경로에 저장
- read_file로 필요할 때 로드

### 2. Context Reduction (컨텍스트 축소)
컨텍스트 윈도우 사용량이 임계값 초과 시 압축합니다.
- Compaction: 오래된 도구 호출/결과 제거
- Summarization: LLM이 대화 요약 (85% 초과 시)

### 3. Context Retrieval (컨텍스트 검색)
필요한 정보만 선택적으로 로드합니다.
- grep: 텍스트 패턴 검색
- glob: 파일명 패턴 매칭
- read_file: 부분 읽기 (offset/limit)

### 4. Context Isolation (컨텍스트 격리)
SubAgent를 통해 독립된 컨텍스트에서 작업합니다.
- task() 도구로 작업 위임
- 메인 컨텍스트 오염 방지
- 복잡한 작업의 격리 처리

### 5. Context Caching (컨텍스트 캐싱)
시스템 프롬프트와 반복 컨텍스트를 캐싱합니다.
- Anthropic Prompt Caching 활용
- API 비용 절감
- 응답 속도 향상

## 연구 워크플로우

1. 연구 요청을 분석하고 TODO 목록 작성
2. 필요시 SubAgent에게 작업 위임
3. 중간 결과를 파일시스템에 저장
4. 최종 보고서 작성 (/final_report.md)

## 중요 원칙

- 대용량 결과는 파일로 저장하고 참조
- 복잡한 작업은 SubAgent에게 위임
- 진행 상황을 TODO로 추적
- 인용 형식: [1], [2], [3]
"""

current_date = datetime.now().strftime("%Y-%m-%d")

BASE_DIR = Path(__file__).resolve().parent.parent
RESEARCH_WORKSPACE_DIR = BASE_DIR / "research_workspace"

_cached_agent = None
_cached_model = None


def _infer_openrouter_model_name(model: BaseChatModel) -> str | None:
    """OpenRouter 모델에서 모델명을 추출합니다.

    Args:
        model: LangChain 모델 인스턴스

    Returns:
        OpenRouter 모델명 (예: "anthropic/claude-3-sonnet") 또는 None
    """
    if detect_provider(model) != ProviderType.OPENROUTER:
        return None
    for attr in ("model_name", "model", "model_id"):
        name = getattr(model, attr, None)
        if isinstance(name, str) and name.strip():
            return name
    return None


def _get_fs_backend() -> FilesystemBackend:
    return FilesystemBackend(
        root_dir=RESEARCH_WORKSPACE_DIR,
        virtual_mode=True,
        max_file_size_mb=20,
    )


def _get_backend_factory():
    fs_backend = _get_fs_backend()

    def backend_factory(rt: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(rt),
            routes={"/": fs_backend},
        )

    return backend_factory


def get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = ChatOpenAI(model="gpt-4.1", temperature=0.0)
    return _cached_model


def get_agent():
    global _cached_agent
    if _cached_agent is None:
        model = get_model()
        backend_factory = _get_backend_factory()

        offloading_strategy = ContextOffloadingStrategy(backend_factory=backend_factory)
        reduction_strategy = ContextReductionStrategy(summarization_model=model)
        openrouter_model_name = _infer_openrouter_model_name(model)
        caching_strategy = ContextCachingStrategy(
            model=model,
            openrouter_model_name=openrouter_model_name,
        )
        telemetry_middleware = PromptCachingTelemetryMiddleware()

        _cached_agent = create_deep_agent(
            model=model,
            system_prompt=CONTEXT_ENGINEERING_SYSTEM_PROMPT,
            backend=backend_factory,
            middleware=[
                offloading_strategy,
                reduction_strategy,
                caching_strategy,
                telemetry_middleware,
            ],
        )
    return _cached_agent


def create_context_aware_agent(
    model: BaseChatModel | str = "gpt-4.1",
    workspace_dir: Path | str | None = None,
    enable_offloading: bool = True,
    enable_reduction: bool = True,
    enable_caching: bool = True,
    enable_cache_telemetry: bool = True,
    offloading_token_limit: int = 20000,
    reduction_threshold: float = 0.85,
    openrouter_model_name: str | None = None,
) -> Any:
    """Context Engineering 전략이 적용된 에이전트를 생성합니다.

    Multi-Provider 지원: Anthropic, OpenAI, Gemini, OpenRouter 모델 사용 가능.
    Provider는 자동 감지되며, Anthropic만 cache_control 마커가 적용됩니다.

    Args:
        model: LLM 모델 객체 또는 모델명 (기본: gpt-4.1)
        workspace_dir: 작업 디렉토리
        enable_offloading: Context Offloading 활성화
        enable_reduction: Context Reduction 활성화
        enable_caching: Context Caching 활성화
        enable_cache_telemetry: Cache 텔레메트리 수집 활성화
        offloading_token_limit: Offloading 토큰 임계값
        reduction_threshold: Reduction 트리거 임계값
        openrouter_model_name: OpenRouter 모델명 강제 지정

    Returns:
        구성된 DeepAgent
    """
    from context_engineering_research_agent.context_strategies.offloading import (
        OffloadingConfig,
    )
    from context_engineering_research_agent.context_strategies.reduction import (
        ReductionConfig,
    )

    if isinstance(model, str):
        llm: BaseChatModel = ChatOpenAI(model=model, temperature=0.0)
    else:
        llm = model

    workspace = Path(workspace_dir) if workspace_dir else RESEARCH_WORKSPACE_DIR
    workspace.mkdir(parents=True, exist_ok=True)

    local_fs_backend = FilesystemBackend(
        root_dir=workspace,
        virtual_mode=True,
        max_file_size_mb=20,
    )

    def local_backend_factory(rt: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(rt),
            routes={"/": local_fs_backend},
        )

    middlewares = []

    if enable_offloading:
        offload_config = OffloadingConfig(
            token_limit_before_evict=offloading_token_limit
        )
        middlewares.append(
            ContextOffloadingStrategy(
                config=offload_config, backend_factory=local_backend_factory
            )
        )

    if enable_reduction:
        reduce_config = ReductionConfig(context_threshold=reduction_threshold)
        middlewares.append(
            ContextReductionStrategy(config=reduce_config, summarization_model=llm)
        )

    if enable_caching:
        inferred_openrouter_model_name = (
            openrouter_model_name
            if openrouter_model_name is not None
            else _infer_openrouter_model_name(llm)
        )
        middlewares.append(
            ContextCachingStrategy(
                model=llm,
                openrouter_model_name=inferred_openrouter_model_name,
            )
        )

    if enable_cache_telemetry:
        middlewares.append(PromptCachingTelemetryMiddleware())

    return create_deep_agent(
        model=llm,
        system_prompt=CONTEXT_ENGINEERING_SYSTEM_PROMPT,
        backend=local_backend_factory,
        middleware=middlewares,
    )
