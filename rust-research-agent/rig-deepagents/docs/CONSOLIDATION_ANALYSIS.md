# rig-deepagents 중복 기능 통합 분석

> **Purpose**: Rig 프레임워크와 중복된 기능을 분석하고 통합/제거 권고안 제시
>
> **Created**: 2026-01-03
> **Updated**: 2026-01-03 (통합 완료)

---

## Executive Summary

| 항목 | 상태 | 결과 | 비고 |
|------|------|------|------|
| **LLM Providers** | ✅ 완료 | RigAgentAdapter로 대체 | OpenAIProvider, AnthropicProvider 제거됨 |
| **ThinkTool** | ✅ 유지 | ToolRuntime 접근 필요 | Rig ThinkTool과 차별화됨 |
| **Tool Trait** | ✅ 완료 | RigToolAdapter 추가 | Rig Tool → rig-deepagents Tool 변환 |

---

## 1. LLM Providers ✅ 완료

### 이전 문제점 (해결됨)

레거시 `OpenAIProvider`와 `AnthropicProvider`는:
- ❌ Tool calling 미지원 (파라미터로 받지만 무시됨!)
- ❌ Multi-turn 미지원
- ❌ Streaming 미지원 (fallback only)
- ❌ PromptHook 미지원

### 해결: RigAgentAdapter

레거시 프로바이더를 제거하고 `RigAgentAdapter`로 대체했습니다:

```
src/llm/                     # 정리 후 구조
├── provider.rs              # LLMProvider trait (유지)
├── config.rs                # LLMConfig (유지)
└── message.rs               # Message 변환 유틸리티 (유지)

src/compat/                  # 새로 추가
├── mod.rs                   # 모듈 re-exports
├── rig_agent_adapter.rs     # RigAgentAdapter 구현
└── rig_tool_adapter.rs      # RigToolAdapter 구현
```

### 사용법

```rust
use rig::client::{CompletionClient, ProviderClient};
use rig_deepagents::{RigAgentAdapter, LLMProvider};

// Rig Agent 생성
let client = rig::providers::openai::Client::from_env();
let agent = client.agent("gpt-4").build();

// RigAgentAdapter로 래핑하여 LLMProvider로 사용
let provider: Arc<dyn LLMProvider> = Arc::new(RigAgentAdapter::new(agent));

// AgentExecutor, CompiledWorkflow 등에서 사용
let executor = AgentExecutor::new(provider, middleware, backend);
```

### 결과

- ✅ Rig의 네이티브 프로바이더 활용
- ✅ 20+ LLM 프로바이더 지원 (OpenAI, Anthropic, Gemini, Ollama 등)
- ✅ LLMProvider trait 하위 호환성 유지
- ✅ ~500 lines 코드 제거

---

## 2. ThinkTool 분석 ✅

### 비교

| 항목 | Rig ThinkTool | rig-deepagents ThinkTool |
|------|---------------|--------------------------|
| **위치** | `rig::tools::think::ThinkTool` | `src/tools/think.rs` |
| **인자** | `thought: String` | `reflection: String` |
| **출력** | 입력 그대로 echo | `[Reflection recorded: N chars]` |
| **ToolRuntime** | ❌ 없음 | ✅ 접근 가능 |
| **스키마** | 기본 | `minLength`, `additionalProperties: false` |

### 차이점 분석

**1. 출력 방식**

```rust
// Rig: 전체 thought를 echo
Ok(args.thought)  // "I need to analyze this..." 전체 반환

// rig-deepagents: 요약만 반환 (prompt pollution 방지)
Ok(format!("[Reflection recorded: {} chars]", args.reflection.len()))
```

**이점**: Agent가 긴 reflection을 쓸 때 컨텍스트 절약

**2. ToolRuntime 접근**

```rust
// rig-deepagents: ToolRuntime으로 상태 접근 가능
async fn execute(&self, args: Value, runtime: &ToolRuntime) -> Result<String, ...> {
    if let Some(tool_call_id) = runtime.tool_call_id() {
        debug!(tool_call_id, "Think tool executed");
    }
    // ...
}
```

**이점**: 트레이싱, 디버깅 정보 추가 가능

**3. 스키마 강화**

```rust
// rig-deepagents: 더 엄격한 스키마
"minLength": 1,
"additionalProperties": false
```

### 🎯 권고: 유지

**이유**:
1. 출력 방식이 의도적으로 다름 (prompt pollution 방지)
2. ToolRuntime 접근 필요
3. 스키마 강화로 LLM 오류 감소

**대안**: 필요시 Rig ThinkTool 어댑터 제공
```rust
/// Rig ThinkTool을 rig-deepagents Tool로 래핑 (필요시)
pub struct RigThinkToolAdapter(rig::tools::think::ThinkTool);

impl Tool for RigThinkToolAdapter {
    fn definition(&self) -> ToolDefinition { ... }
    async fn execute(&self, args: Value, _runtime: &ToolRuntime) -> ... {
        // Rig ThinkTool 호출 후 출력 변환
    }
}
```

---

## 3. Tool Trait 분석 ✅

### 설계 비교

**Rig Tool** (정적 타입):
```rust
pub trait Tool: Sized + Send + Sync {
    const NAME: &'static str;           // 컴파일 타임 상수
    type Error: std::error::Error;      // 구체적 에러 타입
    type Args: Deserialize;             // 컴파일 타임 인자 타입
    type Output: Serialize;             // 컴파일 타임 출력 타입

    fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;
}
```

**rig-deepagents Tool** (동적 타입):
```rust
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn execute(
        &self,
        args: serde_json::Value,     // 런타임 동적 타입
        runtime: &ToolRuntime,       // 상태/백엔드 접근
    ) -> Result<String, MiddlewareError>;
}
```

### 핵심 차이

| 특성 | Rig | rig-deepagents |
|------|-----|----------------|
| 타입 안전성 | 컴파일 타임 | 런타임 |
| 상태 접근 | ❌ | ✅ (ToolRuntime) |
| 성능 | 최적화 가능 | 약간의 오버헤드 |
| 유연성 | 제한적 | 높음 |

### 왜 통합 불가?

**rig-deepagents 도구는 ToolRuntime이 필수**:

```rust
// ReadFileTool: Backend 접근 필요
async fn execute(&self, args: Value, runtime: &ToolRuntime) -> ... {
    let path: String = args["file_path"].as_str()...;
    runtime.backend().read(&path).await  // Backend 필수!
}

// WriteTodosTool: AgentState 접근 필요
async fn execute(&self, args: Value, runtime: &ToolRuntime) -> ... {
    let state = runtime.state();  // State 필수!
    state.update_todos(todos);
}
```

Rig Tool에는 이런 컨텍스트가 없음.

### 🎯 권고: 유지 + 어댑터 패턴

**1. rig-deepagents Tool 유지** (ToolRuntime 필요한 도구용)

**2. Rig Tool 어댑터 추가** (상태 불필요한 도구용)

```rust
/// Rig Tool을 rig-deepagents에서 사용
pub struct RigToolAdapter<T: rig::tool::Tool>(T);

impl<T: rig::tool::Tool> Tool for RigToolAdapter<T> {
    fn definition(&self) -> ToolDefinition {
        // Rig ToolDefinition → rig-deepagents ToolDefinition 변환
        let rig_def = futures::executor::block_on(self.0.definition("".into()));
        ToolDefinition {
            name: rig_def.name,
            description: rig_def.description,
            parameters: rig_def.parameters,
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,  // 무시
    ) -> Result<String, MiddlewareError> {
        let typed_args: T::Args = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        let output = self.0.call(typed_args).await
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        serde_json::to_string(&output)
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))
    }
}
```

**사용 예**:
```rust
use rig::tools::think::ThinkTool as RigThinkTool;

// Rig 도구를 rig-deepagents에서 사용
let think = RigToolAdapter(RigThinkTool);
middleware.add_tool(Arc::new(think));
```

---

## 4. 실행 계획 ✅ 완료

### Phase 1: 문서화 ✅
- [x] RIG_FRAMEWORK_REFERENCE.md
- [x] DECISION_GUIDE.md
- [x] CONSOLIDATION_ANALYSIS.md

### Phase 2: 어댑터 추가 ✅
- [x] `src/compat/mod.rs` - 모듈 루트
- [x] `src/compat/rig_tool_adapter.rs` - Rig Tool 어댑터 (7 tests)
- [x] `src/compat/rig_agent_adapter.rs` - Rig Agent 어댑터 (6 tests)

### Phase 3: 레거시 코드 제거 ✅
- [x] `src/llm/openai.rs` 삭제
- [x] `src/llm/anthropic.rs` 삭제
- [x] `src/llm/mod.rs` 업데이트
- [x] `src/lib.rs` 업데이트
- [x] `src/config.rs` - RigAgentAdapter 사용으로 변경

### Phase 4: 테스트 및 문서 ✅
- [x] `tests/e2e_llm_integration.rs` 업데이트
- [x] 소스 코드 문서 주석 업데이트
- [x] docs/ 문서 업데이트
- [x] 401 라이브러리 테스트 통과
- [x] Clippy 검사 통과

---

## 5. 결론

### 완료된 작업 ✅
- `OpenAIProvider`, `AnthropicProvider` 제거됨
- `RigAgentAdapter`로 대체 완료
- ~500 lines 코드 감소
- 13개 어댑터 테스트 추가

### 유지 권고 ✅
- **ThinkTool**: 차별화된 출력 형식, ToolRuntime 접근
- **Tool trait**: ToolRuntime 접근 필수 (Backend, State)
- **AgentMiddleware**: Rig에 없는 고유 기능
- **Pregel/Checkpointing/Backends**: Rig에 없는 고유 기능

### 향후 개선 가능 영역 🔄
- Streaming 지원 (`LLMProvider::stream` 구현)
- Multi-turn 대화 지원 향상
- PromptHook 통합
