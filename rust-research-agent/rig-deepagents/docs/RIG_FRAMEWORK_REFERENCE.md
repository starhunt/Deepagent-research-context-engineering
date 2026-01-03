# Rig Framework Reference (v0.27.0)

> **Purpose**: Rig 프레임워크의 기능을 정확히 파악하여 rig-deepagents에서 중복 개발을 방지합니다.
>
> **Last Updated**: 2026-01-03
> **Rig Version**: 0.27.0
> **Source**: [github.com/0xPlaygrounds/rig](https://github.com/0xPlaygrounds/rig)

---

## 1. Rig Framework Overview

Rig는 LLM 기반 애플리케이션을 위한 Rust 프레임워크로, 인체공학적 설계와 모듈성에 중점을 둡니다.

### 1.1 Core Modules

```
rig-core/src/
├── agent/           # Agent abstraction (high-level LLM wrapper)
├── completion/      # CompletionModel trait, Message types
├── embeddings/      # EmbeddingModel trait, vector operations
├── tool/            # Tool trait, ToolSet, ToolDyn
├── tools/           # Built-in tools (ThinkTool)
├── providers/       # LLM provider implementations (20+)
├── vector_store/    # VectorStoreIndex trait
├── streaming/       # Streaming response support
├── pipeline/        # Pipeline DSL for composition
├── extractor/       # Structured extraction
├── loaders/         # Document loaders
└── integrations/    # External integrations
```

### 1.2 Key Traits

| Trait | Location | Purpose |
|-------|----------|---------|
| `CompletionModel` | `completion/mod.rs` | LLM completion interface |
| `EmbeddingModel` | `embeddings/mod.rs` | Embedding generation |
| `Tool` | `tool/mod.rs` | Tool definition and execution |
| `ToolEmbedding` | `tool/mod.rs` | RAG-able tools |
| `VectorStoreIndex` | `vector_store/mod.rs` | Vector search |
| `PromptHook` | `agent/prompt_request/mod.rs` | Tool call callbacks |

---

## 2. LLM Providers (20+)

Rig는 다음 LLM 프로바이더를 네이티브로 지원합니다:

| Provider | Completion | Embedding | Notes |
|----------|------------|-----------|-------|
| OpenAI | ✅ | ✅ | GPT-4, GPT-4.1, o1 |
| Anthropic | ✅ | ❌ | Claude 3.5, 4 |
| Azure | ✅ | ✅ | Azure OpenAI |
| Cohere | ✅ | ✅ | Command R |
| DeepSeek | ✅ | ❌ | DeepSeek |
| Galadriel | ✅ | ❌ | |
| Gemini | ✅ | ✅ | Gemini Pro |
| Groq | ✅ | ❌ | Fast inference |
| HuggingFace | ✅ | ✅ | Hub models |
| Hyperbolic | ✅ | ❌ | |
| Mira | ✅ | ❌ | |
| Mistral | ✅ | ✅ | Mistral/Mixtral |
| Moonshot | ✅ | ❌ | |
| Ollama | ✅ | ✅ | Local models |
| OpenRouter | ✅ | ❌ | Multi-provider |
| Perplexity | ✅ | ❌ | |
| Together | ✅ | ✅ | |
| Voyage AI | ❌ | ✅ | Embeddings only |
| xAI | ✅ | ❌ | Grok |

### 2.1 Provider Usage

```rust
use rig::providers::openai::Client;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;

// Create provider from environment
let client = Client::from_env();

// Build an agent with tools
let agent = client
    .agent("gpt-4")
    .preamble("You are a helpful assistant.")
    .tool(MyTool)
    .temperature(0.0)
    .build();

// Execute with automatic tool calling
let response = agent.prompt("Hello!").await?;
```

---

## 3. Tool System

### 3.1 Tool Trait (Rig)

Rig의 Tool trait는 **정적 타입**을 사용합니다:

```rust
// rig-core/src/tool/mod.rs

pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// 도구 이름 (컴파일 타임 상수)
    const NAME: &'static str;

    /// 에러 타입 (구체적 타입)
    type Error: std::error::Error + Send + Sync + 'static;

    /// 인자 타입 (Deserialize 가능)
    type Args: for<'a> Deserialize<'a> + Send + Sync;

    /// 출력 타입 (Serialize 가능)
    type Output: Serialize;

    /// 도구 정의 반환
    fn definition(&self, prompt: String) -> impl Future<Output = ToolDefinition>;

    /// 도구 실행
    fn call(&self, args: Self::Args) -> impl Future<Output = Result<Self::Output, Self::Error>>;
}
```

### 3.2 ThinkTool (Rig Built-in)

Rig는 ThinkTool을 기본 제공합니다:

```rust
// rig-core/src/tools/think.rs

use rig::tools::think::ThinkTool;

// 에이전트에 추가
let agent = client
    .agent("gpt-4")
    .tool(ThinkTool)  // Rig 내장 ThinkTool 사용
    .build();
```

**ThinkTool 구현 (Rig)**:
- `const NAME: &'static str = "think"`
- Args: `{ thought: String }`
- Output: echoes back the thought
- 용도: 복잡한 문제에서 추론 단계 기록

### 3.3 ToolSet

```rust
// 도구 컬렉션 관리
let mut toolset = ToolSet::default();
toolset.add_tool(MyTool);
toolset.add_tool(OtherTool);

// 도구 호출
let result = toolset.call("my_tool", args_json).await?;
```

### 3.4 MCP Integration

```rust
// rmcp feature로 MCP 서버 도구 사용
#[cfg(feature = "rmcp")]
use rig::tool::rmcp::McpTool;

let mcp_tool = McpTool::from_mcp_server(definition, server_sink);
toolset.add_tool(mcp_tool);
```

---

## 4. Agent System

### 4.1 Agent Builder

```rust
let agent = client
    .agent("gpt-4")
    .preamble("System prompt here")
    .tool(ThinkTool)
    .tool(SearchTool)
    .dynamic_context(some_context)  // RAG 컨텍스트
    .temperature(0.7)
    .build();
```

### 4.2 PromptRequest (Multi-turn)

```rust
let response = agent
    .prompt("Hello")
    .multi_turn(5)                    // 최대 5턴
    .with_history(&mut chat_history)  // 대화 기록
    .with_tool_concurrency(3)         // 병렬 도구 실행
    .with_hook(my_hook)               // 콜백 훅
    .extended_details()               // 토큰 사용량 포함
    .await?;

println!("Response: {}", response.output);
println!("Tokens: {:?}", response.total_usage);
```

### 4.3 PromptHook (Callbacks)

```rust
pub trait PromptHook<M: CompletionModel>: Clone + Send + Sync {
    /// LLM 호출 전
    async fn on_completion_call(
        &self, prompt: &Message, history: &[Message], cancel: CancelSignal
    ) {}

    /// LLM 응답 수신 후
    async fn on_completion_response(
        &self, prompt: &Message, response: &CompletionResponse, cancel: CancelSignal
    ) {}

    /// 도구 호출 전
    async fn on_tool_call(
        &self, tool_name: &str, tool_call_id: Option<String>, args: &str, cancel: CancelSignal
    ) {}

    /// 도구 실행 후
    async fn on_tool_result(
        &self, tool_name: &str, tool_call_id: Option<String>, args: &str, result: &str, cancel: CancelSignal
    ) {}
}
```

---

## 5. Vector Stores (9+)

| Store | Crate | Notes |
|-------|-------|-------|
| MongoDB | `rig-mongodb` | Atlas Vector Search |
| LanceDB | `rig-lancedb` | Embedded vector DB |
| Neo4j | `rig-neo4j` | Graph + Vector |
| Qdrant | `rig-qdrant` | High-performance |
| SQLite | `rig-sqlite` | Embedded |
| SurrealDB | `rig-surrealdb` | Multi-model |
| Milvus | `rig-milvus` | Distributed |
| ScyllaDB | `rig-scylladb` | Wide-column |
| AWS S3 Vectors | `rig-s3vectors` | Cloud storage |

---

## 6. Streaming

```rust
use rig::streaming::StreamingCompletion;

let stream = agent.stream_prompt("Hello").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{}", text),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

---

## 7. Examples (Official)

Rig는 다양한 공식 예제를 제공합니다:

### Agent Patterns
- `agent.rs` - Basic agent
- `agent_autonomous.rs` - Self-directed agent
- `agent_with_tools.rs` - Tool integration
- `agent_parallelization.rs` - Concurrent execution
- `agent_routing.rs` - Request routing
- `agent_prompt_chaining.rs` - Sequential workflows
- `agent_with_agent_tool.rs` - Agent as tool

### RAG
- `rag.rs` - Basic RAG
- `rag_dynamic_tools.rs` - Dynamic tool selection
- `vector_search.rs` - Vector operations

### Streaming
- `anthropic_streaming.rs`
- `multi_turn_streaming.rs`
- `ollama_streaming_pause_control.rs`

### Specialized
- `gemini_video_understanding.rs` - Video analysis
- `image_generation.rs` - Image synthesis
- `transcription.rs` - Audio transcription

---

## 8. rig-deepagents vs Rig: Duplication Analysis

### 8.1 Feature Comparison

| Feature | Rig | rig-deepagents | Status |
|---------|-----|----------------|--------|
| **LLM Providers** | 20+ providers | OpenAI/Anthropic wrapper | ⚠️ **Use Rig directly** |
| **Tool Trait** | Static types (`const NAME`, generic Args) | Dynamic (`serde_json::Value`) | ✅ Different purposes |
| **ThinkTool** | Built-in | Custom impl | ⚠️ **Use Rig's** |
| **ToolSet** | `HashMap<String, ToolType>` | `ToolRegistry` | Similar - adapter OK |
| **PromptHook** | Callbacks for tool calls | `before_model`/`after_model` | ✅ Complementary |
| **Streaming** | Native | Not implemented | ⚠️ **Use Rig's** |
| **Vector Stores** | 9+ implementations | None | ⚠️ **Use Rig's** |

### 8.2 rig-deepagents Unique Features

These features are **NOT available in Rig** and should be maintained:

| Feature | Purpose | Location |
|---------|---------|----------|
| **AgentMiddleware** | Tool injection, prompt modification | `middleware/traits.rs` |
| **Pregel Runtime** | Graph-based workflow execution | `pregel/` |
| **Checkpointing** | State persistence (SQLite/Redis/Postgres) | `pregel/checkpoint/` |
| **Backend Abstraction** | Filesystem/Memory/Composite | `backends/` |
| **SubAgent System** | Task delegation to child agents | `middleware/subagent.rs` |
| **Human-in-the-Loop** | Interrupt/approval flow | `middleware/human_in_loop.rs` |
| **SummarizationMiddleware** | Token budget management | `middleware/summarization.rs` |
| **SkillsMiddleware** | Progressive skill disclosure | `skills/` |
| **Model Hooks** | `before_model`/`after_model` | `middleware/traits.rs` |
| **PatchToolCalls** | Tool call modification | `middleware/patch_tool_calls.rs` |

### 8.3 Tool Trait Design Differences

**Rig Tool (Static)**:
```rust
impl Tool for MyTool {
    const NAME: &'static str = "my_tool";
    type Error = MyError;
    type Args = MyArgs;      // 컴파일 타임 타입
    type Output = MyOutput;  // 컴파일 타임 타입

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error>;
}
```

**rig-deepagents Tool (Dynamic)**:
```rust
impl Tool for MyTool {
    fn definition(&self) -> ToolDefinition;

    async fn execute(
        &self,
        args: serde_json::Value,  // 런타임 타입
        runtime: &ToolRuntime,    // 상태 접근 가능
    ) -> Result<String, MiddlewareError>;
}
```

**차이점**:
- Rig: 컴파일 타임 타입 안전성, 성능 최적화
- rig-deepagents: 런타임 유연성, `ToolRuntime` 통한 상태 접근

### 8.4 Recommendations

#### ✅ Integration Complete: RigAgentAdapter

레거시 `OpenAIProvider`와 `AnthropicProvider`는 **제거**되었습니다.
대신 `RigAgentAdapter`를 사용하여 Rig의 네이티브 프로바이더를 래핑합니다:

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

**ThinkTool**: rig-deepagents의 `ThinkTool`은 `ToolRuntime` 접근을 위해 유지됩니다.

**Streaming**: Rig의 streaming 기능 사용

#### ✅ Keep rig-deepagents Features

1. **AgentMiddleware**: Rig에 없는 고유 기능
2. **Pregel Runtime**: 워크플로우 그래프 실행
3. **Checkpointing**: 상태 영속성
4. **Backend Abstraction**: 파일시스템 추상화
5. **Tool with Runtime**: `ToolRuntime` 접근이 필요한 도구

#### ✅ Adapter Pattern Implemented

Rig Tool을 rig-deepagents에서 사용하려면 `RigToolAdapter` 사용:

```rust
use rig::tools::think::ThinkTool;
use rig_deepagents::{RigToolAdapter, ToolRegistry};

// Rig 내장 ThinkTool을 rig-deepagents Tool로 변환
let rig_think = RigToolAdapter::new(ThinkTool);

// ToolRegistry에 등록
let mut registry = ToolRegistry::new();
registry.register(Arc::new(rig_think));
```

**RigToolAdapter**는 `src/compat/rig_tool_adapter.rs`에 구현되어 있습니다.

---

## 9. Integration Strategy

### Phase 1: Use Rig Providers Directly

```rust
// In rig-deepagents demo/examples
use rig::providers::openai::Client;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::tools::think::ThinkTool;

let client = Client::from_env();
let agent = client
    .agent("gpt-4")
    .tool(ThinkTool)
    .build();
```

### Phase 2: Hybrid Approach

rig-deepagents의 고유 기능과 Rig의 LLM/Tool 시스템을 함께 사용:

```rust
// Rig agent with rig-deepagents middleware
let rig_agent = client.agent("gpt-4").build();

// rig-deepagents middleware stack
let middleware = MiddlewareStack::new()
    .add(FilesystemMiddleware::new(backend))
    .add(SubAgentMiddleware::new(...))
    .add(HumanInTheLoopMiddleware::new(...));

// Combined execution
let executor = HybridExecutor::new(rig_agent, middleware);
```

---

## 10. Reference Links

- **Rig Docs**: https://docs.rig.rs
- **Rig GitHub**: https://github.com/0xPlaygrounds/rig
- **Rig Examples**: https://github.com/0xPlaygrounds/rig/tree/main/rig-core/examples
- **Anthropic Think Tool**: https://anthropic.com/engineering/claude-think-tool

---

## Appendix A: File Locations

### Rig (rig-core 0.27.0)
```
~/.cargo/registry/src/.../rig-core-0.27.0/src/
├── lib.rs                    # Module exports
├── tool/mod.rs               # Tool trait, ToolSet
├── tools/think.rs            # ThinkTool implementation
├── agent/mod.rs              # Agent builder
├── agent/prompt_request/     # PromptRequest, PromptHook
│   ├── mod.rs
│   └── streaming.rs
├── completion/mod.rs         # CompletionModel trait
└── providers/                # All LLM providers
    ├── openai/
    ├── anthropic/
    └── ...
```

### rig-deepagents
```
rust-research-agent/crates/rig-deepagents/src/
├── lib.rs                    # Module exports
├── middleware/traits.rs      # Tool trait (dynamic), AgentMiddleware
├── tools/                    # Tool implementations
│   ├── think.rs              # Custom ThinkTool (⚠️ duplicate)
│   └── tavily.rs             # Tavily search
├── llm/                      # LLM providers (⚠️ can use Rig's)
│   ├── openai.rs
│   └── anthropic.rs
├── pregel/                   # Unique: Graph runtime
├── backends/                 # Unique: Backend abstraction
└── middleware/               # Unique: Middleware system
```
