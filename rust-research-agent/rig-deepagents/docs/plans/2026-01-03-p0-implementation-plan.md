# P0 구현 계획: 프로덕션 대체를 위한 필수 작업

> **목표**: Rust rig-deepagents가 Python LangChain DeepAgent를 프로덕션에서 100% 대체할 수 있도록 핵심 기능 완성
>
> **예상 기간**: 3-4주 (Codex 검토 후 조정)
> **우선순위**: P0 (Critical)
> **Reviewed by**: Codex (gpt-5.2-codex) - 2026-01-03

---

## Codex Review Summary

### 발견된 Critical Issues

| Severity | Issue | Reference |
|----------|-------|-----------|
| 🔴 Critical | `resume()`가 현재 런타임에서 동작 불가 - `run_inner`가 항상 `superstep = 0`부터 시작 | `runtime.rs:167,169` |
| 🟠 High | Checkpoint가 `WorkflowMessage` 하드코딩, generic `M` 지원 불가 | `runtime.rs:52`, `mod.rs:98` |
| 🟠 High | `retry_counts` 미체크포인팅 → resume 시 재시도 의미 변경 | `runtime.rs:56,354` |
| 🟠 High | Rig API 불일치: `stream_prompt`가 `MultiTurnStreamItem` 반환 | rig-core 0.27 API |
| 🟡 Medium | Tool call 파싱이 pure JSON만 처리, 혼합 응답 누락 가능 | `rig_agent_adapter.rs:215` |

---

## Executive Summary (수정됨)

| Task | 현재 상태 | 목표 | 예상 공수 |
|------|----------|------|----------|
| **P0-1: Checkpointing 통합** | 구현되었으나 연결 안됨 | Runtime에 wire + resume API | 2주 |
| **P0-2: RigAgentAdapter 강화** | 기본 기능만 | Full message + chat API | 1.5주 |
| **P0-3: Streaming (Optional)** | Stub 상태 | MultiTurnStreamItem 변환 | 1주 (defer 가능) |

---

## P0-1: Checkpointing을 PregelRuntime에 통합

### 현재 상태 분석

**✅ 이미 구현됨:**
```rust
// src/pregel/checkpoint/mod.rs
pub struct Checkpoint<S> {
    pub workflow_id: String,
    pub superstep: usize,
    pub state: S,
    pub vertex_states: HashMap<VertexId, VertexState>,
    pub pending_messages: HashMap<VertexId, Vec<WorkflowMessage>>,
}

pub trait Checkpointer<S: WorkflowState> {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError>;
    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError>;
    async fn latest(&self) -> Result<Option<Checkpoint<S>>, PregelError>;
}
```

**✅ 이미 구현됨:**
```rust
// src/pregel/config.rs
impl PregelConfig {
    pub fn should_checkpoint(&self, superstep: usize) -> bool {
        self.checkpointing_enabled() && superstep > 0 && superstep % self.checkpoint_interval == 0
    }
}
```

**❌ 누락된 부분:**
- `PregelRuntime`이 `Checkpointer`를 보유하지 않음
- `run()` 루프에서 `should_checkpoint()` 확인 후 저장 로직 없음
- 체크포인트에서 resume하는 기능 없음

### 구현 계획

#### Task 1.1: PregelRuntime에 Checkpointer 추가 (3일)

> ⚠️ **Codex 피드백 반영**: Checkpoint를 `WorkflowMessage` 전용으로 특수화하고,
> `run_from_checkpoint()` API 추가 필요

**파일**: `src/pregel/runtime.rs`

**Option A (권장): WorkflowMessage 전용 impl block**

```rust
// Generic runtime은 변경 없음
pub struct PregelRuntime<S, M> { ... }

// WorkflowMessage 전용 체크포인팅 구현
impl<S> PregelRuntime<S, WorkflowMessage>
where
    S: WorkflowState,
{
    /// Attach a checkpointer for state persistence
    pub fn with_checkpointer(
        mut self,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Self {
        self.checkpointer = Some(checkpointer);
        self.workflow_id = workflow_id.into();
        self
    }

    /// Run workflow from a checkpoint (for resume)
    ///
    /// # Critical Fix (Codex Review)
    /// 기존 run()은 superstep=0부터 시작하므로, 체크포인트에서
    /// 재개하려면 별도의 진입점 필요
    pub async fn run_from_checkpoint(
        &mut self,
        checkpoint: Checkpoint<S>
    ) -> Result<WorkflowResult<S>, PregelError> {
        // Restore state from checkpoint
        self.restore_from_checkpoint(&checkpoint)?;

        // Continue from checkpoint superstep
        self.run_inner_from(checkpoint.state, checkpoint.superstep).await
    }

    /// Resume from the latest checkpoint
    pub async fn resume(&mut self) -> Result<Option<WorkflowResult<S>>, PregelError> {
        if let Some(checkpointer) = &self.checkpointer {
            if let Some(checkpoint) = checkpointer.latest().await? {
                let result = self.run_from_checkpoint(checkpoint).await?;
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    /// Internal: Restore vertex states, message queues, and retry counts
    fn restore_from_checkpoint(&mut self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError> {
        // Validate workflow_id matches
        if checkpoint.workflow_id != self.workflow_id {
            return Err(PregelError::CheckpointMismatch {
                expected: self.workflow_id.clone(),
                found: checkpoint.workflow_id.clone(),
            });
        }

        // Restore vertex states
        self.vertex_states = checkpoint.vertex_states.clone();

        // Restore pending messages
        for (vid, messages) in &checkpoint.pending_messages {
            if let Some(queue) = self.message_queues.get_mut(vid) {
                *queue = messages.clone();
            }
        }

        // NEW (Codex 피드백): Restore retry counts from metadata
        if let Some(retry_json) = checkpoint.metadata.get("retry_counts") {
            if let Ok(counts) = serde_json::from_str(retry_json) {
                self.retry_counts = counts;
            }
        }

        Ok(())
    }

    /// Internal: Create checkpoint including retry_counts
    fn create_checkpoint(&self, superstep: usize, state: &S) -> Checkpoint<S> {
        let mut checkpoint = Checkpoint::new(
            &self.workflow_id,
            superstep,
            state.clone(),
            self.vertex_states.clone(),
            self.message_queues.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );

        // NEW (Codex 피드백): Include retry_counts in metadata
        if let Ok(retry_json) = serde_json::to_string(&self.retry_counts) {
            checkpoint = checkpoint.with_metadata("retry_counts", retry_json);
        }

        checkpoint
    }
}
```

**새로운 run_inner_from() 메서드 (superstep 시작점 지정)**:

```rust
async fn run_inner_from(
    &mut self,
    initial_state: S,
    start_superstep: usize
) -> Result<WorkflowResult<S>, PregelError> {
    let mut state = initial_state;
    let mut superstep = start_superstep;

    loop {
        // Check max supersteps (adjusted for resume)
        if superstep >= self.config.max_supersteps {
            return Err(PregelError::MaxSuperstepsExceeded(superstep));
        }

        // ... existing loop logic ...

        superstep += 1;

        // Checkpoint if interval reached
        if self.config.should_checkpoint(superstep) {
            self.save_checkpoint(superstep, &state).await?;
        }
    }
}
```

#### Task 1.2: run() 루프에 Checkpoint 저장 로직 추가 (1일)

**파일**: `src/pregel/runtime.rs`

```rust
pub async fn run(&mut self, initial_state: S) -> Result<WorkflowResult<S>, PregelError> {
    let mut state = initial_state;
    let mut superstep = 0;

    // ... existing timeout wrapper ...

    loop {
        // Check termination
        if self.is_terminated(&state) {
            break;
        }

        // Execute superstep
        let updates = self.execute_superstep(&state).await?;
        state = state.apply_updates(updates);
        superstep += 1;

        // NEW: Checkpoint if interval reached
        if self.config.should_checkpoint(superstep) {
            self.save_checkpoint(superstep, &state).await?;
        }

        // Check max supersteps
        if superstep >= self.config.max_supersteps {
            break;
        }
    }

    // ... return result ...
}

async fn save_checkpoint(&self, superstep: usize, state: &S) -> Result<(), PregelError> {
    if let Some(checkpointer) = &self.checkpointer {
        let checkpoint = self.create_checkpoint(superstep, state);
        checkpointer.save(&checkpoint).await?;
        tracing::info!(superstep, "Checkpoint saved");
    }
    Ok(())
}
```

#### Task 1.3: CompiledWorkflow에 Checkpointer 전달 (1일)

**파일**: `src/workflow/compiled.rs`

```rust
impl<S: WorkflowState + Serialize> CompiledWorkflow<S> {
    /// Compile with checkpointer for fault tolerance
    pub fn compile_with_checkpointer(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        // ... existing compilation ...
        // Pass checkpointer to runtime
    }

    /// Resume workflow from latest checkpoint
    pub async fn resume(&mut self) -> Result<Option<usize>, PregelError> {
        self.runtime.resume().await
    }
}
```

#### Task 1.4: 테스트 작성 (2일)

**파일**: `tests/integration_checkpointing.rs`

```rust
#[tokio::test]
async fn test_checkpoint_save_during_execution() {
    // Verify checkpoints are created at correct intervals
}

#[tokio::test]
async fn test_resume_from_checkpoint() {
    // Verify workflow can resume from saved state
}

#[tokio::test]
async fn test_checkpoint_with_pending_messages() {
    // Verify messages are preserved across checkpoint/resume
}

#[tokio::test]
async fn test_checkpoint_backend_integration() {
    // Test with File/SQLite backends
}
```

---

## P0-2: RigAgentAdapter 강화

### 현재 상태 분석

**❌ 문제점:**
```rust
// src/compat/rig_agent_adapter.rs:167-186
fn build_prompt_with_tools(messages: &[Message], tools: &[ToolDefinition]) -> String {
    // ...
    // Find the last user message ← 마지막 사용자 메시지만 사용!
    let last_user_msg = messages
        .iter()
        .rfind(|m| m.role == Role::User)
        .map(|m| m.content.clone())
        .unwrap_or_default();
    // ...
}
```

```rust
// src/compat/rig_agent_adapter.rs:122-127
async fn complete(
    &self,
    messages: &[Message],
    tools: &[ToolDefinition],
    _config: Option<&LLMConfig>,  // ← _config 무시됨!
) -> Result<LLMResponse, DeepAgentError> {
```

### 구현 계획

#### Task 2.1: Rig Chat API 통합 (3일)

> ⚠️ **Codex 피드백 반영**: 단순 프롬프트 문자열 대신 Rig의 `Message` 타입을
> 사용하여 `agent.completion(prompt, history)` 호출

**파일**: `src/compat/rig_agent_adapter.rs`

**새로운 접근법: Rig Message 변환**

```rust
use rig::completion::Message as RigMessage;

/// Convert rig-deepagents messages to Rig's native Message format
fn convert_to_rig_messages(messages: &[Message]) -> (RigMessage, Vec<RigMessage>) {
    let mut history = Vec::new();

    for msg in messages.iter().take(messages.len().saturating_sub(1)) {
        let rig_msg = match msg.role {
            Role::System => RigMessage::system(&msg.content),
            Role::User => RigMessage::user(&msg.content),
            Role::Assistant => {
                // Include tool calls in assistant message if present
                if let Some(tool_calls) = &msg.tool_calls {
                    let calls_str = tool_calls.iter()
                        .map(|tc| format!("[{}({})]", tc.name, tc.arguments))
                        .collect::<Vec<_>>()
                        .join(" ");
                    RigMessage::assistant(format!("{}\n{}", msg.content, calls_str))
                } else {
                    RigMessage::assistant(&msg.content)
                }
            },
            Role::Tool => RigMessage::user(format!("[Tool Result]: {}", msg.content)),
        };
        history.push(rig_msg);
    }

    // Last message becomes the prompt
    let prompt = messages.last()
        .map(|m| RigMessage::user(&m.content))
        .unwrap_or_else(|| RigMessage::user(""));

    (prompt, history)
}
```

**수정된 complete() 메서드**:

```rust
#[async_trait]
impl<M> LLMProvider for RigAgentAdapter<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, DeepAgentError> {
        // Convert to Rig message format
        let (prompt, history) = convert_to_rig_messages(messages);

        // Include tool schemas in system context
        let tools_context = if !tools.is_empty() {
            Some(build_tools_section(tools))
        } else {
            None
        };

        // Use Rig's completion API with chat history
        // (Codex 피드백: Agent::completion requires prompt + history)
        let response = self.agent
            .completion(prompt, history)
            .await
            .map_err(|e| DeepAgentError::LlmError(e.to_string()))?
            .send()
            .await
            .map_err(|e| DeepAgentError::LlmError(e.to_string()))?;

        // Parse response with improved tool call extraction
        let message = parse_response_for_tool_calls(&response.output);

        Ok(LLMResponse::new(message))
    }
}
```

**개선된 Tool Call 파싱 (혼합 응답 처리)**:

```rust
/// Parse LLM response for potential tool calls
///
/// Codex 피드백: 혼합 텍스트+JSON 응답에서 tool calls 추출
fn parse_response_for_tool_calls(response: &str) -> Message {
    // 1. Try pure JSON first
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(tool_calls_val) = parsed.get("tool_calls") {
            if let Ok(tool_calls) = extract_tool_calls(tool_calls_val) {
                return Message::assistant_with_tool_calls("", tool_calls);
            }
        }
    }

    // 2. NEW: Try to find JSON in mixed text/JSON response
    if let Some(json_start) = response.find("{\"tool_calls\"") {
        if let Some(json_end) = find_matching_brace(response, json_start) {
            let json_part = &response[json_start..=json_end];
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_part) {
                if let Some(tool_calls_val) = parsed.get("tool_calls") {
                    if let Ok(tool_calls) = extract_tool_calls(tool_calls_val) {
                        let text_part = response[..json_start].trim();
                        return Message::assistant_with_tool_calls(text_part, tool_calls);
                    }
                }
            }
        }
    }

    // 3. Default: treat as normal text response
    Message::assistant(response)
}

/// Find matching closing brace for JSON extraction
fn find_matching_brace(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}
```

#### Task 2.2: LLMConfig 적용 (1일)

**파일**: `src/compat/rig_agent_adapter.rs`

```rust
async fn complete(
    &self,
    messages: &[Message],
    tools: &[ToolDefinition],
    config: Option<&LLMConfig>,
) -> Result<LLMResponse, DeepAgentError> {
    let prompt = build_full_prompt(messages, tools);

    // Apply config if provided
    let response = if let Some(cfg) = config {
        // Use Rig's completion request builder with config
        self.agent
            .completion(&prompt)
            .temperature(cfg.temperature.unwrap_or(0.0) as f32)
            .max_tokens(cfg.max_tokens.unwrap_or(4096) as u32)
            .send()
            .await
    } else {
        self.agent.prompt(&prompt).await
    };

    // ... rest of processing ...
}
```

**참고**: Rig의 `Agent.completion()` 메서드를 사용하여 temperature, max_tokens 등 설정 적용

#### Task 2.3: Streaming 구현 (3일) - Optional/Deferrable

> ⚠️ **Codex 피드백**: Rig의 `stream_prompt`는 `MultiTurnStreamItem` 반환,
> `Result<String, rig::Error>` 아님. 복잡성으로 인해 defer 고려

**파일**: `src/compat/rig_agent_adapter.rs`

**수정된 Streaming 구현 (Rig API 준수)**:

```rust
use rig::agent::prompt_request::streaming::{
    StreamingResult, MultiTurnStreamItem, StreamedAssistantContent
};

async fn stream(
    &self,
    messages: &[Message],
    tools: &[ToolDefinition],
    config: Option<&LLMConfig>,
) -> Result<LLMResponseStream, DeepAgentError> {
    let (prompt, history) = convert_to_rig_messages(messages);

    // Use Rig's streaming API (returns MultiTurnStreamItem stream)
    let rig_stream = self.agent
        .stream_chat(prompt, history)
        .await
        .map_err(|e| DeepAgentError::LlmError(e.to_string()))?;

    // Convert Rig stream to our LLMResponseStream
    Ok(LLMResponseStream::from_rig_stream(rig_stream))
}
```

**파일**: `src/llm/provider.rs` - LLMResponseStream 확장

```rust
use futures::StreamExt;

impl LLMResponseStream {
    /// Create from Rig's native MultiTurnStreamItem stream
    ///
    /// Codex 피드백: MultiTurnStreamItem variants:
    /// - StreamAssistantItem(StreamedAssistantContent<R>)
    /// - StreamUserItem(StreamedUserContent)
    /// - Final(usage)
    pub fn from_rig_stream<R>(
        stream: StreamingResult<R>
    ) -> Self
    where
        R: Send + 'static,
    {
        let converted = stream.filter_map(|item| async {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
                    // Extract text delta from StreamedAssistantContent
                    match content {
                        StreamedAssistantContent::Text(text) => {
                            Some(MessageChunk::Content(text.text))
                        }
                        StreamedAssistantContent::ToolCall(tc) => {
                            Some(MessageChunk::ToolCall {
                                id: tc.id,
                                name: tc.name,
                                arguments_delta: tc.arguments,
                            })
                        }
                        _ => None,
                    }
                }
                Ok(MultiTurnStreamItem::Final(usage)) => {
                    Some(MessageChunk::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                    })
                }
                _ => None,
            }
        });

        LLMResponseStream::new(Box::pin(converted))
    }
}
```

**대안: Streaming 연기 (권장)**

Streaming 복잡성으로 인해 P0에서 제외하고 P1으로 연기:

```rust
async fn stream(...) -> Result<LLMResponseStream, DeepAgentError> {
    // Fallback to complete (existing behavior)
    tracing::warn!("Streaming not yet implemented, falling back to complete");
    let response = self.complete(messages, tools, config).await?;
    Ok(LLMResponseStream::from_complete(response))
}
```

#### Task 2.4: 테스트 강화 (1일)

**파일**: `src/compat/rig_agent_adapter.rs` (test module)

```rust
#[test]
fn test_build_full_prompt_conversation_history() {
    let messages = vec![
        Message::system("You are helpful"),
        Message::user("Hello"),
        Message::assistant("Hi there!"),
        Message::user("What is 2+2?"),
    ];

    let prompt = build_full_prompt(&messages, &[]);

    assert!(prompt.contains("System"));
    assert!(prompt.contains("You are helpful"));
    assert!(prompt.contains("Hello"));
    assert!(prompt.contains("Hi there!"));
    assert!(prompt.contains("What is 2+2?"));
}

#[test]
fn test_build_full_prompt_with_tool_calls() {
    let messages = vec![
        Message::user("Search for Rust"),
        Message::assistant_with_tool_calls("", vec![
            ToolCall { id: "1".into(), name: "search".into(), arguments: json!({"q": "Rust"}) }
        ]),
        Message::tool("1", "Results: ..."),
    ];

    let prompt = build_full_prompt(&messages, &[]);

    assert!(prompt.contains("Tool Call: search"));
    assert!(prompt.contains("Tool Result"));
}
```

---

## 실행 순서 (Codex 피드백 반영)

```
Week 1:
├── Day 1-3: Task 1.1 - PregelRuntime에 Checkpointer 추가
│            (run_from_checkpoint, retry_counts 포함)
├── Day 4: Task 1.2 - run_inner_from() + 저장 로직
└── Day 5: Task 1.3 - CompiledWorkflow 연동

Week 2:
├── Day 1-3: Task 2.1 - Rig Chat API 통합
│            (Message 변환, completion(prompt, history))
├── Day 4: Task 2.2 - LLMConfig 적용
└── Day 5: Task 1.4 - Checkpointing 통합 테스트

Week 3:
├── Day 1-2: Task 2.4 - 어댑터 테스트 강화
├── Day 3: EdgeDriven vs MessageBased resume 테스트
├── Day 4: 버그 수정 및 엣지 케이스 처리
└── Day 5: 문서 업데이트 및 최종 검증

Week 4 (Optional):
├── Day 1-3: Task 2.3 - Streaming 구현 (P1으로 연기 가능)
└── Day 4-5: 최종 E2E 테스트
```

### Scope 조정 옵션

| Option | 범위 | 기간 | 권장 |
|--------|------|------|------|
| **A: Full** | Checkpointing + Adapter + Streaming | 4주 | |
| **B: Core (권장)** | Checkpointing + Adapter (Streaming defer) | 3주 | ✅ |
| **C: Minimal** | Checkpointing only | 2주 | |

---

## 검증 기준

### P0-1 완료 조건
- [ ] `cargo test checkpoint` - 모든 체크포인트 관련 테스트 통과
- [ ] 워크플로우 실행 중 지정된 interval에 체크포인트 저장 확인
- [ ] 체크포인트에서 resume 후 정상 실행 확인
- [ ] File/SQLite 백엔드로 E2E 테스트 통과

### P0-2 완료 조건
- [ ] `cargo test rig_agent_adapter` - 모든 어댑터 테스트 통과
- [ ] 멀티턴 대화에서 전체 히스토리 포함 확인
- [ ] LLMConfig (temperature, max_tokens) 적용 확인
- [ ] Streaming API 정상 동작 확인

---

## 위험 요소 및 완화

| 위험 | 영향 | 완화 방법 |
|------|------|-----------|
| Rig API 변경 | 컴파일 실패 | rig-core 0.27 버전 고정 |
| 체크포인트 크기 | 성능 저하 | 압축 옵션 기본 활성화 |
| Streaming 복잡성 | 일정 지연 | 필요시 fallback 유지 |
| 메시지 변환 오류 | LLM 응답 품질 저하 | 상세 단위 테스트 |

---

## 다음 단계 (P1)

P0 완료 후:
1. **Streaming 완전 구현** (P0에서 연기 시)
2. 추가 LLM 프로바이더 설정 (Gemini, Ollama, etc.)
3. 분산 실행 지원 (Redis 기반 메시지 큐)
4. 성능 벤치마크 및 최적화
5. Context-length 관리 (truncation/summarization)

---

## Appendix: Codex Review 원본

### Issues 발견

1. **[Critical]** `resume()` 동작 불가 - `run_inner`가 항상 `superstep = 0`부터 시작
2. **[High]** Checkpoint가 `WorkflowMessage` 하드코딩
3. **[High]** `retry_counts` 미체크포인팅
4. **[High]** Rig API 불일치 (`MultiTurnStreamItem` vs `Result<String>`)
5. **[Medium]** Tool call 파싱이 pure JSON만 처리

### 권장 개선사항

1. Checkpointing을 `WorkflowMessage` 전용 impl block으로 특수화
2. `run_from_checkpoint(state, superstep)` API 도입
3. `retry_counts`를 메타데이터에 포함
4. Rig Chat API 사용 (`agent.completion(prompt, history)`)
5. Feature flag 기반 백엔드 테스트

### 누락된 고려사항

- Workflow ID 생성 및 검증
- 그래프 구조 변경 시 동작
- LLMConfig 필드 (model/api_base) 처리
- Context-length 관리
- ExecutionMode별 resume 테스트 매트릭스
