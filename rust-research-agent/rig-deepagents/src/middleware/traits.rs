// src/middleware/traits.rs
//! AgentMiddleware 트레이트 정의
//!
//! Python Reference: langchain/agents/middleware/types.py

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::state::{AgentState, Message, Todo, FileData};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::llm::{LLMConfig, TokenUsage};

/// 상태 업데이트 커맨드
/// Python: langgraph.types.Command
#[derive(Debug, Clone)]
pub enum StateUpdate {
    /// 메시지 추가
    AddMessages(Vec<Message>),
    /// 메시지 전체 교체 (SummarizationMiddleware 용)
    SetMessages(Vec<Message>),
    /// Todo 업데이트
    SetTodos(Vec<Todo>),
    /// 파일 업데이트 (None = 삭제)
    UpdateFiles(HashMap<String, Option<FileData>>),
    /// 복합 업데이트
    Batch(Vec<StateUpdate>),
}

impl StateUpdate {
    /// Apply this update to the given AgentState.
    pub fn apply(&self, state: &mut AgentState) {
        match self {
            StateUpdate::AddMessages(msgs) => {
                state.messages.extend(msgs.clone());
            }
            StateUpdate::SetMessages(msgs) => {
                state.messages = msgs.clone();
            }
            StateUpdate::SetTodos(todos) => {
                state.todos = todos.clone();
            }
            StateUpdate::UpdateFiles(files) => {
                for (path, data) in files {
                    if let Some(d) = data {
                        state.files.insert(path.clone(), d.clone());
                    } else {
                        state.files.remove(path);
                    }
                }
            }
            StateUpdate::Batch(updates) => {
                for update in updates {
                    update.apply(state);
                }
            }
        }
    }
}

// ============================================================================
// Model Hook Types (Python: langchain/agents/middleware/types.py)
// ============================================================================

/// Model 호출 요청 컨텍스트
///
/// LLM 호출 전에 미들웨어가 검사하고 수정할 수 있는 요청 정보.
///
/// # Example
///
/// ```rust,ignore
/// let request = ModelRequest::new(messages, tools)
///     .with_config(LLMConfig::default());
/// ```
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// 현재 대화 메시지
    pub messages: Vec<Message>,
    /// 사용 가능한 도구 정의
    pub tools: Vec<ToolDefinition>,
    /// LLM 설정 (온도, max_tokens 등)
    pub config: Option<LLMConfig>,
}

impl ModelRequest {
    /// 새 ModelRequest 생성
    pub fn new(messages: Vec<Message>, tools: Vec<ToolDefinition>) -> Self {
        Self { messages, tools, config: None }
    }

    /// LLM 설정 추가
    pub fn with_config(mut self, config: LLMConfig) -> Self {
        self.config = Some(config);
        self
    }
}

/// Model 호출 응답
///
/// LLM 호출 후 미들웨어가 검사할 수 있는 응답 정보.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    /// LLM이 생성한 메시지
    pub message: Message,
    /// 토큰 사용량 (있는 경우)
    pub usage: Option<TokenUsage>,
}

impl ModelResponse {
    /// 새 ModelResponse 생성
    pub fn new(message: Message) -> Self {
        Self { message, usage: None }
    }

    /// 토큰 사용량 추가
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Model hook 제어 흐름
///
/// 미들웨어가 model 호출 전후에 실행 흐름을 제어할 수 있게 합니다.
#[derive(Debug, Default)]
pub enum ModelControl {
    /// 정상적으로 다음 단계 진행
    #[default]
    Continue,
    /// 요청을 수정하고 계속 진행
    ModifyRequest(ModelRequest),
    /// Model 호출을 건너뛰고 이 응답 사용 (캐싱용)
    Skip(ModelResponse),
    /// 실행을 인터럽트하고 인간 승인 대기 (HumanInTheLoop)
    Interrupt(InterruptRequest),
}

// ============================================================================
// Human-in-the-Loop Types
// ============================================================================

/// 인간 승인을 위한 인터럽트 요청
///
/// HumanInTheLoopMiddleware가 특정 도구 호출에 대해 승인을 요청할 때 사용.
#[derive(Debug, Clone)]
pub struct InterruptRequest {
    /// 승인이 필요한 액션 목록
    pub action_requests: Vec<ActionRequest>,
    /// 각 액션에 대한 리뷰 설정
    pub review_configs: Vec<ReviewConfig>,
}

impl InterruptRequest {
    /// 새 InterruptRequest 생성
    pub fn new(action_requests: Vec<ActionRequest>, review_configs: Vec<ReviewConfig>) -> Self {
        Self { action_requests, review_configs }
    }

    /// 단일 액션으로 InterruptRequest 생성
    pub fn single(action: ActionRequest, config: ReviewConfig) -> Self {
        Self {
            action_requests: vec![action],
            review_configs: vec![config],
        }
    }
}

/// 승인이 필요한 개별 액션
#[derive(Debug, Clone)]
pub struct ActionRequest {
    /// 도구 호출 ID
    pub id: String,
    /// 도구 이름
    pub name: String,
    /// 도구 인자
    pub args: serde_json::Value,
    /// 사용자에게 보여줄 설명 (선택)
    pub description: Option<String>,
}

impl ActionRequest {
    /// 새 ActionRequest 생성
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
            description: None,
        }
    }

    /// 설명 추가
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// 리뷰 설정
#[derive(Debug, Clone)]
pub struct ReviewConfig {
    /// 대상 액션 이름
    pub action_name: String,
    /// 허용되는 결정 유형
    pub allowed_decisions: Vec<Decision>,
}

impl ReviewConfig {
    /// 새 ReviewConfig 생성
    pub fn new(action_name: impl Into<String>, allowed_decisions: Vec<Decision>) -> Self {
        Self {
            action_name: action_name.into(),
            allowed_decisions,
        }
    }

    /// 모든 결정 허용
    pub fn allow_all(action_name: impl Into<String>) -> Self {
        Self::new(action_name, vec![Decision::Approve, Decision::Reject, Decision::Edit])
    }

    /// 승인/거부만 허용
    pub fn approve_reject_only(action_name: impl Into<String>) -> Self {
        Self::new(action_name, vec![Decision::Approve, Decision::Reject])
    }
}

/// 사용자 결정 유형
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Decision {
    /// 액션 승인
    Approve,
    /// 액션 거부
    Reject,
    /// 액션 수정 후 실행
    Edit,
}

/// 도구 정의
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool execution result with optional state updates.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub message: String,
    pub updates: Vec<StateUpdate>,
}

impl ToolResult {
    /// Create a ToolResult with a message and no state updates.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            updates: Vec::new(),
        }
    }

    /// Add a single state update.
    pub fn with_update(mut self, update: StateUpdate) -> Self {
        self.updates.push(update);
        self
    }

    /// Add multiple state updates.
    pub fn with_updates(mut self, updates: Vec<StateUpdate>) -> Self {
        self.updates.extend(updates);
        self
    }
}

/// 도구 인터페이스
#[async_trait]
pub trait Tool: Send + Sync {
    /// 도구 정의 반환
    fn definition(&self) -> ToolDefinition;

    /// 도구 실행
    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError>;
}

/// 동적 도구 타입
pub type DynTool = Arc<dyn Tool>;

/// Tool registry for managing tool implementations
///
/// Maps tool names to their implementations for execution.
///
/// # Example
///
/// ```ignore
/// use rig_deepagents::middleware::{ToolRegistry, DynTool};
/// use rig_deepagents::tools::ThinkTool;
/// use std::sync::Arc;
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(ThinkTool));
///
/// // Look up and execute
/// if let Some(tool) = registry.get("think") {
///     let result = tool.execute(args, &runtime).await?;
///     println!("{}", result.message);
/// }
/// ```
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, DynTool>,
}

impl ToolRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool implementation
    pub fn register(&mut self, tool: DynTool) {
        let name = tool.definition().name;
        self.tools.insert(name, tool);
    }

    /// Register multiple tools at once
    pub fn register_all(&mut self, tools: Vec<DynTool>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&DynTool> {
        self.tools.get(name)
    }

    /// Get all tool definitions (schemas) for LLM
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Get all tool names
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// AgentMiddleware 트레이트
///
/// Python Reference: AgentMiddleware(Generic[StateT, ContextT])
///
/// 핵심 기능:
/// - tools(): 에이전트에 자동 주입할 도구 목록 반환
/// - modify_system_prompt(): 시스템 프롬프트 수정 (체이닝)
/// - before_agent() / after_agent(): 에이전트 라이프사이클 훅
/// - before_model() / after_model(): LLM 호출 전후 훅 (NEW)
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::middleware::{AgentMiddleware, ModelControl, ModelRequest, ModelResponse};
///
/// struct LoggingMiddleware;
///
/// #[async_trait]
/// impl AgentMiddleware for LoggingMiddleware {
///     fn name(&self) -> &str { "logging" }
///
///     async fn before_model(&self, request: &mut ModelRequest, ...) -> Result<ModelControl, _> {
///         tracing::info!("Calling LLM with {} messages", request.messages.len());
///         Ok(ModelControl::Continue)
///     }
///
///     async fn after_model(&self, response: &ModelResponse, ...) -> Result<ModelControl, _> {
///         tracing::info!("LLM responded with: {:?}", response.message.role);
///         Ok(ModelControl::Continue)
///     }
/// }
/// ```
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// 미들웨어 이름
    fn name(&self) -> &str;

    /// 이 미들웨어가 제공하는 도구 목록
    fn tools(&self) -> Vec<DynTool> {
        vec![]
    }

    /// 시스템 프롬프트 수정
    fn modify_system_prompt(&self, prompt: String) -> String {
        prompt
    }

    // =========================================================================
    // Agent Lifecycle Hooks
    // =========================================================================

    /// 에이전트 실행 전 훅 - 전체 에이전트 루프 시작 전에 호출
    ///
    /// Python: before_agent(self, state, runtime) -> dict | None
    async fn before_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }

    /// 에이전트 실행 후 훅 - 전체 에이전트 루프 완료 후에 호출
    ///
    /// Python: after_agent(self, state, runtime) -> dict | None
    async fn after_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }

    // =========================================================================
    // Model Call Hooks (NEW - Python Parity)
    // =========================================================================

    /// LLM 호출 전 훅 - 각 LLM 호출 직전에 실행
    ///
    /// 사용 사례:
    /// - 메시지/도구 수정 (`ModelControl::ModifyRequest`)
    /// - 캐시된 응답 반환 (`ModelControl::Skip`)
    /// - 요청 로깅/모니터링
    ///
    /// # Returns
    ///
    /// - `ModelControl::Continue` - 정상 진행
    /// - `ModelControl::ModifyRequest(req)` - 수정된 요청으로 진행
    /// - `ModelControl::Skip(resp)` - LLM 호출 건너뛰고 이 응답 사용
    /// - `ModelControl::Interrupt(req)` - 실행 인터럽트
    async fn before_model(
        &self,
        _request: &mut ModelRequest,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        Ok(ModelControl::Continue)
    }

    /// LLM 호출 후 훅 - 각 LLM 응답 수신 직후에 실행
    ///
    /// 사용 사례:
    /// - HumanInTheLoop 인터럽트 (`ModelControl::Interrupt`)
    /// - 응답 검사/로깅
    /// - 응답 기반 조건부 처리
    ///
    /// # Returns
    ///
    /// - `ModelControl::Continue` - 정상 진행
    /// - `ModelControl::Interrupt(req)` - 인간 승인 대기
    async fn after_model(
        &self,
        _response: &ModelResponse,
        _state: &AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        Ok(ModelControl::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mock_tool".to_string(),
                description: "A mock tool for testing".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _runtime: &ToolRuntime,
        ) -> Result<ToolResult, MiddlewareError> {
            Ok(ToolResult::new("mock result"))
        }
    }

    struct MockMiddleware;

    #[async_trait]
    impl AgentMiddleware for MockMiddleware {
        fn name(&self) -> &str {
            "mock"
        }

        fn tools(&self) -> Vec<DynTool> {
            vec![Arc::new(MockTool)]
        }

        fn modify_system_prompt(&self, prompt: String) -> String {
            format!("{}\n\nMock middleware addition.", prompt)
        }
    }

    #[test]
    fn test_middleware_tools() {
        let middleware = MockMiddleware;
        let tools = middleware.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "mock_tool");
    }

    #[test]
    fn test_middleware_prompt_modification() {
        let middleware = MockMiddleware;
        let result = middleware.modify_system_prompt("Base prompt".to_string());
        assert!(result.contains("Base prompt"));
        assert!(result.contains("Mock middleware addition"));
    }
}
