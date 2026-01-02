//! Agent executor - 메시지 처리 및 도구 실행 루프
//!
//! Python Reference: deepagents/graph.py

use std::sync::Arc;

use crate::backends::Backend;
use crate::error::DeepAgentError;
use crate::llm::{LLMProvider, LLMConfig};
use crate::middleware::{MiddlewareStack, DynTool};
use crate::runtime::{RuntimeConfig, ToolRuntime};
use crate::state::{AgentState, Message, ToolCall};

/// Agent Executor
///
/// 에이전트 실행 루프를 관리합니다:
/// 1. 미들웨어 before hooks 실행
/// 2. LLM 호출
/// 3. 도구 호출 처리
/// 4. 반복 (도구 호출이 없을 때까지)
/// 5. 미들웨어 after hooks 실행
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::{AgentExecutor, OpenAIProvider, MiddlewareStack};
///
/// let provider = Arc::new(OpenAIProvider::from_env()?);
/// let executor = AgentExecutor::new(provider, MiddlewareStack::new(), backend);
/// let result = executor.run(initial_state).await?;
/// ```
pub struct AgentExecutor {
    llm: Arc<dyn LLMProvider>,
    middleware: MiddlewareStack,
    backend: Arc<dyn Backend>,
    max_iterations: usize,
    config: Option<LLMConfig>,
    /// Additional tools to inject (beyond middleware tools)
    additional_tools: Vec<DynTool>,
    /// System prompt to prepend to messages
    system_prompt: Option<String>,
    /// Current recursion depth (for nested subagent calls)
    recursion_depth: usize,
    /// Maximum recursion depth
    max_recursion: usize,
}

impl AgentExecutor {
    /// Create a new agent executor with the given LLM provider
    pub fn new(
        llm: Arc<dyn LLMProvider>,
        middleware: MiddlewareStack,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self {
            llm,
            middleware,
            backend,
            max_iterations: 50,
            config: None,
            additional_tools: Vec::new(),
            system_prompt: None,
            recursion_depth: 0,
            max_recursion: 100,  // Default matches Python
        }
    }

    /// Set the maximum number of iterations for the agent loop
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set LLM configuration for all calls
    pub fn with_config(mut self, config: LLMConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add additional tools to the executor (beyond middleware tools)
    ///
    /// These tools are merged with middleware-provided tools during execution.
    pub fn with_tools(mut self, tools: Vec<DynTool>) -> Self {
        self.additional_tools = tools;
        self
    }

    /// Set a system prompt to prepend to messages
    ///
    /// This system message is added at the start of every execution.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set recursion depth for nested subagent calls (H2 fix)
    ///
    /// This is propagated to the ToolRuntime so nested `task` calls
    /// see the correct recursion depth.
    pub fn with_recursion_depth(mut self, depth: usize) -> Self {
        self.recursion_depth = depth;
        self
    }

    /// Set maximum recursion depth
    pub fn with_max_recursion(mut self, max: usize) -> Self {
        self.max_recursion = max;
        self
    }

    /// 에이전트 실행
    pub async fn run(&self, initial_state: AgentState) -> Result<AgentState, DeepAgentError> {
        let mut state = initial_state;

        // Prepend system prompt if configured
        if let Some(ref system_prompt) = self.system_prompt {
            // Insert system message at the beginning
            let system_msg = Message::system(system_prompt);
            state.messages.insert(0, system_msg);
        }

        // Create runtime with proper recursion configuration (H2 fix)
        let runtime_config = RuntimeConfig {
            debug: false,
            max_recursion: self.max_recursion,
            current_recursion: self.recursion_depth,
        };
        let runtime = ToolRuntime::new(state.clone(), self.backend.clone())
            .with_config(runtime_config);

        // Before hooks 실행 (미들웨어 스택이 내부적으로 상태 업데이트 적용)
        let _before_updates = self.middleware.before_agent(&mut state, &runtime).await
            .map_err(DeepAgentError::Middleware)?;

        // 도구 수집 (middleware tools + additional tools)
        let mut tools = self.middleware.collect_tools();
        tools.extend(self.additional_tools.iter().cloned());
        let tool_definitions: Vec<_> = tools.iter()
            .map(|t| t.definition())
            .collect();

        // 메인 실행 루프
        for iteration in 0..self.max_iterations {
            tracing::debug!(iteration, "Agent iteration");

            // LLM 호출 (using new LLMProvider interface)
            let llm_response = self.llm.complete(
                &state.messages,
                &tool_definitions,
                self.config.as_ref(),
            ).await?;

            let response = llm_response.message;
            state.add_message(response.clone());

            // 도구 호출이 없으면 종료
            if !response.has_tool_calls() {
                tracing::debug!("No tool calls, finishing");
                break;
            }

            // 도구 호출 처리
            if let Some(tool_calls) = &response.tool_calls {
                for call in tool_calls {
                    let result = self.execute_tool_call(call, &tools, &runtime).await;
                    let tool_message = Message::tool(&result, &call.id);
                    state.add_message(tool_message);
                }
            }
        }

        // After hooks 실행 (미들웨어 스택이 내부적으로 상태 업데이트 적용)
        let _after_updates = self.middleware.after_agent(&mut state, &runtime).await
            .map_err(DeepAgentError::Middleware)?;

        Ok(state)
    }

    /// 도구 호출 실행
    async fn execute_tool_call(
        &self,
        call: &ToolCall,
        tools: &[DynTool],
        runtime: &ToolRuntime,
    ) -> String {
        let tool = tools.iter().find(|t| t.definition().name == call.name);

        match tool {
            Some(t) => {
                match t.execute(call.arguments.clone(), runtime).await {
                    Ok(result) => result,
                    Err(e) => format!("Tool error: {}", e),
                }
            }
            None => format!("Unknown tool: {}", call.name),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::backends::MemoryBackend;
    use crate::llm::LLMResponse;
    use crate::middleware::ToolDefinition;

    /// Mock LLM for testing that implements the new LLMProvider trait
    struct MockLLM {
        responses: Vec<Message>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockLLM {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn simple() -> Self {
            Self::new(vec![Message::assistant("Hello! I'm a mock assistant.")])
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, DeepAgentError> {
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let message = self.responses.get(count).cloned().unwrap_or_else(|| {
                Message::assistant("Default response")
            });
            Ok(LLMResponse::new(message))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    #[tokio::test]
    async fn test_executor_basic() {
        let llm = Arc::new(MockLLM::simple());
        let backend = Arc::new(MemoryBackend::new());
        let middleware = MiddlewareStack::new();

        let executor = AgentExecutor::new(llm, middleware, backend);

        let initial_state = AgentState::with_messages(vec![
            Message::user("Hello!")
        ]);

        let result = executor.run(initial_state).await.unwrap();

        assert!(result.messages.len() >= 2);
        assert!(result.last_assistant_message().is_some());
    }

    #[tokio::test]
    async fn test_executor_with_tool_calls() {
        use crate::state::ToolCall;

        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"file_path": "/test.txt"}),
        };

        let responses = vec![
            Message::assistant_with_tool_calls("", vec![tool_call]),
            Message::assistant("Done reading file."),
        ];

        let llm = Arc::new(MockLLM::new(responses));
        let backend = Arc::new(MemoryBackend::new());

        // Pre-populate backend with test file
        backend.write("/test.txt", "Hello World").await.unwrap();

        let middleware = MiddlewareStack::new();
        let executor = AgentExecutor::new(llm, middleware, backend);

        let initial_state = AgentState::with_messages(vec![
            Message::user("Read the test file")
        ]);

        let result = executor.run(initial_state).await.unwrap();

        // Should have: user, assistant (tool call), tool result, assistant (final)
        assert!(result.messages.len() >= 4);
    }

    #[tokio::test]
    async fn test_executor_max_iterations() {
        // Create LLM that always returns tool calls
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "some_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let responses: Vec<Message> = (0..100)
            .map(|_| Message::assistant_with_tool_calls("", vec![tool_call.clone()]))
            .collect();

        let llm = Arc::new(MockLLM::new(responses));
        let backend = Arc::new(MemoryBackend::new());
        let middleware = MiddlewareStack::new();

        let executor = AgentExecutor::new(llm, middleware, backend)
            .with_max_iterations(5);

        let initial_state = AgentState::with_messages(vec![
            Message::user("Test")
        ]);

        let result = executor.run(initial_state).await.unwrap();

        // Should have stopped after max iterations
        // Each iteration adds: assistant (tool call) + tool result = 2 messages
        // Plus initial user message = 1
        // So max of 5 iterations = 1 + (5 * 2) = 11 messages
        assert!(result.messages.len() <= 11);
    }

    #[tokio::test]
    async fn test_executor_with_config() {
        let llm = Arc::new(MockLLM::simple());
        let backend = Arc::new(MemoryBackend::new());
        let middleware = MiddlewareStack::new();

        let config = LLMConfig::new("test-model")
            .with_temperature(0.5)
            .with_max_tokens(1000);

        let executor = AgentExecutor::new(llm, middleware, backend)
            .with_config(config);

        let initial_state = AgentState::with_messages(vec![
            Message::user("Hello with config!")
        ]);

        let result = executor.run(initial_state).await.unwrap();

        assert!(result.messages.len() >= 2);
    }
}
