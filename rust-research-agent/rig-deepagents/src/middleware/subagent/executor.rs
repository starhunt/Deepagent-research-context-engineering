//! SubAgent executor implementation
//!
//! This module provides the execution infrastructure for SubAgents,
//! including the factory pattern for creating and running subagent executors.
//!
//! # Design
//!
//! The executor uses a factory pattern to enable:
//! - Testability (can mock the factory in tests)
//! - Flexibility (different execution strategies)
//! - Configuration injection (model, middleware, etc.)
//!
//! Python Reference: deepagents/middleware/subagents.py

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::time::timeout;

use crate::backends::Backend;
use crate::error::MiddlewareError;
use crate::executor::AgentExecutor;
use crate::llm::LLMProvider;
use crate::middleware::{AgentMiddleware, MiddlewareStack};
use crate::runtime::ToolRuntime;

use super::spec::{SubAgentKind, SubAgentResult, SubAgentSpec};
use super::state_isolation::IsolatedState;

/// Factory trait for creating and executing SubAgents
///
/// This trait abstracts the SubAgent execution process, allowing for:
/// - Different execution strategies
/// - Mocking in tests
/// - Custom execution environments
///
/// # Example
///
/// ```rust,ignore
/// struct MyExecutorFactory { /* ... */ }
///
/// #[async_trait]
/// impl SubAgentExecutorFactory for MyExecutorFactory {
///     async fn execute(
///         &self,
///         subagent: &SubAgentKind,
///         prompt: &str,
///         state: IsolatedState,
///         runtime: &ToolRuntime,
///     ) -> Result<SubAgentResult, MiddlewareError> {
///         // Custom execution logic
///     }
/// }
/// ```
#[async_trait]
pub trait SubAgentExecutorFactory: Send + Sync {
    /// Execute a subagent with the given context
    ///
    /// # Arguments
    ///
    /// * `subagent` - The subagent specification or compiled agent
    /// * `prompt` - The task description/prompt for the subagent
    /// * `state` - Isolated state from parent (files only)
    /// * `runtime` - Tool runtime with increased recursion depth
    ///
    /// # Returns
    ///
    /// SubAgentResult containing the final message and file updates
    async fn execute(
        &self,
        subagent: &SubAgentKind,
        prompt: &str,
        state: IsolatedState,
        runtime: &ToolRuntime,
    ) -> Result<SubAgentResult, MiddlewareError>;
}

/// Configuration for DefaultSubAgentExecutorFactory
#[derive(Clone)]
pub struct SubAgentExecutorConfig {
    /// Default LLM model for subagents without explicit model
    pub default_model: Arc<dyn LLMProvider>,

    /// Default middleware applied to all subagents
    pub default_middleware: Vec<Arc<dyn AgentMiddleware>>,

    /// Default backend for file operations
    pub backend: Arc<dyn Backend>,

    /// Maximum iterations for subagent execution
    pub max_iterations: usize,
}

impl SubAgentExecutorConfig {
    /// Create a new configuration with required components
    pub fn new(
        default_model: Arc<dyn LLMProvider>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self {
            default_model,
            default_middleware: Vec::new(),
            backend,
            max_iterations: 25,  // Reasonable default for subagents
        }
    }

    /// Add default middleware
    pub fn with_middleware(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.default_middleware.push(middleware);
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
}

/// Default executor factory using AgentExecutor
///
/// This is the standard implementation that uses the existing
/// AgentExecutor infrastructure to run subagents.
pub struct DefaultSubAgentExecutorFactory {
    config: SubAgentExecutorConfig,
}

impl DefaultSubAgentExecutorFactory {
    /// Create a new factory with the given configuration
    pub fn new(config: SubAgentExecutorConfig) -> Self {
        Self { config }
    }

    /// Build middleware stack for a subagent
    fn build_middleware_stack(&self, spec: &SubAgentSpec) -> MiddlewareStack {
        let mut stack = MiddlewareStack::new();

        // Add default middleware first
        for mw in &self.config.default_middleware {
            stack = stack.with_middleware_arc(mw.clone());
        }

        // Add spec-specific middleware (overrides defaults if same name)
        for mw in &spec.middleware {
            stack = stack.with_middleware_arc(mw.clone());
        }

        stack
    }

    /// Execute a spec-based subagent
    async fn execute_spec(
        &self,
        spec: &SubAgentSpec,
        prompt: &str,
        state: IsolatedState,
        runtime: &ToolRuntime,
    ) -> Result<SubAgentResult, MiddlewareError> {
        // Use spec's model or default
        let model = spec.model.clone().unwrap_or_else(|| self.config.default_model.clone());

        // Build middleware stack
        let middleware = self.build_middleware_stack(spec);

        // Create executor
        let mut executor = AgentExecutor::new(model, middleware, self.config.backend.clone());

        // Apply max iterations from spec or config
        if let Some(max_iter) = spec.max_iterations {
            executor = executor.with_max_iterations(max_iter);
        } else {
            executor = executor.with_max_iterations(self.config.max_iterations);
        }

        // Apply system prompt from spec (H1 fix)
        if !spec.system_prompt.is_empty() {
            executor = executor.with_system_prompt(&spec.system_prompt);
        }

        // Apply tools from spec (H1 fix)
        if !spec.tools.is_empty() {
            executor = executor.with_tools(spec.tools.clone());
        }

        // Apply recursion depth from parent runtime (H2 fix)
        // This ensures nested task calls see the correct recursion depth
        executor = executor
            .with_recursion_depth(runtime.config().current_recursion)
            .with_max_recursion(runtime.config().max_recursion);

        // Convert isolated state to AgentState with prompt
        let initial_state = state.to_agent_state(prompt);

        // Execute with timeout support (default 5 minutes if not specified)
        let timeout_duration = spec.timeout.unwrap_or(Duration::from_secs(300));

        let result_state = match timeout(timeout_duration, executor.run(initial_state)).await {
            Ok(result) => result.map_err(|e| MiddlewareError::SubAgentExecution(e.to_string()))?,
            Err(_) => {
                tracing::warn!(
                    subagent = %spec.name,
                    timeout_secs = timeout_duration.as_secs(),
                    "SubAgent execution timed out"
                );
                return Err(MiddlewareError::SubAgentTimeout {
                    subagent_id: spec.name.clone(),
                    duration_secs: timeout_duration.as_secs(),
                });
            }
        };

        // Extract final message
        let final_message = result_state
            .last_assistant_message()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "SubAgent completed without response.".to_string());

        Ok(SubAgentResult {
            final_message,
            files: result_state.files,
            success: true,
        })
    }
}

#[async_trait]
impl SubAgentExecutorFactory for DefaultSubAgentExecutorFactory {
    async fn execute(
        &self,
        subagent: &SubAgentKind,
        prompt: &str,
        state: IsolatedState,
        runtime: &ToolRuntime,
    ) -> Result<SubAgentResult, MiddlewareError> {
        match subagent {
            SubAgentKind::Spec(spec) => {
                self.execute_spec(spec, prompt, state, runtime).await
            }
            SubAgentKind::Compiled(compiled) => {
                // For compiled subagents, use their pre-built executor
                compiled.executor.execute(prompt, state.files).await
            }
        }
    }
}

/// Mock executor factory for testing
///
/// Returns predefined responses without actually running an agent.
#[cfg(test)]
pub struct MockSubAgentExecutorFactory {
    /// Response to return for all executions
    response: String,
    /// Whether execution should succeed
    should_succeed: bool,
}

#[cfg(test)]
impl MockSubAgentExecutorFactory {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            should_succeed: true,
        }
    }

    pub fn failing(error_message: impl Into<String>) -> Self {
        Self {
            response: error_message.into(),
            should_succeed: false,
        }
    }
}

#[cfg(test)]
#[async_trait]
impl SubAgentExecutorFactory for MockSubAgentExecutorFactory {
    async fn execute(
        &self,
        _subagent: &SubAgentKind,
        _prompt: &str,
        _state: IsolatedState,
        _runtime: &ToolRuntime,
    ) -> Result<SubAgentResult, MiddlewareError> {
        if self.should_succeed {
            Ok(SubAgentResult::success(&self.response))
        } else {
            Err(MiddlewareError::SubAgentExecution(self.response.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::llm::LLMResponse;
    use crate::middleware::ToolDefinition;
    use crate::state::{AgentState, Message};
    use crate::llm::LLMConfig;

    /// Mock LLM for testing
    struct MockLLM {
        response: String,
    }

    impl MockLLM {
        fn new(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, crate::error::DeepAgentError> {
            Ok(LLMResponse::new(Message::assistant(&self.response)))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    #[tokio::test]
    async fn test_mock_executor_factory_success() {
        let factory = MockSubAgentExecutorFactory::new("Task completed successfully");

        let spec = SubAgentSpec::new("test", "Test agent");
        let state = IsolatedState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let result = factory
            .execute(&SubAgentKind::Spec(spec), "Do something", state, &runtime)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.final_message, "Task completed successfully");
    }

    #[tokio::test]
    async fn test_mock_executor_factory_failure() {
        let factory = MockSubAgentExecutorFactory::failing("Execution failed");

        let spec = SubAgentSpec::new("test", "Test agent");
        let state = IsolatedState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let result = factory
            .execute(&SubAgentKind::Spec(spec), "Do something", state, &runtime)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_default_executor_factory() {
        let mock_llm = Arc::new(MockLLM::new("Research completed"));
        let backend = Arc::new(MemoryBackend::new());

        let config = SubAgentExecutorConfig::new(mock_llm, backend.clone())
            .with_max_iterations(5);

        let factory = DefaultSubAgentExecutorFactory::new(config);

        let spec = SubAgentSpec::builder("researcher")
            .description("Research agent")
            .system_prompt("You are a researcher")
            .build();

        let state = IsolatedState::new();
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let result = factory
            .execute(&SubAgentKind::Spec(spec), "Research quantum computing", state, &runtime)
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.final_message.contains("Research completed"));
    }

    #[test]
    fn test_executor_config_builder() {
        let mock_llm = Arc::new(MockLLM::new("test"));
        let backend = Arc::new(MemoryBackend::new());

        let config = SubAgentExecutorConfig::new(mock_llm, backend)
            .with_max_iterations(10);

        assert_eq!(config.max_iterations, 10);
    }
}
