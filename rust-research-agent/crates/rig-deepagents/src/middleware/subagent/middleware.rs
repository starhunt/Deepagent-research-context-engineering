//! SubAgentMiddleware - Middleware for task delegation
//!
//! This middleware provides the `task` tool to agents, enabling them to
//! delegate work to specialized sub-agents.
//!
//! # Features
//!
//! - Injects the `task` tool into the agent's tool set
//! - Adds system prompt explaining task delegation patterns
//! - Configurable subagent registry and executor factory
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_deepagents::middleware::{SubAgentMiddleware, SubAgentMiddlewareConfig};
//!
//! let config = SubAgentMiddlewareConfig::new(default_model, backend)
//!     .with_subagent(SubAgentKind::Spec(researcher))
//!     .with_subagent(SubAgentKind::Spec(synthesizer));
//!
//! let middleware = SubAgentMiddleware::new(config);
//! let stack = MiddlewareStack::new().with_middleware(middleware);
//! ```
//!
//! Python Reference: deepagents/middleware/subagents.py

use std::sync::Arc;

use async_trait::async_trait;

use crate::backends::Backend;
use crate::llm::LLMProvider;
use crate::middleware::{AgentMiddleware, DynTool};

use super::executor::{DefaultSubAgentExecutorFactory, SubAgentExecutorConfig};
use super::spec::{SubAgentKind, SubAgentRegistry};
use super::task_tool::TaskTool;
use super::TASK_SYSTEM_PROMPT;

/// Configuration for SubAgentMiddleware
#[derive(Clone)]
pub struct SubAgentMiddlewareConfig {
    /// Default LLM for subagents
    pub default_model: Arc<dyn LLMProvider>,

    /// Backend for file operations
    pub backend: Arc<dyn Backend>,

    /// Registered subagents
    pub subagents: Vec<SubAgentKind>,

    /// Custom system prompt (None uses default)
    pub system_prompt: Option<String>,

    /// Whether to include a general-purpose agent automatically
    pub include_general_purpose: bool,

    /// Maximum iterations for subagent execution
    pub max_iterations: usize,

    /// Default middleware for all subagents
    pub default_middleware: Vec<Arc<dyn AgentMiddleware>>,
}

impl SubAgentMiddlewareConfig {
    /// Create a new configuration with required components
    pub fn new(default_model: Arc<dyn LLMProvider>, backend: Arc<dyn Backend>) -> Self {
        Self {
            default_model,
            backend,
            subagents: Vec::new(),
            system_prompt: None,
            include_general_purpose: false,
            max_iterations: 25,
            default_middleware: Vec::new(),
        }
    }

    /// Add a subagent
    pub fn with_subagent(mut self, subagent: SubAgentKind) -> Self {
        self.subagents.push(subagent);
        self
    }

    /// Add multiple subagents
    pub fn with_subagents(mut self, subagents: Vec<SubAgentKind>) -> Self {
        self.subagents.extend(subagents);
        self
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Include a general-purpose agent
    pub fn with_general_purpose(mut self) -> Self {
        self.include_general_purpose = true;
        self
    }

    /// Set max iterations for subagent execution
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Add default middleware for all subagents
    pub fn with_default_middleware(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.default_middleware.push(middleware);
        self
    }
}

/// Middleware that provides task delegation to sub-agents
///
/// This middleware injects the `task` tool and system prompt explaining
/// how to delegate work to specialized sub-agents.
pub struct SubAgentMiddleware {
    /// The task tool for delegation
    task_tool: Arc<TaskTool>,

    /// System prompt addition
    system_prompt: String,

    /// Whether any subagents are registered
    has_subagents: bool,
}

impl SubAgentMiddleware {
    /// Create a new SubAgentMiddleware with the given configuration
    pub fn new(config: SubAgentMiddlewareConfig) -> Self {
        // Build registry
        let mut registry = SubAgentRegistry::new();
        for subagent in config.subagents {
            registry.register(subagent);
        }

        // Add general-purpose agent if requested
        if config.include_general_purpose && !registry.contains("general-purpose") {
            use super::spec::SubAgentSpec;

            let general_purpose = SubAgentSpec::builder("general-purpose")
                .description("General-purpose agent for miscellaneous tasks")
                .system_prompt(
                    "You are a helpful general-purpose assistant. \
                     Complete the task described to you thoroughly and accurately.",
                )
                .build();

            registry.register(SubAgentKind::Spec(general_purpose));
        }

        let has_subagents = !registry.is_empty();

        // Build executor config
        let executor_config = SubAgentExecutorConfig::new(
            config.default_model.clone(),
            config.backend.clone(),
        )
        .with_max_iterations(config.max_iterations);

        // Create executor factory
        let executor_factory = Arc::new(DefaultSubAgentExecutorFactory::new(executor_config));

        // Create task tool
        let task_tool = Arc::new(TaskTool::new(Arc::new(registry), executor_factory));

        // Build system prompt
        let system_prompt = config
            .system_prompt
            .unwrap_or_else(|| TASK_SYSTEM_PROMPT.to_string());

        Self {
            task_tool,
            system_prompt,
            has_subagents,
        }
    }

    /// Create a builder for more control
    pub fn builder(
        default_model: Arc<dyn LLMProvider>,
        backend: Arc<dyn Backend>,
    ) -> SubAgentMiddlewareBuilder {
        SubAgentMiddlewareBuilder::new(default_model, backend)
    }

    /// Check if any subagents are registered
    pub fn has_subagents(&self) -> bool {
        self.has_subagents
    }
}

#[async_trait]
impl AgentMiddleware for SubAgentMiddleware {
    fn name(&self) -> &str {
        "subagent"
    }

    fn tools(&self) -> Vec<DynTool> {
        if self.has_subagents {
            vec![self.task_tool.clone()]
        } else {
            // Don't inject task tool if no subagents are registered
            vec![]
        }
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        if self.has_subagents {
            format!("{}\n\n{}", prompt, self.system_prompt)
        } else {
            prompt
        }
    }
}

/// Builder for SubAgentMiddleware
pub struct SubAgentMiddlewareBuilder {
    config: SubAgentMiddlewareConfig,
}

impl SubAgentMiddlewareBuilder {
    /// Create a new builder
    pub fn new(default_model: Arc<dyn LLMProvider>, backend: Arc<dyn Backend>) -> Self {
        Self {
            config: SubAgentMiddlewareConfig::new(default_model, backend),
        }
    }

    /// Add a subagent
    pub fn with_subagent(mut self, subagent: SubAgentKind) -> Self {
        self.config = self.config.with_subagent(subagent);
        self
    }

    /// Set custom system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config = self.config.with_system_prompt(prompt);
        self
    }

    /// Include general-purpose agent
    pub fn with_general_purpose(mut self) -> Self {
        self.config = self.config.with_general_purpose();
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config = self.config.with_max_iterations(max);
        self
    }

    /// Build the middleware
    pub fn build(self) -> SubAgentMiddleware {
        SubAgentMiddleware::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::llm::{LLMConfig, LLMResponse};
    use crate::middleware::ToolDefinition;
    use crate::state::Message;

    struct MockLLM;

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, crate::error::DeepAgentError> {
            Ok(LLMResponse::new(Message::assistant("Done")))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    fn create_test_config() -> SubAgentMiddlewareConfig {
        let model = Arc::new(MockLLM);
        let backend = Arc::new(MemoryBackend::new());
        SubAgentMiddlewareConfig::new(model, backend)
    }

    #[test]
    fn test_middleware_with_subagents() {
        use super::super::spec::SubAgentSpec;

        let config = create_test_config()
            .with_subagent(SubAgentKind::Spec(SubAgentSpec::new(
                "researcher",
                "Research agent",
            )));

        let middleware = SubAgentMiddleware::new(config);

        assert!(middleware.has_subagents());
        assert_eq!(middleware.tools().len(), 1);
        assert!(middleware
            .modify_system_prompt("Base".to_string())
            .contains("Task Delegation"));
    }

    #[test]
    fn test_middleware_without_subagents() {
        let config = create_test_config();
        let middleware = SubAgentMiddleware::new(config);

        assert!(!middleware.has_subagents());
        assert!(middleware.tools().is_empty());
        assert_eq!(
            middleware.modify_system_prompt("Base".to_string()),
            "Base"
        );
    }

    #[test]
    fn test_middleware_with_general_purpose() {
        let config = create_test_config().with_general_purpose();
        let middleware = SubAgentMiddleware::new(config);

        assert!(middleware.has_subagents());
        assert_eq!(middleware.tools().len(), 1);

        // Check that general-purpose is in the tool definition
        let tool_def = middleware.tools()[0].definition();
        assert!(tool_def.description.contains("general-purpose"));
    }

    #[test]
    fn test_middleware_custom_system_prompt() {
        use super::super::spec::SubAgentSpec;

        let config = create_test_config()
            .with_subagent(SubAgentKind::Spec(SubAgentSpec::new("test", "Test")))
            .with_system_prompt("Custom instructions");

        let middleware = SubAgentMiddleware::new(config);

        let prompt = middleware.modify_system_prompt("Base".to_string());
        assert!(prompt.contains("Custom instructions"));
    }

    #[test]
    fn test_middleware_builder() {
        use super::super::spec::SubAgentSpec;

        let model = Arc::new(MockLLM);
        let backend = Arc::new(MemoryBackend::new());

        let middleware = SubAgentMiddleware::builder(model, backend)
            .with_subagent(SubAgentKind::Spec(SubAgentSpec::new(
                "researcher",
                "Research",
            )))
            .with_max_iterations(10)
            .with_general_purpose()
            .build();

        assert!(middleware.has_subagents());
        // Should have both researcher and general-purpose
        let tool_def = middleware.tools()[0].definition();
        assert!(tool_def.description.contains("researcher"));
        assert!(tool_def.description.contains("general-purpose"));
    }

    #[test]
    fn test_middleware_name() {
        let config = create_test_config();
        let middleware = SubAgentMiddleware::new(config);
        assert_eq!(middleware.name(), "subagent");
    }
}
