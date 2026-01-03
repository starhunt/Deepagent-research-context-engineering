//! SubAgent specification types
//!
//! This module defines the core types for SubAgent specifications,
//! following the Python DeepAgents pattern of SubAgent vs CompiledSubAgent.
//!
//! Python Reference: deepagents/middleware/subagents.py

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::llm::LLMProvider;
use crate::middleware::{AgentMiddleware, DynTool};

/// SubAgent specification for dynamic agent creation
///
/// This represents a "Simple SubAgent" in Python DeepAgents terminology.
/// The agent is created at runtime from this specification when invoked.
///
/// # Example
///
/// ```rust,ignore
/// let researcher = SubAgentSpec::builder("researcher")
///     .description("Conducts web research on topics")
///     .system_prompt("You are a research agent...")
///     .tool(Arc::new(TavilySearchTool::new()))
///     .build();
/// ```
#[derive(Clone)]
pub struct SubAgentSpec {
    /// Unique name for this subagent (used in task tool)
    pub name: String,

    /// Description shown to orchestrator for delegation decisions
    pub description: String,

    /// System prompt for the subagent
    pub system_prompt: String,

    /// Tools available to this subagent
    pub tools: Vec<DynTool>,

    /// Optional model override (uses default if None)
    pub model: Option<Arc<dyn LLMProvider>>,

    /// Additional middleware for this subagent
    pub middleware: Vec<Arc<dyn AgentMiddleware>>,

    /// Maximum execution time
    pub timeout: Option<Duration>,

    /// Maximum iterations for the agent loop
    pub max_iterations: Option<usize>,
}

impl SubAgentSpec {
    /// Create a new SubAgentSpec builder
    pub fn builder(name: impl Into<String>) -> SubAgentSpecBuilder {
        SubAgentSpecBuilder::new(name)
    }

    /// Create a minimal SubAgentSpec
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            system_prompt: String::new(),
            tools: Vec::new(),
            model: None,
            middleware: Vec::new(),
            timeout: None,
            max_iterations: None,
        }
    }
}

/// Builder for SubAgentSpec
pub struct SubAgentSpecBuilder {
    spec: SubAgentSpec,
}

impl SubAgentSpecBuilder {
    /// Create a new builder with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            spec: SubAgentSpec {
                name: name.into(),
                description: String::new(),
                system_prompt: String::new(),
                tools: Vec::new(),
                model: None,
                middleware: Vec::new(),
                timeout: None,
                max_iterations: None,
            },
        }
    }

    /// Set the description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.spec.description = description.into();
        self
    }

    /// Set the system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.spec.system_prompt = prompt.into();
        self
    }

    /// Add a tool
    pub fn tool(mut self, tool: DynTool) -> Self {
        self.spec.tools.push(tool);
        self
    }

    /// Add multiple tools
    pub fn tools(mut self, tools: Vec<DynTool>) -> Self {
        self.spec.tools.extend(tools);
        self
    }

    /// Set the model override
    pub fn model(mut self, model: Arc<dyn LLMProvider>) -> Self {
        self.spec.model = Some(model);
        self
    }

    /// Add middleware
    pub fn middleware(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.spec.middleware.push(middleware);
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.spec.timeout = Some(timeout);
        self
    }

    /// Set max iterations
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.spec.max_iterations = Some(max);
        self
    }

    /// Build the SubAgentSpec
    pub fn build(self) -> SubAgentSpec {
        self.spec
    }
}

/// Pre-compiled subagent executor
///
/// This is a trait for executors that can be pre-compiled and reused.
/// Implementations typically wrap an AgentExecutor with specific configuration.
#[async_trait::async_trait]
pub trait CompiledSubAgentExecutor: Send + Sync {
    /// Execute the subagent with the given prompt
    async fn execute(
        &self,
        prompt: &str,
        files: HashMap<String, crate::state::FileData>,
    ) -> Result<SubAgentResult, crate::error::MiddlewareError>;
}

/// Pre-compiled subagent (for CompiledSubAgent pattern)
///
/// This represents a "CompiledSubAgent" in Python DeepAgents terminology.
/// The agent is pre-compiled and ready for immediate execution.
///
/// # Example
///
/// ```rust,ignore
/// // Create a compiled researcher agent
/// let researcher = CompiledSubAgent::new(
///     "researcher",
///     "Autonomous research agent",
///     Arc::new(MyResearcherExecutor::new()),
/// );
/// ```
pub struct CompiledSubAgent {
    /// Unique name for this subagent
    pub name: String,

    /// Description shown to orchestrator
    pub description: String,

    /// Pre-compiled executor
    pub executor: Arc<dyn CompiledSubAgentExecutor>,
}

impl Clone for CompiledSubAgent {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            executor: self.executor.clone(),
        }
    }
}

impl CompiledSubAgent {
    /// Create a new CompiledSubAgent
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        executor: Arc<dyn CompiledSubAgentExecutor>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            executor,
        }
    }
}

/// Result from subagent execution
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    /// Final message content from subagent
    pub final_message: String,

    /// Files created/modified by the subagent
    pub files: HashMap<String, crate::state::FileData>,

    /// Whether the subagent completed successfully
    pub success: bool,
}

impl SubAgentResult {
    /// Create a successful result
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            final_message: message.into(),
            files: HashMap::new(),
            success: true,
        }
    }

    /// Create a failed result
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            final_message: message.into(),
            files: HashMap::new(),
            success: false,
        }
    }

    /// Add files to the result
    pub fn with_files(mut self, files: HashMap<String, crate::state::FileData>) -> Self {
        self.files = files;
        self
    }
}

/// Unified SubAgent type (either spec or compiled)
///
/// This enum allows the registry to hold both types of subagents.
#[derive(Clone)]
pub enum SubAgentKind {
    /// Dynamically created from specification
    Spec(SubAgentSpec),
    /// Pre-compiled and ready to execute
    Compiled(CompiledSubAgent),
}

impl SubAgentKind {
    /// Get the name of this subagent
    pub fn name(&self) -> &str {
        match self {
            SubAgentKind::Spec(spec) => &spec.name,
            SubAgentKind::Compiled(compiled) => &compiled.name,
        }
    }

    /// Get the description of this subagent
    pub fn description(&self) -> &str {
        match self {
            SubAgentKind::Spec(spec) => &spec.description,
            SubAgentKind::Compiled(compiled) => &compiled.description,
        }
    }

    /// Check if this is a compiled subagent
    pub fn is_compiled(&self) -> bool {
        matches!(self, SubAgentKind::Compiled(_))
    }
}

/// Registry of available subagents
///
/// Provides lookup by name and formatted descriptions for the task tool.
///
/// # Example
///
/// ```rust,ignore
/// let mut registry = SubAgentRegistry::new();
/// registry.register(SubAgentKind::Spec(researcher_spec));
/// registry.register(SubAgentKind::Compiled(synthesizer));
///
/// // Get available agents for task tool description
/// let descriptions = registry.format_descriptions();
/// ```
#[derive(Default)]
pub struct SubAgentRegistry {
    agents: HashMap<String, SubAgentKind>,
}

impl SubAgentRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subagent
    pub fn register(&mut self, agent: SubAgentKind) {
        let name = agent.name().to_string();
        self.agents.insert(name, agent);
    }

    /// Get a subagent by name
    pub fn get(&self, name: &str) -> Option<&SubAgentKind> {
        self.agents.get(name)
    }

    /// Check if a subagent exists
    pub fn contains(&self, name: &str) -> bool {
        self.agents.contains_key(name)
    }

    /// Get all registered agent names
    pub fn agent_names(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of registered agents
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Format descriptions for task tool
    ///
    /// Returns a formatted string listing all available subagents.
    pub fn format_descriptions(&self) -> String {
        if self.agents.is_empty() {
            return "No subagents available.".to_string();
        }

        let mut lines = Vec::new();
        lines.push("Available subagents:".to_string());

        for (name, agent) in &self.agents {
            let kind = if agent.is_compiled() { "compiled" } else { "spec" };
            lines.push(format!(
                "- **{}** [{}]: {}",
                name,
                kind,
                agent.description()
            ));
        }

        lines.join("\n")
    }

    /// Create a builder-style registry
    pub fn with_agent(mut self, agent: SubAgentKind) -> Self {
        self.register(agent);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_spec_builder() {
        let spec = SubAgentSpec::builder("researcher")
            .description("Conducts research")
            .system_prompt("You are a researcher")
            .max_iterations(10)
            .timeout(Duration::from_secs(60))
            .build();

        assert_eq!(spec.name, "researcher");
        assert_eq!(spec.description, "Conducts research");
        assert_eq!(spec.system_prompt, "You are a researcher");
        assert_eq!(spec.max_iterations, Some(10));
        assert_eq!(spec.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_subagent_result() {
        let result = SubAgentResult::success("Task completed")
            .with_files(HashMap::new());

        assert!(result.success);
        assert_eq!(result.final_message, "Task completed");
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = SubAgentRegistry::new();

        let researcher = SubAgentSpec::new("researcher", "Research agent");
        registry.register(SubAgentKind::Spec(researcher));

        assert!(registry.contains("researcher"));
        assert!(!registry.contains("unknown"));
        assert_eq!(registry.len(), 1);

        let names = registry.agent_names();
        assert!(names.contains(&"researcher"));
    }

    #[test]
    fn test_registry_format_descriptions() {
        let registry = SubAgentRegistry::new()
            .with_agent(SubAgentKind::Spec(
                SubAgentSpec::new("researcher", "Conducts web research"),
            ))
            .with_agent(SubAgentKind::Spec(
                SubAgentSpec::new("synthesizer", "Synthesizes findings"),
            ));

        let descriptions = registry.format_descriptions();

        assert!(descriptions.contains("researcher"));
        assert!(descriptions.contains("synthesizer"));
        assert!(descriptions.contains("Conducts web research"));
    }

    #[test]
    fn test_subagent_kind_methods() {
        let spec = SubAgentKind::Spec(SubAgentSpec::new("test", "Test agent"));

        assert_eq!(spec.name(), "test");
        assert_eq!(spec.description(), "Test agent");
        assert!(!spec.is_compiled());
    }
}
