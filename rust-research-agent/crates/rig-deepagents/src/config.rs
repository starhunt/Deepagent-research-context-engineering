//! Production Configuration Module
//!
//! Provides environment-based configuration for production deployments.
//! Reads API keys and settings from environment variables.
//!
//! # Environment Variables
//!
//! | Variable | Purpose | Required |
//! |----------|---------|----------|
//! | `OPENAI_API_KEY` | OpenAI API authentication | Yes (for OpenAI) |
//! | `ANTHROPIC_API_KEY` | Anthropic API authentication | Yes (for Anthropic) |
//! | `TAVILY_API_KEY` | Tavily Search API authentication | Yes (for search) |
//!
//! # Example
//!
//! ```ignore
//! use rig_deepagents::config::ProductionConfig;
//!
//! let config = ProductionConfig::from_env()?;
//!
//! // Get configured LLM provider
//! let llm = config.llm_provider()?;
//!
//! // Get research tools
//! let tools = config.research_tools()?;
//!
//! // Build workflow with production settings
//! let workflow = config.build_research_workflow()?;
//! ```

use std::sync::Arc;
use std::time::Duration;

use crate::error::DeepAgentError;
use crate::llm::{LLMConfig, LLMProvider, OpenAIProvider};
use crate::middleware::{Tool, ToolDefinition};
use crate::pregel::config::ExecutionMode;
use crate::pregel::PregelConfig;
use crate::research::{ResearchConfig, ResearchWorkflowBuilder};
use crate::tools::{TavilySearchTool, ThinkTool};
use crate::workflow::graph::BuiltWorkflowGraph;
use crate::ResearchState;

/// Production configuration loaded from environment variables
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    /// LLM provider to use
    pub llm_provider_type: LLMProviderType,

    /// Model name override (optional)
    pub model: Option<String>,

    /// Temperature for LLM calls
    pub temperature: f64,

    /// Maximum tokens for LLM responses
    pub max_tokens: u64,

    /// Maximum searches in research workflow
    pub max_searches: usize,

    /// Maximum research directions
    pub max_directions: usize,

    /// Workflow timeout in seconds
    pub workflow_timeout_secs: u64,

    /// Vertex (node) timeout in seconds
    pub vertex_timeout_secs: u64,

    /// Checkpoint interval (0 = disabled)
    pub checkpoint_interval: usize,

    /// Parallelism level
    pub parallelism: usize,

    /// Enable tracing
    pub tracing_enabled: bool,

    /// Tavily search retry count
    pub tavily_max_retries: u32,

    /// Tavily search timeout in seconds
    pub tavily_timeout_secs: u64,
}

/// Supported LLM provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LLMProviderType {
    OpenAI,
    Anthropic,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            llm_provider_type: LLMProviderType::OpenAI,
            model: None,
            temperature: 0.0,
            max_tokens: 4096,
            max_searches: 6,
            max_directions: 3,
            workflow_timeout_secs: 3600,
            vertex_timeout_secs: 300,
            checkpoint_interval: 10,
            parallelism: num_cpus::get(),
            tracing_enabled: true,
            tavily_max_retries: 3,
            tavily_timeout_secs: 30,
        }
    }
}

impl ProductionConfig {
    /// Create a new production config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    ///
    /// Reads optional overrides from environment:
    /// - `LLM_PROVIDER`: "openai" or "anthropic"
    /// - `LLM_MODEL`: Model name
    /// - `LLM_TEMPERATURE`: Temperature value
    /// - `MAX_SEARCHES`: Research search budget
    /// - `WORKFLOW_TIMEOUT`: Timeout in seconds
    pub fn from_env() -> Result<Self, DeepAgentError> {
        let mut config = Self::default();

        // LLM provider selection
        if let Ok(provider) = std::env::var("LLM_PROVIDER") {
            config.llm_provider_type = match provider.to_lowercase().as_str() {
                "anthropic" | "claude" => LLMProviderType::Anthropic,
                _ => LLMProviderType::OpenAI,
            };
        }

        // Model override
        if let Ok(model) = std::env::var("LLM_MODEL") {
            config.model = Some(model);
        }

        // Temperature
        if let Ok(temp) = std::env::var("LLM_TEMPERATURE") {
            if let Ok(t) = temp.parse() {
                config.temperature = t;
            }
        }

        // Max searches
        if let Ok(searches) = std::env::var("MAX_SEARCHES") {
            if let Ok(s) = searches.parse() {
                config.max_searches = s;
            }
        }

        // Workflow timeout
        if let Ok(timeout) = std::env::var("WORKFLOW_TIMEOUT") {
            if let Ok(t) = timeout.parse() {
                config.workflow_timeout_secs = t;
            }
        }

        // Parallelism
        if let Ok(par) = std::env::var("PARALLELISM") {
            if let Ok(p) = par.parse() {
                config.parallelism = p;
            }
        }

        Ok(config)
    }

    /// Set the LLM provider type
    pub fn with_provider(mut self, provider: LLMProviderType) -> Self {
        self.llm_provider_type = provider;
        self
    }

    /// Set the model name
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set max searches
    pub fn with_max_searches(mut self, max: usize) -> Self {
        self.max_searches = max;
        self
    }

    /// Set max directions
    pub fn with_max_directions(mut self, max: usize) -> Self {
        self.max_directions = max;
        self
    }

    /// Set workflow timeout
    pub fn with_workflow_timeout(mut self, secs: u64) -> Self {
        self.workflow_timeout_secs = secs;
        self
    }

    /// Set checkpoint interval
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Create the LLM provider based on configuration
    ///
    /// # Environment Variables
    ///
    /// - `OPENAI_API_KEY` - Required for OpenAI provider
    /// - `ANTHROPIC_API_KEY` - Required for Anthropic provider
    pub fn llm_provider(&self) -> Result<Arc<dyn LLMProvider>, DeepAgentError> {
        match self.llm_provider_type {
            LLMProviderType::OpenAI => {
                let provider = match &self.model {
                    Some(model) => OpenAIProvider::from_env_with_model(model)?,
                    None => OpenAIProvider::from_env()?,
                };
                Ok(Arc::new(provider))
            }
            LLMProviderType::Anthropic => {
                use crate::llm::AnthropicProvider;
                let provider = match &self.model {
                    Some(model) => AnthropicProvider::from_env_with_model(model)?,
                    None => AnthropicProvider::from_env()?,
                };
                Ok(Arc::new(provider))
            }
        }
    }

    /// Create LLM configuration
    pub fn llm_config(&self) -> LLMConfig {
        let model = self.model.clone().unwrap_or_else(|| {
            match self.llm_provider_type {
                LLMProviderType::OpenAI => "gpt-4.1".to_string(),
                LLMProviderType::Anthropic => "claude-3-5-sonnet-latest".to_string(),
            }
        });

        LLMConfig::new(model)
            .with_temperature(self.temperature)
            .with_max_tokens(self.max_tokens)
    }

    /// Create research tool definitions
    ///
    /// # Environment Variables
    ///
    /// - `TAVILY_API_KEY` - Required for Tavily search
    pub fn research_tools(&self) -> Result<Vec<ToolDefinition>, DeepAgentError> {
        let tavily = TavilySearchTool::from_env()?
            .with_timeout(Duration::from_secs(self.tavily_timeout_secs))
            .with_max_retries(self.tavily_max_retries);

        let think = ThinkTool;

        Ok(vec![tavily.definition(), think.definition()])
    }

    /// Create Pregel runtime configuration
    pub fn pregel_config(&self) -> PregelConfig {
        PregelConfig::default()
            .with_max_supersteps(100)
            .with_parallelism(self.parallelism)
            .with_checkpoint_interval(self.checkpoint_interval)
            .with_vertex_timeout(Duration::from_secs(self.vertex_timeout_secs))
            .with_workflow_timeout(Duration::from_secs(self.workflow_timeout_secs))
            .with_tracing(self.tracing_enabled)
            .with_execution_mode(ExecutionMode::EdgeDriven)
    }

    /// Create research configuration
    pub fn research_config(&self) -> ResearchConfig {
        ResearchConfig::new()
            .with_max_searches(self.max_searches)
            .with_max_directions(self.max_directions)
            .with_timeout(self.workflow_timeout_secs)
    }

    /// Build research workflow graph
    pub fn build_research_workflow(&self) -> Result<BuiltWorkflowGraph<ResearchState>, DeepAgentError> {
        let workflow_graph = ResearchWorkflowBuilder::new()
            .max_searches(self.max_searches)
            .max_directions(self.max_directions)
            .build()
            .map_err(|e| DeepAgentError::AgentExecution(format!("Workflow build error: {}", e)))?;

        workflow_graph
            .build()
            .map_err(|e| DeepAgentError::AgentExecution(format!("Graph build error: {}", e)))
    }

    /// Create initial research state
    pub fn create_research_state(&self, query: impl Into<String>) -> ResearchState {
        ResearchState::new(query).with_max_searches(self.max_searches)
    }
}

/// Builder for creating a complete production setup
pub struct ProductionSetup {
    config: ProductionConfig,
    llm: Option<Arc<dyn LLMProvider>>,
    tools: Vec<ToolDefinition>,
}

impl ProductionSetup {
    /// Create a new production setup with configuration
    pub fn new(config: ProductionConfig) -> Self {
        Self {
            config,
            llm: None,
            tools: vec![],
        }
    }

    /// Initialize from environment
    pub fn from_env() -> Result<Self, DeepAgentError> {
        let config = ProductionConfig::from_env()?;
        let mut setup = Self::new(config);
        setup.initialize()?;
        Ok(setup)
    }

    /// Initialize LLM and tools from environment
    pub fn initialize(&mut self) -> Result<(), DeepAgentError> {
        self.llm = Some(self.config.llm_provider()?);
        self.tools = self.config.research_tools()?;
        Ok(())
    }

    /// Get the LLM provider
    pub fn llm(&self) -> Option<Arc<dyn LLMProvider>> {
        self.llm.clone()
    }

    /// Get the tool definitions
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Get the Pregel configuration
    pub fn pregel_config(&self) -> PregelConfig {
        self.config.pregel_config()
    }

    /// Build and compile a research workflow
    pub fn build_workflow(
        &self,
    ) -> Result<crate::workflow::CompiledWorkflow<ResearchState>, DeepAgentError> {
        let graph = self.config.build_research_workflow()?;
        let pregel_config = self.config.pregel_config();

        crate::workflow::CompiledWorkflow::compile_with_tools(
            graph,
            pregel_config,
            self.llm.clone(),
            self.tools.clone(),
        )
        .map_err(|e| DeepAgentError::AgentExecution(format!("Workflow compile error: {}", e)))
    }

    /// Create initial research state
    pub fn create_state(&self, query: impl Into<String>) -> ResearchState {
        self.config.create_research_state(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_defaults() {
        let config = ProductionConfig::default();

        assert_eq!(config.llm_provider_type, LLMProviderType::OpenAI);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.max_searches, 6);
        assert_eq!(config.max_directions, 3);
        assert_eq!(config.checkpoint_interval, 10);
        assert!(config.tracing_enabled);
    }

    #[test]
    fn test_production_config_builder() {
        let config = ProductionConfig::new()
            .with_provider(LLMProviderType::Anthropic)
            .with_model("claude-3-opus")
            .with_temperature(0.5)
            .with_max_searches(10)
            .with_max_directions(5)
            .with_workflow_timeout(7200)
            .with_checkpoint_interval(5);

        assert_eq!(config.llm_provider_type, LLMProviderType::Anthropic);
        assert_eq!(config.model, Some("claude-3-opus".to_string()));
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_searches, 10);
        assert_eq!(config.max_directions, 5);
        assert_eq!(config.workflow_timeout_secs, 7200);
        assert_eq!(config.checkpoint_interval, 5);
    }

    #[test]
    fn test_llm_config_creation() {
        let config = ProductionConfig::new()
            .with_model("gpt-4-turbo")
            .with_temperature(0.7);

        let llm_config = config.llm_config();

        assert_eq!(llm_config.model, "gpt-4-turbo");
        assert_eq!(llm_config.temperature, Some(0.7));
        assert_eq!(llm_config.max_tokens, Some(4096));
    }

    #[test]
    fn test_pregel_config_creation() {
        let config = ProductionConfig::new()
            .with_workflow_timeout(1800)
            .with_checkpoint_interval(20);

        let pregel = config.pregel_config();

        assert_eq!(pregel.checkpoint_interval, 20);
        assert!(pregel.tracing_enabled);
    }

    #[test]
    fn test_research_config_creation() {
        let config = ProductionConfig::new()
            .with_max_searches(8)
            .with_max_directions(4);

        let research = config.research_config();

        assert_eq!(research.max_searches, 8);
        assert_eq!(research.max_directions, 4);
    }

    #[test]
    fn test_create_research_state() {
        let config = ProductionConfig::new().with_max_searches(10);

        let state = config.create_research_state("Test query");

        assert_eq!(state.query, "Test query");
        assert_eq!(state.max_searches, 10);
    }
}
