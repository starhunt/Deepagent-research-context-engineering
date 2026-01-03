//! TaskTool implementation for SubAgent delegation
//!
//! This module provides the `task` tool that allows agents to delegate
//! work to specialized sub-agents.
//!
//! # How It Works
//!
//! 1. Agent calls `task(subagent_type, description)`
//! 2. TaskTool validates the request and checks recursion limit
//! 3. Looks up the subagent in the registry
//! 4. Creates isolated state (no messages/todos from parent)
//! 5. Executes subagent with the task description
//! 6. Returns the subagent's response as a ToolMessage
//!
//! Python Reference: deepagents/middleware/subagents.py (_create_task_tool)

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

use super::executor::SubAgentExecutorFactory;
use super::spec::SubAgentRegistry;
use super::state_isolation::IsolatedState;

/// Arguments for the task tool
#[derive(Debug, Deserialize, Serialize)]
pub struct TaskArgs {
    /// Type/name of subagent to invoke
    pub subagent_type: String,

    /// Task description for the subagent
    pub description: String,
}

/// Task tool for delegating work to sub-agents
///
/// This tool enables agent orchestration by allowing the main agent
/// to delegate specialized tasks to sub-agents.
///
/// # Example
///
/// ```rust,ignore
/// let task_tool = TaskTool::new(registry, executor_factory);
///
/// // In agent's tool call:
/// // task(subagent_type="researcher", description="Research quantum computing")
/// ```
pub struct TaskTool {
    /// Registry of available sub-agents
    registry: Arc<SubAgentRegistry>,

    /// Factory for executing sub-agents
    executor_factory: Arc<dyn SubAgentExecutorFactory>,

    /// Custom tool description (optional)
    custom_description: Option<String>,
}

impl TaskTool {
    /// Create a new TaskTool
    pub fn new(
        registry: Arc<SubAgentRegistry>,
        executor_factory: Arc<dyn SubAgentExecutorFactory>,
    ) -> Self {
        Self {
            registry,
            executor_factory,
            custom_description: None,
        }
    }

    /// Set a custom tool description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.custom_description = Some(description.into());
        self
    }

    /// Generate tool description with available agents
    fn generate_description(&self) -> String {
        let base_description = self.custom_description.clone().unwrap_or_else(|| {
            "Delegate a task to a specialized sub-agent. Use this tool to delegate \
             work that requires specialized expertise or when you want to parallelize \
             independent tasks."
                .to_string()
        });

        let agent_descriptions = self.registry.format_descriptions();

        format!("{}\n\n{}", base_description, agent_descriptions)
    }

    /// Generate JSON schema for parameters
    fn generate_parameters_schema(&self) -> serde_json::Value {
        let agent_names: Vec<&str> = self.registry.agent_names();

        if agent_names.is_empty() {
            // No agents registered - still provide the schema
            serde_json::json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of sub-agent to use for this task"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description for the sub-agent"
                    }
                },
                "required": ["subagent_type", "description"]
            })
        } else {
            // Include enum of available agents
            serde_json::json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of sub-agent to use for this task",
                        "enum": agent_names
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description for the sub-agent"
                    }
                },
                "required": ["subagent_type", "description"]
            })
        }
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task".to_string(),
            description: self.generate_description(),
            parameters: self.generate_parameters_schema(),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        // Parse arguments
        let args: TaskArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid task arguments: {}", e)))?;

        tracing::info!(
            subagent_type = %args.subagent_type,
            description_len = args.description.len(),
            "Executing task tool"
        );

        // Check recursion limit
        if runtime.is_recursion_limit_exceeded() {
            tracing::warn!(
                current = runtime.config().current_recursion,
                max = runtime.config().max_recursion,
                "Recursion limit exceeded"
            );
            return Err(MiddlewareError::RecursionLimit(format!(
                "Maximum recursion depth ({}) exceeded. Cannot delegate to '{}'.",
                runtime.config().max_recursion,
                args.subagent_type
            )));
        }

        // Look up subagent
        let subagent = self.registry.get(&args.subagent_type).ok_or_else(|| {
            tracing::error!(subagent_type = %args.subagent_type, "SubAgent not found");
            MiddlewareError::SubAgentNotFound(format!(
                "Unknown sub-agent type: '{}'. Available: {:?}",
                args.subagent_type,
                self.registry.agent_names()
            ))
        })?;

        // Create isolated state from parent
        let isolated_state = IsolatedState::from_parent(runtime.state());

        // Create child runtime with increased recursion
        let child_runtime = runtime.with_increased_recursion();

        tracing::debug!(
            recursion_depth = child_runtime.config().current_recursion,
            "Executing subagent"
        );

        // Execute subagent
        let result = self
            .executor_factory
            .execute(subagent, &args.description, isolated_state, &child_runtime)
            .await?;

        tracing::info!(
            subagent_type = %args.subagent_type,
            success = result.success,
            files_modified = result.files.len(),
            "SubAgent execution completed"
        );

        // Format response
        if result.success {
            Ok(ToolResult::new(format!(
                "[SubAgent '{}' completed]\n\n{}",
                args.subagent_type, result.final_message
            )))
        } else {
            Ok(ToolResult::new(format!(
                "[SubAgent '{}' failed]\n\n{}",
                args.subagent_type, result.final_message
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::middleware::subagent::executor::MockSubAgentExecutorFactory;
    use crate::middleware::subagent::spec::{SubAgentKind, SubAgentSpec};
    use crate::runtime::RuntimeConfig;
    use crate::state::AgentState;

    fn create_test_registry() -> SubAgentRegistry {
        SubAgentRegistry::new()
            .with_agent(SubAgentKind::Spec(SubAgentSpec::new(
                "researcher",
                "Conducts web research",
            )))
            .with_agent(SubAgentKind::Spec(SubAgentSpec::new(
                "synthesizer",
                "Synthesizes findings",
            )))
    }

    fn create_test_runtime() -> ToolRuntime {
        let backend = Arc::new(MemoryBackend::new());
        ToolRuntime::new(AgentState::new(), backend)
    }

    #[test]
    fn test_task_tool_definition() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor);

        let definition = tool.definition();

        assert_eq!(definition.name, "task");
        assert!(definition.description.contains("researcher"));
        assert!(definition.description.contains("synthesizer"));

        // Check parameters schema
        let params = definition.parameters;
        assert!(params["properties"]["subagent_type"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "researcher"));
    }

    #[tokio::test]
    async fn test_task_tool_execute_success() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Research completed!"));
        let tool = TaskTool::new(registry, executor);

        let runtime = create_test_runtime();

        let args = serde_json::json!({
            "subagent_type": "researcher",
            "description": "Research quantum computing"
        });

        let result = tool.execute(args, &runtime).await.unwrap();

        assert!(result.message.contains("Research completed!"));
        assert!(result.message.contains("researcher"));
    }

    #[tokio::test]
    async fn test_task_tool_unknown_agent() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor);

        let runtime = create_test_runtime();

        let args = serde_json::json!({
            "subagent_type": "unknown_agent",
            "description": "Do something"
        });

        let result = tool.execute(args, &runtime).await;

        assert!(result.is_err());
        match result {
            Err(MiddlewareError::SubAgentNotFound(msg)) => {
                assert!(msg.contains("unknown_agent"));
            }
            _ => panic!("Expected SubAgentNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_task_tool_recursion_limit() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor);

        let backend = Arc::new(MemoryBackend::new());
        let config = RuntimeConfig::with_max_recursion(2);
        let mut runtime = ToolRuntime::new(AgentState::new(), backend).with_config(config);

        // Exceed recursion limit
        runtime = runtime.with_increased_recursion();
        runtime = runtime.with_increased_recursion();

        let args = serde_json::json!({
            "subagent_type": "researcher",
            "description": "Research something"
        });

        let result = tool.execute(args, &runtime).await;

        assert!(result.is_err());
        match result {
            Err(MiddlewareError::RecursionLimit(msg)) => {
                assert!(msg.contains("2"));
            }
            _ => panic!("Expected RecursionLimit error"),
        }
    }

    #[tokio::test]
    async fn test_task_tool_invalid_args() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor);

        let runtime = create_test_runtime();

        // Missing required field
        let args = serde_json::json!({
            "subagent_type": "researcher"
            // missing "description"
        });

        let result = tool.execute(args, &runtime).await;

        assert!(result.is_err());
        match result {
            Err(MiddlewareError::ToolExecution(msg)) => {
                assert!(msg.contains("Invalid task arguments"));
            }
            _ => panic!("Expected ToolExecution error"),
        }
    }

    #[test]
    fn test_task_tool_custom_description() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor).with_description("Custom description");

        let definition = tool.definition();

        assert!(definition.description.contains("Custom description"));
    }

    #[test]
    fn test_task_tool_empty_registry() {
        let registry = Arc::new(SubAgentRegistry::new());
        let executor = Arc::new(MockSubAgentExecutorFactory::new("Result"));
        let tool = TaskTool::new(registry, executor);

        let definition = tool.definition();

        // Should still have valid schema
        assert!(definition.parameters["properties"]["subagent_type"].is_object());
        assert!(definition.description.contains("No subagents available"));
    }
}
