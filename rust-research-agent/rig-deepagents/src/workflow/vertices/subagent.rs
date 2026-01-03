//! SubAgentVertex: Pregel vertex for SubAgent delegation in workflows
//!
//! This vertex allows workflows to delegate tasks to specialized sub-agents
//! within the Pregel execution model.
//!
//! # Usage in Workflows
//!
//! ```rust,ignore
//! let subagent_vertex = SubAgentVertex::new(
//!     "researcher_node",
//!     SubAgentNodeConfig {
//!         agent_name: "researcher".to_string(),
//!         max_recursion: 5,
//!         ..Default::default()
//!     },
//!     registry,
//!     executor_factory,
//! );
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use crate::backends::Backend;
use crate::middleware::subagent::{
    IsolatedState, SubAgentExecutorFactory, SubAgentRegistry,
};
use crate::pregel::error::PregelError;
use crate::pregel::message::WorkflowMessage;
use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::{ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId};
use crate::workflow::node::SubAgentNodeConfig;

/// A Pregel vertex that delegates to a SubAgent
///
/// This vertex extracts a prompt from incoming messages, looks up the
/// specified sub-agent, and executes it with isolated state.
pub struct SubAgentVertex<S: WorkflowState> {
    /// Vertex identifier
    id: VertexId,

    /// SubAgent configuration
    config: SubAgentNodeConfig,

    /// Registry of available sub-agents
    registry: Arc<SubAgentRegistry>,

    /// Factory for executing sub-agents
    executor_factory: Arc<dyn SubAgentExecutorFactory>,

    /// Backend for file operations (H3 fix: use workflow's backend instead of fresh MemoryBackend)
    backend: Arc<dyn Backend>,

    /// Phantom data for state type
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState> SubAgentVertex<S> {
    /// Create a new SubAgentVertex
    ///
    /// # Arguments
    ///
    /// * `id` - Unique vertex identifier
    /// * `config` - SubAgent node configuration
    /// * `registry` - Registry of available sub-agents
    /// * `executor_factory` - Factory for creating sub-agent executors
    /// * `backend` - Backend for file operations (H3 fix: files are persisted to workflow's backend)
    pub fn new(
        id: impl Into<VertexId>,
        config: SubAgentNodeConfig,
        registry: Arc<SubAgentRegistry>,
        executor_factory: Arc<dyn SubAgentExecutorFactory>,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self {
            id: id.into(),
            config,
            registry,
            executor_factory,
            backend,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Extract prompt from incoming messages
    fn extract_prompt(&self, messages: &[WorkflowMessage]) -> String {
        messages
            .iter()
            .filter_map(|m| match m {
                WorkflowMessage::Data { value, .. } => {
                    // Try to extract string value
                    value.as_str().map(|s| s.to_string())
                }
                WorkflowMessage::Completed { result, .. } => result.clone(),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl<S: WorkflowState> Vertex<S, WorkflowMessage> for SubAgentVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        tracing::info!(
            vertex_id = %self.id,
            agent_name = %self.config.agent_name,
            superstep = ctx.superstep,
            "SubAgentVertex compute starting"
        );

        // Check recursion limit using superstep as proxy
        if ctx.superstep >= self.config.max_recursion {
            tracing::warn!(
                vertex_id = %self.id,
                superstep = ctx.superstep,
                max_recursion = self.config.max_recursion,
                "Max recursion depth reached"
            );

            // Send error message and halt
            ctx.send_message(
                "output",
                WorkflowMessage::Data {
                    key: "error".to_string(),
                    value: serde_json::json!({
                        "error": "max_recursion_exceeded",
                        "message": format!(
                            "SubAgent '{}' exceeded max recursion depth of {}",
                            self.config.agent_name,
                            self.config.max_recursion
                        )
                    }),
                },
            );

            return Ok(ComputeResult::halt(S::Update::empty()));
        }

        // Extract prompt from incoming messages
        let prompt = self.extract_prompt(ctx.messages);

        if prompt.is_empty() {
            tracing::warn!(
                vertex_id = %self.id,
                "No prompt found in incoming messages"
            );

            ctx.send_message(
                "output",
                WorkflowMessage::Data {
                    key: "error".to_string(),
                    value: serde_json::json!({
                        "error": "no_prompt",
                        "message": "No prompt was provided to SubAgent"
                    }),
                },
            );

            return Ok(ComputeResult::halt(S::Update::empty()));
        }

        // Look up subagent
        let subagent = self.registry.get(&self.config.agent_name).ok_or_else(|| {
            PregelError::vertex_error(
                self.id.clone(),
                format!("SubAgent '{}' not found in registry", self.config.agent_name),
            )
        })?;

        // Create isolated state (empty for now - could be enhanced to carry files from workflow state)
        let isolated_state = IsolatedState::new();

        // Create runtime with workflow's backend (H3 fix)
        // This ensures subagent file operations persist to the workflow's backend
        use crate::runtime::ToolRuntime;
        use crate::state::AgentState;

        let runtime = ToolRuntime::new(AgentState::new(), self.backend.clone());

        // Execute subagent
        let result = self
            .executor_factory
            .execute(subagent, &prompt, isolated_state, &runtime)
            .await
            .map_err(|e| {
                PregelError::vertex_error(
                    self.id.clone(),
                    format!("SubAgent execution failed: {}", e),
                )
            })?;

        tracing::info!(
            vertex_id = %self.id,
            success = result.success,
            "SubAgent execution completed"
        );

        // Send result to output
        ctx.send_message(
            "output",
            WorkflowMessage::Data {
                key: "subagent_result".to_string(),
                value: serde_json::json!({
                    "agent": self.config.agent_name,
                    "success": result.success,
                    "message": result.final_message,
                    "files_modified": result.files.len(),
                }),
            },
        );

        // Also send completion message
        ctx.send_message(
            "output",
            WorkflowMessage::Completed {
                source: self.id.clone(),
                result: Some(result.final_message.clone()),
            },
        );

        Ok(ComputeResult::halt(S::Update::empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::middleware::subagent::{SubAgentKind, SubAgentResult, SubAgentSpec};
    use crate::pregel::state::UnitState;

    // Mock executor factory for testing
    struct MockExecutorFactory {
        response: String,
    }

    #[async_trait]
    impl SubAgentExecutorFactory for MockExecutorFactory {
        async fn execute(
            &self,
            _subagent: &SubAgentKind,
            _prompt: &str,
            _state: IsolatedState,
            _runtime: &crate::runtime::ToolRuntime,
        ) -> Result<SubAgentResult, crate::error::MiddlewareError> {
            Ok(SubAgentResult::success(&self.response))
        }
    }

    fn create_test_registry() -> SubAgentRegistry {
        SubAgentRegistry::new()
            .with_agent(SubAgentKind::Spec(SubAgentSpec::new(
                "researcher",
                "Research agent",
            )))
    }

    fn create_test_backend() -> Arc<dyn Backend> {
        Arc::new(MemoryBackend::new())
    }

    #[test]
    fn test_subagent_vertex_creation() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockExecutorFactory {
            response: "Done".to_string(),
        });
        let backend = create_test_backend();

        let config = SubAgentNodeConfig {
            agent_name: "researcher".to_string(),
            ..Default::default()
        };

        let vertex: SubAgentVertex<UnitState> =
            SubAgentVertex::new("subagent_node", config, registry, executor, backend);

        assert_eq!(vertex.id().as_str(), "subagent_node");
    }

    #[test]
    fn test_extract_prompt_from_data_message() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockExecutorFactory {
            response: "Done".to_string(),
        });
        let backend = create_test_backend();

        let config = SubAgentNodeConfig {
            agent_name: "researcher".to_string(),
            ..Default::default()
        };

        let vertex: SubAgentVertex<UnitState> =
            SubAgentVertex::new("test", config, registry, executor, backend);

        let messages = vec![WorkflowMessage::Data {
            key: "prompt".to_string(),
            value: serde_json::json!("Research quantum computing"),
        }];

        let prompt = vertex.extract_prompt(&messages);
        assert_eq!(prompt, "Research quantum computing");
    }

    #[test]
    fn test_extract_prompt_multiple_messages() {
        let registry = Arc::new(create_test_registry());
        let executor = Arc::new(MockExecutorFactory {
            response: "Done".to_string(),
        });
        let backend = create_test_backend();

        let config = SubAgentNodeConfig {
            agent_name: "researcher".to_string(),
            ..Default::default()
        };

        let vertex: SubAgentVertex<UnitState> =
            SubAgentVertex::new("test", config, registry, executor, backend);

        let messages = vec![
            WorkflowMessage::Data {
                key: "prompt1".to_string(),
                value: serde_json::json!("Part 1"),
            },
            WorkflowMessage::Data {
                key: "prompt2".to_string(),
                value: serde_json::json!("Part 2"),
            },
        ];

        let prompt = vertex.extract_prompt(&messages);
        assert!(prompt.contains("Part 1"));
        assert!(prompt.contains("Part 2"));
    }
}
