//! RouterVertex: Conditional routing based on state or LLM decisions
//!
//! Implements the Vertex trait for router nodes that route messages
//! based on state field inspection or LLM-based classification.

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

use crate::llm::LLMProvider;
use crate::pregel::error::PregelError;
use crate::pregel::message::WorkflowMessage;
use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::{ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId};
use crate::workflow::node::{Branch, BranchCondition, RouterNodeConfig, RoutingStrategy};

/// A router vertex that routes messages based on state inspection or LLM decisions
pub struct RouterVertex<S: WorkflowState> {
    id: VertexId,
    config: RouterNodeConfig,
    llm: Option<Arc<dyn LLMProvider>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState + Serialize> RouterVertex<S> {
    /// Create a new router vertex
    pub fn new(
        id: impl Into<VertexId>,
        config: RouterNodeConfig,
        llm: Option<Arc<dyn LLMProvider>>,
    ) -> Self {
        Self {
            id: id.into(),
            config,
            llm,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Evaluate a branch condition against a value
    fn evaluate_condition(&self, value: &Value, condition: &BranchCondition) -> bool {
        match condition {
            BranchCondition::Equals { value: expected } => value == expected,
            BranchCondition::In { values } => values.contains(value),
            BranchCondition::Matches { pattern } => {
                if let Some(value_str) = value.as_str() {
                    match regex::Regex::new(pattern) {
                        Ok(re) => re.is_match(value_str),
                        Err(_) => false, // Invalid regex doesn't match
                    }
                } else {
                    false
                }
            }
            BranchCondition::IsTruthy => {
                match value {
                    Value::Bool(b) => *b,
                    Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
                    Value::String(s) => !s.is_empty(),
                    Value::Array(arr) => !arr.is_empty(),
                    Value::Object(obj) => !obj.is_empty(),
                    Value::Null => false,
                }
            }
            BranchCondition::IsFalsy => !self.evaluate_condition(value, &BranchCondition::IsTruthy),
            BranchCondition::Always => true,
        }
    }

    /// Resolve a state field path to a value
    fn resolve_state_field(&self, state: &S, path: &str) -> Option<Value> {
        // This is a simplified implementation that only handles top-level fields
        // For a more complete implementation, we'd need a JSON path library
        // For now, we'll just handle simple field access
        let path_parts: Vec<&str> = path.split('.').collect();
        
        if path_parts.is_empty() {
            return None;
        }

        // For now, just handle the top-level field
        // In a real implementation, we'd need to recursively navigate the state structure
        // This assumes the state has a method to access fields by path
        // Since we don't have a specific method for this, we'll use a generic approach
        // that works with serde_json::Value
        let state_json = serde_json::to_value(state).ok()?;
        
        // Navigate the JSON path
        let mut current = &state_json;
        for part in path_parts {
            if let Value::Object(obj) = current {
                current = obj.get(part)?;
            } else {
                return None;
            }
        }
        
        Some(current.clone())
    }

    /// Route based on state field inspection
    fn route_by_state_field(&self, state: &S, branches: &[Branch], default: Option<&str>) -> Option<String> {
        if let RoutingStrategy::StateField { ref field } = self.config.strategy {
            if let Some(field_value) = self.resolve_state_field(state, field) {
                for branch in branches {
                    if self.evaluate_condition(&field_value, &branch.condition) {
                        return Some(branch.target.clone());
                    }
                }
            }
        }
        
        default.map(|s| s.to_string())
    }

    /// Route based on LLM decision
    async fn route_by_llm_decision(&self, state: &S, branches: &[Branch]) -> Result<Option<String>, PregelError> {
        let llm = self.llm.as_ref().ok_or_else(|| {
            PregelError::vertex_error(
                self.id.clone(),
                "LLM decision strategy requires an LLM provider",
            )
        })?;

        if let RoutingStrategy::LLMDecision { ref prompt, .. } = self.config.strategy {
            // Build a prompt that describes the routing options
            let state_json = serde_json::to_value(state)
                .map_err(|e| PregelError::vertex_error(self.id.clone(), e.to_string()))?;
            
            let mut routing_prompt = prompt.clone();
            routing_prompt.push_str("\n\nCurrent state:\n");
            routing_prompt.push_str(&format!("{}", state_json));
            routing_prompt.push_str("\n\nAvailable branches:\n");
            
            for (i, branch) in branches.iter().enumerate() {
                routing_prompt.push_str(&format!("{}. {}: ", i + 1, branch.target));
                
                match &branch.condition {
                    BranchCondition::Equals { value } => {
                        routing_prompt.push_str(&format!("Value equals {}", value));
                    }
                    BranchCondition::In { values } => {
                        routing_prompt.push_str(&format!("Value in {:?}", values));
                    }
                    BranchCondition::Matches { pattern } => {
                        routing_prompt.push_str(&format!("Value matches regex '{}'", pattern));
                    }
                    BranchCondition::IsTruthy => {
                        routing_prompt.push_str("Value is truthy");
                    }
                    BranchCondition::IsFalsy => {
                        routing_prompt.push_str("Value is falsy");
                    }
                    BranchCondition::Always => {
                        routing_prompt.push_str("Always");
                    }
                }
                routing_prompt.push('\n');
            }
            
            routing_prompt.push_str("\nRespond with only the target branch name that should be selected based on the current state.");
            
            // Call the LLM to make the routing decision
            let messages = vec![crate::state::Message::user(&routing_prompt)];
            let response = llm
                .complete(&messages, &[], None)
                .await
                .map_err(|e| PregelError::vertex_error(self.id.clone(), e.to_string()))?;
            
            // Extract the target branch from the response
            let content = response.message.content.trim();
            
            // Look for the target in the branches
            for branch in branches {
                if content == branch.target {
                    return Ok(Some(branch.target.clone()));
                }
                
                // Also check if the response contains the target name
                if content.to_lowercase().contains(&branch.target.to_lowercase()) {
                    return Ok(Some(branch.target.clone()));
                }
            }
            
            // If no exact match, try to parse as a number (for numbered options)
            if let Ok(index) = content.parse::<usize>() {
                if index > 0 && index <= branches.len() {
                    return Ok(Some(branches[index - 1].target.clone()));
                }
            }
        }
        
        Ok(None)
    }
}

#[async_trait]
impl<S: WorkflowState + Serialize> Vertex<S, WorkflowMessage> for RouterVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        // Determine the target branch based on the routing strategy
        let target = match &self.config.strategy {
            RoutingStrategy::StateField { .. } => {
                self.route_by_state_field(ctx.state, &self.config.branches, self.config.default.as_deref())
            }
            RoutingStrategy::LLMDecision { .. } => {
                self.route_by_llm_decision(ctx.state, &self.config.branches).await?
            }
        };

        // Send the message to the selected target or default
        if let Some(target_vertex) = target {
            // Forward all incoming messages to the target
            for msg in ctx.messages {
                ctx.send_message(target_vertex.clone(), msg.clone());
            }
        } else {
            // If no branch matched and no default was provided, halt
            return Ok(ComputeResult::halt(S::Update::empty()));
        }

        // Router vertices typically halt after routing
        Ok(ComputeResult::halt(S::Update::empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DeepAgentError;
    use crate::llm::{LLMConfig, LLMResponse};
    use crate::pregel::state::UnitUpdate;
    use crate::pregel::vertex::VertexState;
    use crate::state::Message as StateMessage;
    use serde_json::json;
    use std::sync::Mutex;

    // Mock state for testing
    #[derive(Clone, Debug, Default, serde::Serialize)]
    struct TestState {
        phase: String,
        count: i32,
        active: bool,
        tags: Vec<String>,
    }

    impl TestState {
        fn new(phase: &str, count: i32, active: bool, tags: Vec<&str>) -> Self {
            Self {
                phase: phase.to_string(),
                count,
                active,
                tags: tags.into_iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl crate::pregel::state::WorkflowState for TestState {
        type Update = UnitUpdate;

        fn apply_update(&self, _update: Self::Update) -> Self {
            self.clone()
        }

        fn merge_updates(_updates: Vec<Self::Update>) -> Self::Update {
            UnitUpdate
        }

        fn is_terminal(&self) -> bool {
            false
        }
    }

    // Mock LLM provider for testing
    struct MockLLMProvider {
        responses: Arc<Mutex<Vec<String>>>,
    }

    impl MockLLMProvider {
        fn new() -> Self {
            Self {
                responses: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_response(self, content: impl Into<String>) -> Self {
            self.responses.lock().unwrap().push(content.into());
            self
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLMProvider {
        async fn complete(
            &self,
            _messages: &[StateMessage],
            _tools: &[crate::middleware::ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, DeepAgentError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(DeepAgentError::AgentExecution(
                    "No more mock responses".to_string(),
                ));
            }
            let content = responses.remove(0);
            Ok(LLMResponse::new(StateMessage::assistant(&content)))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    #[tokio::test]
    async fn test_router_state_field_equals() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "phase".to_string(),
            },
            branches: vec![
                Branch {
                    target: "exploration".to_string(),
                    condition: BranchCondition::Equals {
                        value: json!("exploratory"),
                    },
                },
                Branch {
                    target: "synthesis".to_string(),
                    condition: BranchCondition::Equals {
                        value: json!("synthesis"),
                    },
                },
            ],
            default: Some("done".to_string()),
        };

        let vertex = RouterVertex::<TestState>::new("router", config, None);

        // Test with "exploratory" phase
        let test_state = TestState::new("exploratory", 0, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("exploration")));
    }

    #[tokio::test]
    async fn test_router_state_field_in() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "count".to_string(),
            },
            branches: vec![
                Branch {
                    target: "low".to_string(),
                    condition: BranchCondition::In {
                        values: vec![json!(1), json!(2), json!(3)],
                    },
                },
                Branch {
                    target: "high".to_string(),
                    condition: BranchCondition::In {
                        values: vec![json!(10), json!(20), json!(30)],
                    },
                },
            ],
            default: Some("other".to_string()),
        };

        let vertex = RouterVertex::<TestState>::new("router", config, None);

        // Test with count = 2 (should go to "low")
        let test_state = TestState::new("test", 2, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("low")));
    }

    #[tokio::test]
    async fn test_router_state_field_matches_regex() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "phase".to_string(),
            },
            branches: vec![
                Branch {
                    target: "search".to_string(),
                    condition: BranchCondition::Matches {
                        pattern: "^search.*".to_string(),
                    },
                },
                Branch {
                    target: "analysis".to_string(),
                    condition: BranchCondition::Matches {
                        pattern: "^analyze.*".to_string(),
                    },
                },
            ],
            default: Some("default".to_string()),
        };

        let vertex = RouterVertex::<TestState>::new("router", config, None);

        // Test with phase = "searching" (should match "^search.*")
        let test_state = TestState::new("searching", 0, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("search")));
    }

    #[tokio::test]
    async fn test_router_default_branch() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "phase".to_string(),
            },
            branches: vec![
                Branch {
                    target: "exploration".to_string(),
                    condition: BranchCondition::Equals {
                        value: json!("exploratory"),
                    },
                },
            ],
            default: Some("fallback".to_string()),
        };

        let vertex = RouterVertex::<TestState>::new("router", config, None);

        // Test with phase = "unknown" (should go to default)
        let test_state = TestState::new("unknown", 0, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("fallback")));
    }

    #[tokio::test]
    async fn test_router_is_truthy_falsy() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "active".to_string(),
            },
            branches: vec![
                Branch {
                    target: "active_route".to_string(),
                    condition: BranchCondition::IsTruthy,
                },
                Branch {
                    target: "inactive_route".to_string(),
                    condition: BranchCondition::IsFalsy,
                },
            ],
            default: Some("default".to_string()),
        };

        let vertex = RouterVertex::<TestState>::new("router", config.clone(), None);

        // Test with active = true (should go to "active_route")
        let test_state = TestState::new("test", 0, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("active_route")));

        // Test with active = false (should go to "inactive_route")
        let vertex2 = RouterVertex::<TestState>::new("router2", config, None);

        let test_state2 = TestState::new("test", 0, false, vec![]);
        let messages2 = vec![WorkflowMessage::data("input", "test")];
        let mut ctx2 = ComputeContext::new(VertexId::new("router2"), &messages2, 0, &test_state2);

        let result2: ComputeResult<UnitUpdate> = vertex2.compute(&mut ctx2).await.unwrap();
        assert_eq!(result2.state, VertexState::Halted);

        let outbox2 = ctx2.into_outbox();
        assert!(outbox2.contains_key(&VertexId::new("inactive_route")));
    }

    #[tokio::test]
    async fn test_router_llm_decision() {
        let config = RouterNodeConfig {
            strategy: RoutingStrategy::LLMDecision {
                prompt: "Route based on the phase".to_string(),
                model: None,
            },
            branches: vec![
                Branch {
                    target: "exploration".to_string(),
                    condition: BranchCondition::Always, // Placeholder for LLM decision
                },
                Branch {
                    target: "synthesis".to_string(),
                    condition: BranchCondition::Always, // Placeholder for LLM decision
                },
            ],
            default: Some("default".to_string()),
        };

        let mock_llm = MockLLMProvider::new().with_response("exploration");
        let vertex = RouterVertex::<TestState>::new("router", config, Some(Arc::new(mock_llm)));

        let test_state = TestState::new("exploratory", 0, true, vec![]);
        let messages = vec![WorkflowMessage::data("input", "test")];
        let mut ctx = ComputeContext::new(VertexId::new("router"), &messages, 0, &test_state);

        let result: ComputeResult<UnitUpdate> = vertex.compute(&mut ctx).await.unwrap();
        assert_eq!(result.state, VertexState::Halted);

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("exploration")));
    }
}