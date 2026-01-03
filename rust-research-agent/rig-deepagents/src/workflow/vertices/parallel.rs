//! Parallel execution vertices (FanOut/FanIn)
//!
//! Implements vertices for parallelizing workflow execution and synchronizing results.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::pregel::error::PregelError;
use crate::pregel::message::WorkflowMessage;
use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::{ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId, VertexState};
use crate::workflow::node::{FanInNodeConfig, FanOutNodeConfig, MergeStrategy, SplitStrategy};

/// Type alias for FanIn's message buffer (source_id, message)
type ReceivedMessages = Arc<Mutex<Vec<(Option<String>, WorkflowMessage)>>>;

/// FanOut Vertex: Dispatches messages to multiple targets
pub struct FanOutVertex<S: WorkflowState> {
    id: VertexId,
    config: FanOutNodeConfig,
    /// Round-robin counter
    rr_counter: Arc<Mutex<usize>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState> FanOutVertex<S> {
    pub fn new(id: impl Into<VertexId>, config: FanOutNodeConfig) -> Self {
        Self {
            id: id.into(),
            config,
            rr_counter: Arc::new(Mutex::new(0)),
            _phantom: std::marker::PhantomData,
        }
    }

    fn get_next_target(&self) -> Option<VertexId> {
        if self.config.targets.is_empty() {
            return None;
        }
        let mut counter = self.rr_counter.lock().unwrap();
        let idx = *counter % self.config.targets.len();
        *counter += 1;
        Some(VertexId::new(&self.config.targets[idx]))
    }
}

#[async_trait]
impl<S: WorkflowState> Vertex<S, WorkflowMessage> for FanOutVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        if self.config.targets.is_empty() {
            return Ok(ComputeResult::halt(S::Update::empty()));
        }

        for msg in ctx.messages {
            match self.config.split_strategy {
                SplitStrategy::Broadcast => {
                    // Convert targets to iterator of &str which implements Into<VertexId>
                    ctx.broadcast(self.config.targets.iter().map(|t| t.as_str()), msg.clone());
                }
                SplitStrategy::RoundRobin => {
                    if let Some(target) = self.get_next_target() {
                        ctx.send_message(target, msg.clone());
                    }
                }
                SplitStrategy::Split => {
                    // Try to extract array from message payload
                    let items = match msg {
                        WorkflowMessage::Data { value, .. } => {
                            let root = value;
                            // If split_path is provided, try to navigate to it
                            if let Some(path) = &self.config.split_path {
                                root.pointer(path)
                                    .or_else(|| root.get(path))
                                    .cloned()
                            } else {
                                Some(root.clone())
                            }
                        }
                        _ => None,
                    };

                    if let Some(Value::Array(arr)) = items {
                        // Distribute items to targets
                        // Strategy: 1-to-1 if counts match, otherwise round-robin
                        for (i, item) in arr.into_iter().enumerate() {
                            let target_idx = i % self.config.targets.len();
                            let target = &self.config.targets[target_idx];
                            
                            ctx.send_message(
                                target.as_str(),
                                WorkflowMessage::Data {
                                    key: format!("item_{}", i),
                                    value: item,
                                },
                            );
                        }
                    } else {
                        // Fallback: broadcast if not an array
                        ctx.broadcast(self.config.targets.iter().map(|t| t.as_str()), msg.clone());
                    }
                }
            }
        }

        Ok(ComputeResult::halt(S::Update::empty()))
    }
}

/// FanIn Vertex: Waits for multiple sources and merges results
pub struct FanInVertex<S: WorkflowState> {
    id: VertexId,
    config: FanInNodeConfig,
    /// Store received messages.
    /// Since we can't easily identify sender of Data messages, we track count and payload.
    /// Vector stores (source_id_opt, message)
    received: ReceivedMessages,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState> FanInVertex<S> {
    pub fn new(id: impl Into<VertexId>, config: FanInNodeConfig) -> Self {
        Self {
            id: id.into(),
            config,
            received: Arc::new(Mutex::new(Vec::new())),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if we have received input from all expected sources
    fn check_completion(&self, received: &[(Option<String>, WorkflowMessage)]) -> bool {
        // If we know sources, check if we have messages from all of them
        // For Data messages without source ID, we fall back to counting
        
        let expected_count = self.config.sources.len();
        if received.len() >= expected_count {
            return true;
        }

        // Check for explicitly completed sources
        let completed_sources: usize = received
            .iter()
            .filter_map(|(src, msg)| {
                match msg {
                    WorkflowMessage::Completed { source, .. } => Some(source.as_str()),
                    _ => src.as_deref(),
                }
            })
            .fold(HashSet::new(), |mut acc, src| {
                acc.insert(src.to_string());
                acc
            })
            .len();

        // If we have distinct sources matching expected count
        if completed_sources >= expected_count {
            return true;
        }

        false
    }

    fn merge_results(&self, received: Vec<(Option<String>, WorkflowMessage)>) -> Value {
        // Extract values from messages
        let values: Vec<Value> = received
            .into_iter()
            .filter_map(|(_, msg)| match msg {
                WorkflowMessage::Data { value, .. } => Some(value),
                WorkflowMessage::Completed { result: Some(res), .. } => {
                    Some(Value::String(res))
                }
                WorkflowMessage::ResearchFinding { summary, .. } => {
                    Some(Value::String(summary))
                }
                _ => None,
            })
            .collect();

        match self.config.merge_strategy {
            MergeStrategy::Collect => Value::Array(values),
            MergeStrategy::First => values.first().cloned().unwrap_or(Value::Null),
            MergeStrategy::Last => values.last().cloned().unwrap_or(Value::Null),
            MergeStrategy::Concat => {
                let s = values
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                Value::String(s)
            }
            MergeStrategy::Merge => {
                let mut merged = json!({});
                for val in values {
                    if let Value::Object(map) = val {
                        for (k, v) in map {
                            merged[k] = v;
                        }
                    }
                }
                merged
            }
        }
    }
}

#[async_trait]
impl<S: WorkflowState> Vertex<S, WorkflowMessage> for FanInVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        let mut received_lock = self.received.lock().unwrap();

        // Process incoming messages
        for msg in ctx.messages {
            let source = match msg {
                WorkflowMessage::Completed { source, .. } => Some(source.as_str().to_string()),
                _ => None,
            };
            received_lock.push((source, msg.clone()));
        }

        // Check if ready to merge
        if self.check_completion(&received_lock) {
            let received_data = std::mem::take(&mut *received_lock);
            let result = self.merge_results(received_data);

            // Send merged result
            ctx.send_message(
                "output",
                WorkflowMessage::Data {
                    key: "merged_result".to_string(),
                    value: result,
                },
            );

            // Also signal completion
            ctx.send_message(
                "output",
                WorkflowMessage::Completed {
                    source: self.id.clone(),
                    result: None,
                },
            );

            Ok(ComputeResult::halt(S::Update::empty()))
        } else {
            // Keep waiting
            Ok(ComputeResult::active(S::Update::empty()))
        }
    }
    
    // Ensure vertex stays alive to receive more messages
    fn on_reactivation(&self, _messages: &[WorkflowMessage]) -> VertexState {
        VertexState::Active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::state::UnitState;

    // Helper to create test context
    fn create_ctx<'a>(
        vertex_id: &'a str,
        messages: &'a [WorkflowMessage],
        state: &'a UnitState,
    ) -> ComputeContext<'a, UnitState, WorkflowMessage> {
        ComputeContext::new(VertexId::new(vertex_id), messages, 0, state)
    }

    #[tokio::test]
    async fn test_fanout_broadcast() {
        let config = FanOutNodeConfig {
            targets: vec!["a".into(), "b".into()],
            split_strategy: SplitStrategy::Broadcast,
            ..Default::default()
        };
        let vertex = FanOutVertex::<UnitState>::new("fanout", config);
        let msg = WorkflowMessage::data("test", 1);
        
        let messages = [msg];
        let mut ctx = create_ctx("fanout", &messages, &UnitState);
        let result = vertex.compute(&mut ctx).await.unwrap();

        assert!(result.state.is_halted());
        let outbox = ctx.into_outbox();
        assert_eq!(outbox.len(), 2);
        assert!(outbox.contains_key(&VertexId::new("a")));
        assert!(outbox.contains_key(&VertexId::new("b")));
    }

    #[tokio::test]
    async fn test_fanout_split_array() {
        let config = FanOutNodeConfig {
            targets: vec!["a".into(), "b".into()],
            split_strategy: SplitStrategy::Split,
            split_path: Some("/items".into()),
        };
        let vertex = FanOutVertex::<UnitState>::new("fanout", config);
        
        // Input message with array at /items
        let msg = WorkflowMessage::Data {
            key: "input".into(),
            value: json!({ "items": [1, 2, 3, 4] }),
        };
        
        let messages = [msg];
        let mut ctx = create_ctx("fanout", &messages, &UnitState);
        vertex.compute(&mut ctx).await.unwrap();

        let outbox = ctx.into_outbox();
        
        // Should distribute 4 items to 2 targets
        let msgs_a = outbox.get(&VertexId::new("a")).unwrap();
        let msgs_b = outbox.get(&VertexId::new("b")).unwrap();
        
        assert_eq!(msgs_a.len(), 2); // 1, 3
        assert_eq!(msgs_b.len(), 2); // 2, 4
    }

    #[tokio::test]
    async fn test_fanin_collect_all() {
        let config = FanInNodeConfig {
            sources: vec!["a".into(), "b".into()],
            merge_strategy: MergeStrategy::Collect,
            ..Default::default()
        };
        let vertex = FanInVertex::<UnitState>::new("fanin", config);

        // First superstep: receive from 'a'
        let msg_a = WorkflowMessage::Data { key: "a".into(), value: json!(1) };
        let messages1 = [msg_a];
        let mut ctx1 = create_ctx("fanin", &messages1, &UnitState);
        let res1 = vertex.compute(&mut ctx1).await.unwrap();
        assert!(res1.state.is_active()); // Waiting for 'b'

        // Second superstep: receive from 'b'
        let msg_b = WorkflowMessage::Data { key: "b".into(), value: json!(2) };
        let messages2 = [msg_b];
        let mut ctx2 = create_ctx("fanin", &messages2, &UnitState);
        let res2 = vertex.compute(&mut ctx2).await.unwrap();
        
        assert!(res2.state.is_halted()); // Done
        
        let outbox = ctx2.into_outbox();
        let output = &outbox.get(&VertexId::new("output")).unwrap()[0];
        
        if let WorkflowMessage::Data { value, .. } = output {
            assert!(value.is_array());
            let arr = value.as_array().unwrap();
            assert_eq!(arr.len(), 2);
        } else {
            panic!("Expected Data message");
        }
    }

    #[tokio::test]
    async fn test_fanin_merge_objects() {
        let config = FanInNodeConfig {
            sources: vec!["a".into(), "b".into()],
            merge_strategy: MergeStrategy::Merge,
            ..Default::default()
        };
        let vertex = FanInVertex::<UnitState>::new("fanin", config);

        let msgs = vec![
            WorkflowMessage::Data { key: "1".into(), value: json!({"x": 1}) },
            WorkflowMessage::Data { key: "2".into(), value: json!({"y": 2}) },
        ];
        
        let mut ctx = create_ctx("fanin", &msgs, &UnitState);
        vertex.compute(&mut ctx).await.unwrap();

        let outbox = ctx.into_outbox();
        let output = &outbox.get(&VertexId::new("output")).unwrap()[0];
        
        if let WorkflowMessage::Data { value, .. } = output {
            assert_eq!(value["x"], 1);
            assert_eq!(value["y"], 2);
        }
    }

    #[tokio::test]
    async fn test_fanin_waits_for_all_sources() {
        let config = FanInNodeConfig {
            sources: vec!["a".into(), "b".into(), "c".into()],
            ..Default::default()
        };
        let vertex = FanInVertex::<UnitState>::new("fanin", config);

        // Receive 2 out of 3
        let msgs = vec![
            WorkflowMessage::data("1", 1),
            WorkflowMessage::data("2", 2),
        ];
        
        let mut ctx = create_ctx("fanin", &msgs, &UnitState);
        let res = vertex.compute(&mut ctx).await.unwrap();
        
        assert!(res.state.is_active());
        assert!(ctx.into_outbox().is_empty());
    }
}