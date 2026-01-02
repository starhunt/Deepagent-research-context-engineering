//! Vertex (Node) abstractions for Pregel runtime
//!
//! A Vertex represents a computation unit in the workflow graph.
//! Vertices communicate via messages and execute in synchronized supersteps.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use super::error::PregelError;
use super::message::VertexMessage;

/// Unique identifier for a vertex in the workflow graph
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VertexId(pub String);

impl VertexId {
    /// Create a new VertexId
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for VertexId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for VertexId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for VertexId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Vertex execution state (Pregel's "vote to halt" mechanism)
///
/// - `Active`: Vertex will compute in the next superstep
/// - `Halted`: Vertex has voted to halt (will reactivate on message receipt)
/// - `Completed`: Vertex has finished and will not reactivate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VertexState {
    /// Vertex is active and will compute in next superstep
    #[default]
    Active,
    /// Vertex has voted to halt (will reactivate on message receipt)
    Halted,
    /// Vertex has completed and will not reactivate
    Completed,
}

impl VertexState {
    /// Check if the vertex is active
    pub fn is_active(&self) -> bool {
        matches!(self, VertexState::Active)
    }

    /// Check if the vertex is halted (can be reactivated)
    pub fn is_halted(&self) -> bool {
        matches!(self, VertexState::Halted)
    }

    /// Check if the vertex is completed (cannot be reactivated)
    pub fn is_completed(&self) -> bool {
        matches!(self, VertexState::Completed)
    }
}

/// Trait for state updates produced by vertex computation
///
/// State updates are collected from all vertices and merged at the end of each superstep.
pub trait StateUpdate: Clone + Send + Sync + 'static {
    /// Create an empty (no-op) update
    fn empty() -> Self;

    /// Check if this update has no effect
    fn is_empty(&self) -> bool;
}

/// Context provided to a vertex during computation
///
/// Provides access to:
/// - Incoming messages from other vertices
/// - Outbox for sending messages
/// - Current superstep number
/// - Workflow state (read-only)
pub struct ComputeContext<'a, S, M: VertexMessage> {
    /// Messages received from other vertices
    pub messages: &'a [M],
    /// Current superstep number (0-indexed)
    pub superstep: usize,
    /// Read-only access to workflow state
    pub state: &'a S,
    /// Outgoing messages (target vertex -> messages)
    outbox: HashMap<VertexId, Vec<M>>,
    /// Current vertex ID
    vertex_id: VertexId,
}

impl<'a, S, M: VertexMessage> ComputeContext<'a, S, M> {
    /// Create a new compute context
    pub fn new(vertex_id: VertexId, messages: &'a [M], superstep: usize, state: &'a S) -> Self {
        Self {
            messages,
            superstep,
            state,
            outbox: HashMap::new(),
            vertex_id,
        }
    }

    /// Get the current vertex ID
    pub fn id(&self) -> &VertexId {
        &self.vertex_id
    }

    /// Send a message to another vertex
    ///
    /// Messages will be delivered at the start of the next superstep.
    pub fn send_message(&mut self, target: impl Into<VertexId>, message: M) {
        let target = target.into();
        self.outbox.entry(target).or_default().push(message);
    }

    /// Send a message to multiple targets
    pub fn broadcast(&mut self, targets: impl IntoIterator<Item = impl Into<VertexId>>, message: M) {
        for target in targets {
            self.send_message(target.into(), message.clone());
        }
    }

    /// Check if this is the first superstep
    pub fn is_first_superstep(&self) -> bool {
        self.superstep == 0
    }

    /// Check if any messages were received
    pub fn has_messages(&self) -> bool {
        !self.messages.is_empty()
    }

    /// Get the count of received messages
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Consume the context and return the outbox
    pub fn into_outbox(self) -> HashMap<VertexId, Vec<M>> {
        self.outbox
    }
}

use super::state::WorkflowState;

/// The core vertex trait for Pregel computation
///
/// Each vertex in the workflow graph implements this trait.
/// During each superstep, active vertices have their `compute` method called.
///
/// # Type Parameters
///
/// - `S`: The workflow state type (must implement WorkflowState)
/// - `M`: The message type used for vertex communication
///
/// # Example
///
/// ```ignore
/// struct EchoVertex {
///     id: VertexId,
/// }
///
/// #[async_trait]
/// impl Vertex<MyState, WorkflowMessage> for EchoVertex {
///     fn id(&self) -> &VertexId {
///         &self.id
///     }
///
///     async fn compute(
///         &self,
///         ctx: &mut ComputeContext<'_, MyState, WorkflowMessage>,
///     ) -> Result<ComputeResult<MyUpdate>, PregelError> {
///         for msg in ctx.messages {
///             if let WorkflowMessage::Data { key, value } = msg {
///                 ctx.send_message("output", WorkflowMessage::data(format!("echo_{}", key), value.clone()));
///             }
///         }
///         Ok(ComputeResult::halt(MyUpdate::empty()))
///     }
/// }
/// ```
#[async_trait]
pub trait Vertex<S, M>: Send + Sync
where
    S: WorkflowState,
    M: VertexMessage,
{
    /// Get the vertex's unique identifier
    fn id(&self) -> &VertexId;

    /// Execute the vertex's computation
    ///
    /// Called during each superstep for active vertices.
    /// Returns a state update and the next vertex state.
    async fn compute(
        &self,
        ctx: &mut ComputeContext<'_, S, M>,
    ) -> Result<ComputeResult<S::Update>, PregelError>;

    /// Combine multiple messages into one (optional optimization)
    ///
    /// Default implementation returns messages unchanged.
    /// Override to reduce message traffic for commutative/associative operations.
    fn combine_messages(&self, messages: Vec<M>) -> Vec<M> {
        messages
    }

    /// Called when the vertex receives messages while halted
    ///
    /// By default, returns `Active` to reactivate the vertex.
    fn on_reactivation(&self, _messages: &[M]) -> VertexState {
        VertexState::Active
    }
}

/// Result of a vertex computation
#[derive(Debug, Clone)]
pub struct ComputeResult<U: StateUpdate> {
    /// State update to apply
    pub update: U,
    /// New vertex state
    pub state: VertexState,
}

impl<U: StateUpdate> ComputeResult<U> {
    /// Create a result that keeps the vertex active
    pub fn active(update: U) -> Self {
        Self {
            update,
            state: VertexState::Active,
        }
    }

    /// Create a result that halts the vertex
    pub fn halt(update: U) -> Self {
        Self {
            update,
            state: VertexState::Halted,
        }
    }

    /// Create a result that completes the vertex
    pub fn complete(update: U) -> Self {
        Self {
            update,
            state: VertexState::Completed,
        }
    }

    /// Create a result with a specific state
    pub fn with_state(update: U, state: VertexState) -> Self {
        Self { update, state }
    }
}

/// Boxed vertex for dynamic dispatch
pub type BoxedVertex<S, M> = Arc<dyn Vertex<S, M>>;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::message::WorkflowMessage;

    // Test StateUpdate implementation for tests
    #[derive(Clone, Debug, Default)]
    struct TestUpdate {
        delta: i32,
    }

    impl StateUpdate for TestUpdate {
        fn empty() -> Self {
            TestUpdate { delta: 0 }
        }

        fn is_empty(&self) -> bool {
            self.delta == 0
        }
    }

    // Test state for tests
    #[derive(Clone, Debug, Default)]
    #[allow(dead_code)]
    struct TestState {
        value: i32,
    }

    impl super::super::state::WorkflowState for TestState {
        type Update = TestUpdate;

        fn apply_update(&self, update: Self::Update) -> Self {
            TestState {
                value: self.value + update.delta,
            }
        }

        fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
            TestUpdate {
                delta: updates.iter().map(|u| u.delta).sum(),
            }
        }
    }

    // Mock vertex for testing
    struct EchoVertex {
        id: VertexId,
    }

    #[async_trait]
    impl Vertex<TestState, WorkflowMessage> for EchoVertex {
        fn id(&self) -> &VertexId {
            &self.id
        }

        async fn compute(
            &self,
            ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
        ) -> Result<ComputeResult<TestUpdate>, PregelError> {
            // Echo back any data messages
            for msg in ctx.messages {
                if let WorkflowMessage::Data { key, value } = msg {
                    ctx.send_message(
                        "output",
                        WorkflowMessage::data(format!("echo_{}", key), value.clone()),
                    );
                }
            }
            Ok(ComputeResult::halt(TestUpdate::empty()))
        }
    }

    #[tokio::test]
    async fn test_mock_vertex_compute() {
        let vertex = EchoVertex {
            id: VertexId::new("echo"),
        };

        let state = TestState { value: 42 };
        let messages = vec![WorkflowMessage::data("test", "hello")];

        let mut ctx = ComputeContext::new(
            VertexId::new("echo"),
            &messages,
            0,
            &state,
        );

        let result = vertex.compute(&mut ctx).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.state.is_halted());

        let outbox = ctx.into_outbox();
        assert!(outbox.contains_key(&VertexId::new("output")));
        assert_eq!(outbox.get(&VertexId::new("output")).unwrap().len(), 1);
    }

    #[test]
    fn test_compute_context_send_message() {
        let state = TestState { value: 0 };
        let messages: Vec<WorkflowMessage> = vec![];

        let mut ctx = ComputeContext::new(
            VertexId::new("test"),
            &messages,
            0,
            &state,
        );

        ctx.send_message("target1", WorkflowMessage::Activate);
        ctx.send_message("target1", WorkflowMessage::Halt);
        ctx.send_message("target2", WorkflowMessage::Activate);

        let outbox = ctx.into_outbox();
        assert_eq!(outbox.get(&VertexId::new("target1")).unwrap().len(), 2);
        assert_eq!(outbox.get(&VertexId::new("target2")).unwrap().len(), 1);
    }

    #[test]
    fn test_compute_context_broadcast() {
        let state = TestState { value: 0 };
        let messages: Vec<WorkflowMessage> = vec![];

        let mut ctx = ComputeContext::new(
            VertexId::new("broadcaster"),
            &messages,
            0,
            &state,
        );

        let targets = vec!["a", "b", "c"];
        ctx.broadcast(targets, WorkflowMessage::Activate);

        let outbox = ctx.into_outbox();
        assert_eq!(outbox.len(), 3);
        assert!(outbox.contains_key(&VertexId::new("a")));
        assert!(outbox.contains_key(&VertexId::new("b")));
        assert!(outbox.contains_key(&VertexId::new("c")));
    }

    #[test]
    fn test_compute_context_helpers() {
        let state = TestState { value: 0 };
        let messages = vec![WorkflowMessage::Activate, WorkflowMessage::Halt];

        let ctx = ComputeContext::<TestState, WorkflowMessage>::new(
            VertexId::new("test"),
            &messages,
            0,
            &state,
        );

        assert!(ctx.is_first_superstep());
        assert!(ctx.has_messages());
        assert_eq!(ctx.message_count(), 2);
        assert_eq!(ctx.id(), &VertexId::new("test"));

        let ctx2 = ComputeContext::<TestState, WorkflowMessage>::new(
            VertexId::new("test2"),
            &[],
            5,
            &state,
        );

        assert!(!ctx2.is_first_superstep());
        assert!(!ctx2.has_messages());
        assert_eq!(ctx2.message_count(), 0);
    }

    #[test]
    fn test_compute_result_constructors() {
        let update = TestUpdate { delta: 5 };

        let active = ComputeResult::active(update.clone());
        assert!(active.state.is_active());

        let halted = ComputeResult::halt(update.clone());
        assert!(halted.state.is_halted());

        let completed = ComputeResult::complete(update.clone());
        assert!(completed.state.is_completed());

        let custom = ComputeResult::with_state(update, VertexState::Halted);
        assert!(custom.state.is_halted());
    }

    #[test]
    fn test_state_update_trait() {
        let empty = TestUpdate::empty();
        assert!(empty.is_empty());

        let non_empty = TestUpdate { delta: 10 };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_vertex_state_helpers() {
        assert!(VertexState::Active.is_active());
        assert!(!VertexState::Active.is_halted());
        assert!(!VertexState::Active.is_completed());

        assert!(!VertexState::Halted.is_active());
        assert!(VertexState::Halted.is_halted());
        assert!(!VertexState::Halted.is_completed());

        assert!(!VertexState::Completed.is_active());
        assert!(!VertexState::Completed.is_halted());
        assert!(VertexState::Completed.is_completed());
    }

    #[test]
    fn test_vertex_id_from_str() {
        let id: VertexId = "planner".into();
        assert_eq!(id.0, "planner");
    }

    #[test]
    fn test_vertex_id_from_string() {
        let id: VertexId = String::from("router").into();
        assert_eq!(id.0, "router");
    }

    #[test]
    fn test_vertex_id_new() {
        let id = VertexId::new("explorer");
        assert_eq!(id.0, "explorer");
    }

    #[test]
    fn test_vertex_id_equality() {
        let id1: VertexId = "node1".into();
        let id2: VertexId = "node1".into();
        let id3: VertexId = "node2".into();
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_vertex_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VertexId::from("a"));
        set.insert(VertexId::from("b"));
        set.insert(VertexId::from("a")); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_vertex_id_display() {
        let id = VertexId::new("test_node");
        assert_eq!(format!("{}", id), "test_node");
    }

    #[test]
    fn test_vertex_state_default_is_active() {
        assert_eq!(VertexState::default(), VertexState::Active);
    }

    #[test]
    fn test_vertex_state_variants() {
        let active = VertexState::Active;
        let halted = VertexState::Halted;
        let completed = VertexState::Completed;

        assert_ne!(active, halted);
        assert_ne!(halted, completed);
        assert_ne!(active, completed);
    }

    #[test]
    fn test_vertex_id_serialization() {
        let id = VertexId::new("test");
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: VertexId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_vertex_state_serialization() {
        let state = VertexState::Halted;
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: VertexState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }
}
