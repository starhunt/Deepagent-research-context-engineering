//! Error types for Pregel runtime
//!
//! Comprehensive error handling for the Pregel execution engine.

use super::vertex::VertexId;
use thiserror::Error;

/// Errors that can occur during Pregel runtime execution
#[derive(Debug, Error)]
pub enum PregelError {
    /// Maximum supersteps exceeded
    #[error("Max supersteps exceeded: {0}")]
    MaxSuperstepsExceeded(usize),

    /// Vertex computation timed out
    #[error("Vertex timeout: {0:?}")]
    VertexTimeout(VertexId),

    /// Error during vertex computation
    #[error("Vertex error in {vertex_id:?}: {message}")]
    VertexError {
        vertex_id: VertexId,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Error during routing decision
    #[error("Routing error in {vertex_id:?}: {decision}")]
    RoutingError { vertex_id: VertexId, decision: String },

    /// Recursion depth limit exceeded
    #[error("Recursion limit in {vertex_id:?}: depth {depth}, limit {limit}")]
    RecursionLimit {
        vertex_id: VertexId,
        depth: usize,
        limit: usize,
    },

    /// Error in workflow state management
    #[error("State error: {0}")]
    StateError(String),

    /// Error in checkpointing
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Invalid workflow configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Message delivery failed
    #[error("Message delivery failed: {0}")]
    MessageDeliveryError(String),

    /// Workflow terminated by user
    #[error("Workflow cancelled")]
    Cancelled,

    /// Workflow execution timed out
    #[error("Workflow timeout after {0:?}")]
    WorkflowTimeout(std::time::Duration),

    /// Maximum retry attempts exceeded for a vertex
    #[error("Max retries exceeded for vertex {vertex_id:?}: {attempts} attempts")]
    MaxRetriesExceeded { vertex_id: VertexId, attempts: usize },

    /// Checkpoint workflow_id mismatch
    #[error("Checkpoint workflow mismatch: expected {expected}, found {found}")]
    CheckpointMismatch { expected: String, found: String },
}

impl PregelError {
    /// Create a vertex error with a message
    pub fn vertex_error(vertex_id: impl Into<VertexId>, message: impl Into<String>) -> Self {
        Self::VertexError {
            vertex_id: vertex_id.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a vertex error with source
    pub fn vertex_error_with_source(
        vertex_id: impl Into<VertexId>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::VertexError {
            vertex_id: vertex_id.into(),
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a routing error
    pub fn routing_error(vertex_id: impl Into<VertexId>, decision: impl Into<String>) -> Self {
        Self::RoutingError {
            vertex_id: vertex_id.into(),
            decision: decision.into(),
        }
    }

    /// Create a recursion limit error
    pub fn recursion_limit(vertex_id: impl Into<VertexId>, depth: usize, limit: usize) -> Self {
        Self::RecursionLimit {
            vertex_id: vertex_id.into(),
            depth,
            limit,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            PregelError::VertexTimeout(_)
                | PregelError::VertexError { .. }
                | PregelError::MessageDeliveryError(_)
        )
    }

    /// Check if the error is a timeout
    pub fn is_timeout(&self) -> bool {
        matches!(self, PregelError::VertexTimeout(_))
    }

    /// Create a checkpoint error
    pub fn checkpoint_error(message: impl Into<String>) -> Self {
        Self::CheckpointError(message.into())
    }

    /// Create a not implemented error
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented(feature.into())
    }

    /// Create a state error
    pub fn state_error(message: impl Into<String>) -> Self {
        Self::StateError(message.into())
    }

    /// Create a config error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError(message.into())
    }

    /// Create a workflow timeout error
    pub fn workflow_timeout(duration: std::time::Duration) -> Self {
        Self::WorkflowTimeout(duration)
    }

    /// Create a max retries exceeded error
    pub fn max_retries_exceeded(vertex_id: impl Into<VertexId>, attempts: usize) -> Self {
        Self::MaxRetriesExceeded {
            vertex_id: vertex_id.into(),
            attempts,
        }
    }

    /// Create a checkpoint mismatch error
    pub fn checkpoint_mismatch(expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self::CheckpointMismatch {
            expected: expected.into(),
            found: found.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    // Ensure errors are Send + Sync (compile-time check)
    static_assertions::assert_impl_all!(super::PregelError: Send, Sync);
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PregelError::MaxSuperstepsExceeded(100);
        assert_eq!(format!("{}", err), "Max supersteps exceeded: 100");
    }

    #[test]
    fn test_vertex_error() {
        let err = PregelError::vertex_error("node1", "computation failed");
        match err {
            PregelError::VertexError {
                vertex_id,
                message,
                source,
            } => {
                assert_eq!(vertex_id.0, "node1");
                assert_eq!(message, "computation failed");
                assert!(source.is_none());
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_vertex_timeout() {
        let err = PregelError::VertexTimeout(VertexId::from("slow_node"));
        assert!(format!("{}", err).contains("slow_node"));
        assert!(err.is_timeout());
    }

    #[test]
    fn test_routing_error() {
        let err = PregelError::routing_error("router", "no matching branch");
        match err {
            PregelError::RoutingError { vertex_id, decision } => {
                assert_eq!(vertex_id.0, "router");
                assert_eq!(decision, "no matching branch");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_recursion_limit() {
        let err = PregelError::recursion_limit("nested_agent", 6, 5);
        match err {
            PregelError::RecursionLimit {
                vertex_id,
                depth,
                limit,
            } => {
                assert_eq!(vertex_id.0, "nested_agent");
                assert_eq!(depth, 6);
                assert_eq!(limit, 5);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_is_recoverable() {
        assert!(PregelError::VertexTimeout(VertexId::from("x")).is_recoverable());
        assert!(PregelError::vertex_error("x", "err").is_recoverable());
        assert!(PregelError::MessageDeliveryError("err".into()).is_recoverable());

        assert!(!PregelError::MaxSuperstepsExceeded(100).is_recoverable());
        assert!(!PregelError::Cancelled.is_recoverable());
        assert!(!PregelError::recursion_limit("x", 5, 3).is_recoverable());
    }

    #[test]
    fn test_errors_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PregelError>();
    }
}
