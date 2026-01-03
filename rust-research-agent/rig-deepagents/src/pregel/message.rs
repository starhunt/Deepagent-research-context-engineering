//! Message types for Pregel vertex communication
//!
//! Vertices communicate by sending messages to each other.
//! Messages are delivered at the start of each superstep.

use serde::{Deserialize, Serialize};
use super::vertex::VertexId;

/// Trait bound for vertex messages
pub trait VertexMessage: Clone + Send + Sync + 'static {
    /// Create an activation message for edge-driven routing
    fn activation_message() -> Self;
}

/// Priority level for research directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Priority {
    High,
    #[default]
    Medium,
    Low,
}

/// Source information for research findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// URL of the source
    pub url: String,
    /// Title of the source
    pub title: String,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
}

impl Source {
    /// Create a new source
    pub fn new(url: impl Into<String>, title: impl Into<String>, relevance: f32) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            relevance: relevance.clamp(0.0, 1.0),
        }
    }
}

/// Standard message types for workflow coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowMessage {
    /// Trigger vertex activation
    Activate,

    /// Pass data between vertices
    Data {
        key: String,
        value: serde_json::Value,
    },

    /// Signal completion of upstream work
    Completed {
        source: VertexId,
        result: Option<String>,
    },

    /// Request vertex to halt
    Halt,

    /// Research-specific: share findings
    ResearchFinding {
        query: String,
        sources: Vec<Source>,
        summary: String,
    },

    /// Research-specific: suggest new direction
    ResearchDirection {
        topic: String,
        priority: Priority,
        rationale: String,
    },
}

impl VertexMessage for WorkflowMessage {
    fn activation_message() -> Self {
        WorkflowMessage::Activate
    }
}

impl WorkflowMessage {
    /// Create a Data message
    pub fn data(key: impl Into<String>, value: impl Serialize) -> Self {
        Self::Data {
            key: key.into(),
            value: serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
        }
    }

    /// Create a Completed message
    pub fn completed(source: impl Into<VertexId>, result: Option<String>) -> Self {
        Self::Completed {
            source: source.into(),
            result,
        }
    }

    /// Create a ResearchFinding message
    pub fn research_finding(
        query: impl Into<String>,
        sources: Vec<Source>,
        summary: impl Into<String>,
    ) -> Self {
        Self::ResearchFinding {
            query: query.into(),
            sources,
            summary: summary.into(),
        }
    }

    /// Create a ResearchDirection message
    pub fn research_direction(
        topic: impl Into<String>,
        priority: Priority,
        rationale: impl Into<String>,
    ) -> Self {
        Self::ResearchDirection {
            topic: topic.into(),
            priority,
            rationale: rationale.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_workflow_message_serialization() {
        let msg = WorkflowMessage::Data {
            key: "query".into(),
            value: json!("test query"),
        };
        let json_str = serde_json::to_string(&msg).unwrap();
        let deserialized: WorkflowMessage = serde_json::from_str(&json_str).unwrap();

        // Verify roundtrip
        match deserialized {
            WorkflowMessage::Data { key, value } => {
                assert_eq!(key, "query");
                assert_eq!(value, json!("test query"));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_research_finding_message() {
        let msg = WorkflowMessage::ResearchFinding {
            query: "rust async".into(),
            sources: vec![Source {
                url: "https://example.com".into(),
                title: "Example".into(),
                relevance: 0.95,
            }],
            summary: "Rust async is great".into(),
        };
        assert!(matches!(msg, WorkflowMessage::ResearchFinding { .. }));
    }

    #[test]
    fn test_research_direction_message() {
        let msg = WorkflowMessage::research_direction(
            "async runtimes",
            Priority::High,
            "Important for understanding concurrency",
        );
        match msg {
            WorkflowMessage::ResearchDirection {
                topic,
                priority,
                rationale,
            } => {
                assert_eq!(topic, "async runtimes");
                assert_eq!(priority, Priority::High);
                assert!(!rationale.is_empty());
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_source_relevance_clamping() {
        let source = Source::new("https://test.com", "Test", 1.5);
        assert_eq!(source.relevance, 1.0);

        let source = Source::new("https://test.com", "Test", -0.5);
        assert_eq!(source.relevance, 0.0);
    }

    #[test]
    fn test_completed_message() {
        let msg = WorkflowMessage::completed("planner", Some("Plan complete".to_string()));
        match msg {
            WorkflowMessage::Completed { source, result } => {
                assert_eq!(source.0, "planner");
                assert_eq!(result, Some("Plan complete".to_string()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_data_message_helper() {
        let msg = WorkflowMessage::data("count", 42);
        match msg {
            WorkflowMessage::Data { key, value } => {
                assert_eq!(key, "count");
                assert_eq!(value, json!(42));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Medium);
    }

    #[test]
    fn test_activate_and_halt() {
        let activate = WorkflowMessage::Activate;
        let halt = WorkflowMessage::Halt;

        assert!(matches!(activate, WorkflowMessage::Activate));
        assert!(matches!(halt, WorkflowMessage::Halt));
    }

    #[test]
    fn test_activation_message() {
        let msg = WorkflowMessage::activation_message();
        assert!(matches!(msg, WorkflowMessage::Activate));
    }
}
