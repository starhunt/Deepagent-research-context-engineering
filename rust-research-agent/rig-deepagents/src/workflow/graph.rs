//! WorkflowGraph builder DSL.
//!
//! Provides a fluent API for defining nodes, edges, and entry points,
//! then validates and compiles the graph into a built representation.

use std::collections::HashMap;
use std::marker::PhantomData;

use thiserror::Error;

use crate::pregel::WorkflowState;
use crate::workflow::node::NodeKind;

/// Sentinel target for terminal edges.
pub const END: &str = "END";

/// Node definition for a workflow graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub name: String,
    pub kind: NodeKind,
}

/// Edge definition for a workflow graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub condition: Option<String>,
}

/// Errors that can occur while building a workflow graph.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkflowBuildError {
    #[error("workflow entry point not set")]
    NoEntryPoint,
    #[error("unknown node id: {0}")]
    UnknownNode(String),
}

/// Builder for constructing workflow graphs with fluent API.
#[derive(Debug, Clone)]
pub struct WorkflowGraph<S: WorkflowState> {
    name: String,
    nodes: HashMap<String, NodeKind>,
    edges: Vec<GraphEdge>,
    entry_point: Option<String>,
    _state: PhantomData<S>,
}

impl<S: WorkflowState> Default for WorkflowGraph<S> {
    fn default() -> Self {
        Self {
            name: String::new(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: None,
            _state: PhantomData,
        }
    }
}

impl<S: WorkflowState> WorkflowGraph<S> {
    /// Create a new workflow graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a node definition.
    pub fn node(mut self, id: impl Into<String>, kind: NodeKind) -> Self {
        self.nodes.insert(id.into(), kind);
        self
    }

    /// Set the entry point node.
    pub fn entry(mut self, id: impl Into<String>) -> Self {
        self.entry_point = Some(id.into());
        self
    }

    /// Add a direct edge between nodes.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(GraphEdge {
            from: from.into(),
            to: to.into(),
            condition: None,
        });
        self
    }

    /// Add conditional edges from a node.
    pub fn conditional_edges(mut self, from: impl Into<String>, edges: Vec<(&str, &str)>) -> Self {
        let from = from.into();
        for (condition, target) in edges {
            self.edges.push(GraphEdge {
                from: from.clone(),
                to: target.to_string(),
                condition: Some(condition.to_string()),
            });
        }
        self
    }

    /// Validate and build the workflow graph.
    pub fn build(self) -> Result<BuiltWorkflowGraph<S>, WorkflowBuildError> {
        let entry_point = self.entry_point.ok_or(WorkflowBuildError::NoEntryPoint)?;

        if !self.nodes.contains_key(&entry_point) {
            return Err(WorkflowBuildError::UnknownNode(entry_point));
        }

        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        for edge in self.edges {
            if !self.nodes.contains_key(&edge.from) {
                return Err(WorkflowBuildError::UnknownNode(edge.from));
            }
            if edge.to != END && !self.nodes.contains_key(&edge.to) {
                return Err(WorkflowBuildError::UnknownNode(edge.to));
            }
            edges.entry(edge.from).or_default().push(edge.to);
        }

        Ok(BuiltWorkflowGraph {
            nodes: self.nodes,
            edges,
            entry_point,
            name: self.name,
            _state: PhantomData,
        })
    }
}

/// Built workflow graph representation.
#[derive(Debug, Clone)]
pub struct BuiltWorkflowGraph<S: WorkflowState> {
    pub nodes: HashMap<String, NodeKind>,
    pub edges: HashMap<String, Vec<String>>,
    pub entry_point: String,
    pub name: String,
    _state: PhantomData<S>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::UnitState;

    #[test]
    fn test_workflow_builder_basic() {
        let workflow = WorkflowGraph::<UnitState>::new()
            .name("basic")
            .node("start", NodeKind::Passthrough)
            .node("next", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "next")
            .build()
            .unwrap();

        assert_eq!(workflow.name, "basic");
        assert_eq!(workflow.entry_point, "start");
        assert!(workflow.nodes.contains_key("start"));
        assert_eq!(
            workflow.edges.get("start"),
            Some(&vec!["next".to_string()])
        );
    }

    #[test]
    fn test_workflow_builder_missing_entry() {
        let result = WorkflowGraph::<UnitState>::new()
            .node("start", NodeKind::Passthrough)
            .build();

        assert_eq!(result.unwrap_err(), WorkflowBuildError::NoEntryPoint);
    }

    #[test]
    fn test_workflow_builder_invalid_edge() {
        let result = WorkflowGraph::<UnitState>::new()
            .node("start", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "missing")
            .build();

        assert_eq!(
            result.unwrap_err(),
            WorkflowBuildError::UnknownNode("missing".to_string())
        );
    }

    #[test]
    fn test_workflow_conditional_edges() {
        let workflow = WorkflowGraph::<UnitState>::new()
            .node("start", NodeKind::Passthrough)
            .node("a", NodeKind::Passthrough)
            .node("b", NodeKind::Passthrough)
            .entry("start")
            .conditional_edges("start", vec![("if_a", "a"), ("if_b", "b")])
            .build()
            .unwrap();

        assert_eq!(
            workflow.edges.get("start"),
            Some(&vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_workflow_end_sentinel() {
        let workflow = WorkflowGraph::<UnitState>::new()
            .node("start", NodeKind::Passthrough)
            .entry("start")
            .edge("start", END)
            .build()
            .unwrap();

        assert_eq!(
            workflow.edges.get("start"),
            Some(&vec![END.to_string()])
        );
    }
}
