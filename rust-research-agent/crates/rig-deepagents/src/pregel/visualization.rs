//! Mermaid diagram generation for Pregel workflows
//!
//! This module provides visualization helper functions for workflow graphs.
//! The main visualization methods are on `PregelRuntime` (see `runtime.rs`).
//!
//! # Node Shapes
//!
//! Different node types render with distinct Mermaid shapes:
//!
//! | NodeKind    | Shape             | Mermaid Syntax |
//! |-------------|-------------------|----------------|
//! | Agent       | Rectangle         | `id[label]`    |
//! | Tool        | Subroutine        | `id[[label]]`  |
//! | Router      | Diamond           | `id{label}`    |
//! | SubAgent    | Cylinder          | `id[(label)]`  |
//! | FanOut      | Parallelogram     | `id[/label\]`  |
//! | FanIn       | Reverse Para.     | `id[\label/]`  |
//! | Passthrough | Rounded Rectangle | `id(label)`    |
//! | START/END   | Stadium           | `id([label])`  |

use super::vertex::{VertexId, VertexState};
use crate::workflow::NodeKind;

// ============================================================================
// ID Sanitization
// ============================================================================

/// Sanitize a vertex ID for use as a Mermaid node identifier.
///
/// Mermaid node IDs must be alphanumeric (plus underscores).
/// This function replaces any invalid characters with underscores.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(sanitize_id("my-node"), "my_node");
/// assert_eq!(sanitize_id("node.name"), "node_name");
/// assert_eq!(sanitize_id("valid_name"), "valid_name");
/// ```
pub fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ============================================================================
// Node Rendering
// ============================================================================

/// Render a node with the appropriate Mermaid shape based on its kind.
///
/// Returns a Mermaid node declaration like `id[label]` or `id{label}`.
pub fn render_node(id: &VertexId, kind: Option<&NodeKind>) -> String {
    let safe_id = sanitize_id(id.as_str());
    let label = id.as_str();

    match kind {
        Some(NodeKind::Agent(_)) => format!("    {}[{}]", safe_id, label),
        Some(NodeKind::Tool(_)) => format!("    {}[[{}]]", safe_id, label),
        Some(NodeKind::Router(_)) => format!("    {}{{{}}}", safe_id, label),
        Some(NodeKind::SubAgent(_)) => format!("    {}[({})]", safe_id, label),
        Some(NodeKind::FanOut(_)) => format!("    {}[/{}\\]", safe_id, label),
        Some(NodeKind::FanIn(_)) => format!("    {}[\\{}/]", safe_id, label),
        Some(NodeKind::Passthrough) => format!("    {}({})", safe_id, label),
        None => format!("    {}([{}])", safe_id, label), // Stadium for START/END or unknown
    }
}

/// Render a node with state-based CSS class for coloring.
pub fn render_node_with_state(
    id: &VertexId,
    kind: Option<&NodeKind>,
    state: Option<&VertexState>,
) -> String {
    let base = render_node(id, kind);
    match state {
        Some(VertexState::Active) => format!("{}:::active", base),
        Some(VertexState::Halted) => format!("{}:::halted", base),
        Some(VertexState::Completed) => format!("{}:::completed", base),
        None => base,
    }
}

// ============================================================================
// Edge Rendering
// ============================================================================

/// Render an edge between two vertices.
///
/// - Unconditional edges: solid arrow `-->`
/// - Conditional edges: dotted arrow with label `-. "label" .->`
pub fn render_edge(from: &VertexId, to: &VertexId, condition: Option<&str>) -> String {
    let from_safe = sanitize_id(from.as_str());
    let to_safe = sanitize_id(to.as_str());

    match condition {
        Some(label) => format!("    {} -. \"{}\" .-> {}", from_safe, label, to_safe),
        None => format!("    {} --> {}", from_safe, to_safe),
    }
}

// ============================================================================
// Style Definitions
// ============================================================================

/// CSS class definitions for vertex states.
pub const STYLE_DEFS: &str = r#"
    classDef active fill:#90EE90,stroke:#228B22,stroke-width:2px
    classDef halted fill:#FFE4B5,stroke:#FF8C00,stroke-width:1px
    classDef completed fill:#D3D3D3,stroke:#696969,stroke-width:1px
"#;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("simple"), "simple");
        assert_eq!(sanitize_id("with_underscore"), "with_underscore");
        assert_eq!(sanitize_id("with-dash"), "with_dash");
        assert_eq!(sanitize_id("with.dot"), "with_dot");
        assert_eq!(sanitize_id("with space"), "with_space");
        assert_eq!(sanitize_id("CamelCase123"), "CamelCase123");
        assert_eq!(sanitize_id("a/b/c"), "a_b_c");
    }

    #[test]
    fn test_render_edge_unconditional() {
        let from = VertexId::new("agent");
        let to = VertexId::new("tool");
        assert_eq!(render_edge(&from, &to, None), "    agent --> tool");
    }

    #[test]
    fn test_render_edge_conditional() {
        let from = VertexId::new("router");
        let to = VertexId::new("approved");
        assert_eq!(
            render_edge(&from, &to, Some("yes")),
            "    router -. \"yes\" .-> approved"
        );
    }

    #[test]
    fn test_render_edge_with_special_chars() {
        let from = VertexId::new("my-router");
        let to = VertexId::new("next.step");
        assert_eq!(render_edge(&from, &to, None), "    my_router --> next_step");
    }

    #[test]
    fn test_render_node_agent() {
        let id = VertexId::new("planner");
        let kind = NodeKind::Agent(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    planner[planner]");
    }

    #[test]
    fn test_render_node_tool() {
        let id = VertexId::new("search");
        let kind = NodeKind::Tool(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    search[[search]]");
    }

    #[test]
    fn test_render_node_router() {
        let id = VertexId::new("decision");
        let kind = NodeKind::Router(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    decision{decision}");
    }

    #[test]
    fn test_render_node_subagent() {
        let id = VertexId::new("researcher");
        let kind = NodeKind::SubAgent(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    researcher[(researcher)]");
    }

    #[test]
    fn test_render_node_fanout() {
        let id = VertexId::new("split");
        let kind = NodeKind::FanOut(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    split[/split\\]");
    }

    #[test]
    fn test_render_node_fanin() {
        let id = VertexId::new("merge");
        let kind = NodeKind::FanIn(Default::default());
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    merge[\\merge/]");
    }

    #[test]
    fn test_render_node_passthrough() {
        let id = VertexId::new("forward");
        let kind = NodeKind::Passthrough;
        let result = render_node(&id, Some(&kind));
        assert_eq!(result, "    forward(forward)");
    }

    #[test]
    fn test_render_node_unknown() {
        let id = VertexId::new("start");
        let result = render_node(&id, None);
        assert_eq!(result, "    start([start])");
    }

    #[test]
    fn test_render_node_with_state() {
        let id = VertexId::new("agent");
        let kind = NodeKind::Agent(Default::default());

        let active = render_node_with_state(&id, Some(&kind), Some(&VertexState::Active));
        assert!(active.ends_with(":::active"));

        let halted = render_node_with_state(&id, Some(&kind), Some(&VertexState::Halted));
        assert!(halted.ends_with(":::halted"));

        let completed = render_node_with_state(&id, Some(&kind), Some(&VertexState::Completed));
        assert!(completed.ends_with(":::completed"));
    }
}
