//! Vertex implementations for workflow nodes
//!
//! Each vertex type implements the Vertex trait and corresponds to a NodeKind variant.
//!
//! # Available Vertices
//!
//! - [`agent::AgentVertex`]: LLM-based agent with tool calling
//! - [`subagent::SubAgentVertex`]: Delegates to sub-agents from registry

pub mod agent;
pub mod subagent;

// Future vertex implementations:
// pub mod tool;
// pub mod router;
// pub mod parallel;
