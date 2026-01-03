//! Vertex implementations for workflow nodes
//!
//! Each vertex type implements the Vertex trait and corresponds to a NodeKind variant.
//!
//! # Available Vertices
//!
//! - [`agent::AgentVertex`]: LLM-based agent with tool calling
//! - [`tool::ToolVertex`]: Single tool execution with static/dynamic args
//! - [`router::RouterVertex`]: Conditional routing based on state or LLM decisions
//! - [`subagent::SubAgentVertex`]: Delegates to sub-agents from registry
//! - [`parallel::FanOutVertex`]: Broadcasts messages to multiple targets
//! - [`parallel::FanInVertex`]: Synchronizes messages from multiple sources

pub mod agent;
pub mod parallel;
pub mod router;
pub mod subagent;
pub mod tool;

// Re-export main vertex types
pub use agent::AgentVertex;
pub use parallel::{FanInVertex, FanOutVertex};
pub use router::RouterVertex;
pub use subagent::SubAgentVertex;
pub use tool::ToolVertex;
