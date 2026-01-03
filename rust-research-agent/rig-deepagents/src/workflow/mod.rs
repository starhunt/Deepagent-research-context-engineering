//! Workflow Graph System for Pregel-Based Agent Orchestration
//!
//! This module provides the building blocks for constructing and executing
//! agent workflows using a Pregel-inspired graph execution model.
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     WorkflowGraph                            │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │                    Nodes                             │    │
//! │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
//! │  │  │  Agent  │→ │ Router  │→ │ SubAgent│              │    │
//! │  │  └─────────┘  └────┬────┘  └─────────┘              │    │
//! │  │                    │                                 │    │
//! │  │              ┌─────▼─────┐                           │    │
//! │  │              │   Tool    │                           │    │
//! │  │              └───────────┘                           │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                                                              │
//! │  Compile via WorkflowBuilder → CompiledWorkflow              │
//! │  Execute via PregelRuntime                                   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::workflow::{WorkflowGraph, NodeKind, AgentNodeConfig};
//!
//! let workflow = WorkflowGraph::<MyState>::new()
//!     .name("research_agent")
//!     .node("planner", NodeKind::Agent(AgentNodeConfig {
//!         system_prompt: "Plan the research...".into(),
//!         ..Default::default()
//!     }))
//!     .node("researcher", NodeKind::Agent(AgentNodeConfig {
//!         system_prompt: "Execute research...".into(),
//!         ..Default::default()
//!     }))
//!     .entry("planner")
//!     .edge("planner", "researcher")
//!     .edge("researcher", END)
//!     .build()?;
//! ```

pub mod compiled;
pub mod graph;
pub mod node;
pub mod vertices;

pub use node::{
    AgentNodeConfig, Branch, BranchCondition, FanInNodeConfig, FanOutNodeConfig, MergeStrategy,
    NodeKind, RouterNodeConfig, RoutingStrategy, SplitStrategy, StopCondition, SubAgentNodeConfig,
    ToolNodeConfig,
};
pub use graph::{BuiltWorkflowGraph, GraphEdge, GraphNode, WorkflowBuildError, WorkflowGraph, END};
pub use compiled::{CompiledWorkflow, PassthroughVertex, WorkflowCompileError};

pub use vertices::agent::AgentVertex;
