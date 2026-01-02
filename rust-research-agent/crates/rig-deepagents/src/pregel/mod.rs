//! Pregel Runtime for Graph-Based Agent Orchestration
//!
//! This module implements a Pregel-inspired runtime for executing agent workflows.
//! Key concepts:
//!
//! - **Vertex**: Computation unit (Agent, Tool, Router, etc.)
//! - **Edge**: Connection between vertices (Direct, Conditional)
//! - **Superstep**: Synchronized execution phase
//! - **Message**: Communication between vertices
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    PregelRuntime                             │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
//! │  │Superstep│→ │Superstep│→ │Superstep│→ ...                │
//! │  │    0    │  │    1    │  │    2    │                     │
//! │  └─────────┘  └─────────┘  └─────────┘                     │
//! │       │            │            │                           │
//! │       ▼            ▼            ▼                           │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │ Per-Superstep: Deliver → Compute → Collect → Route  │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Execution Modes
//!
//! The runtime supports two execution modes:
//!
//! - `ExecutionMode::MessageBased` (default): All vertices start Active.
//!   Vertices must explicitly send messages. Backward compatible.
//!
//! - `ExecutionMode::EdgeDriven`: Only entry vertex starts Active.
//!   When vertices halt, activation messages are sent to edge targets.
//!   Matches LangGraph's execution model.

pub mod vertex;
pub mod message;
pub mod config;
pub mod error;
pub mod state;
pub mod runtime;
pub mod checkpoint;
pub mod visualization;

// Re-exports
pub use vertex::{
    BoxedVertex, ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId, VertexState,
};
pub use message::{Priority, Source, VertexMessage, WorkflowMessage};
pub use config::{ExecutionMode, PregelConfig, RetryPolicy};
pub use error::PregelError;
pub use state::{UnitState, UnitUpdate, WorkflowState};
pub use runtime::{PregelRuntime, WorkflowResult};
pub use checkpoint::{Checkpoint, Checkpointer, CheckpointerConfig, MemoryCheckpointer, FileCheckpointer, create_checkpointer};
pub use visualization::{sanitize_id, render_node, render_node_with_state, render_edge};
