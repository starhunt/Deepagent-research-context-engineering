//! 미들웨어 모듈
//!
//! AgentMiddleware 트레이트와 구현체들을 제공합니다.

pub mod traits;
pub mod stack;

// Phase 5에서 구현될 미들웨어들
// pub mod todo;
// pub mod filesystem;
// pub mod patch_tool_calls;
// pub mod summarization;
// pub mod subagent;

pub use traits::{AgentMiddleware, DynTool, Tool, ToolDefinition, StateUpdate};
pub use stack::MiddlewareStack;
