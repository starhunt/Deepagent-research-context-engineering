//! Tool implementations for DeepAgents
//!
//! This module provides the 8 core tools auto-injected by middleware:
//! - File operations: read_file, write_file, edit_file, ls, glob, grep
//! - Planning: write_todos
//! - Delegation: task (SubAgent)

mod read_file;
mod write_file;
mod edit_file;
mod ls;
mod glob;
mod grep;
mod write_todos;
mod task;

pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;
pub use edit_file::EditFileTool;
pub use ls::LsTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use write_todos::WriteTodosTool;
pub use task::TaskTool;

use crate::middleware::DynTool;
use std::sync::Arc;

/// 모든 기본 도구 반환 (SubAgent task 제외)
pub fn default_tools() -> Vec<DynTool> {
    vec![
        Arc::new(ReadFileTool),
        Arc::new(WriteFileTool),
        Arc::new(EditFileTool),
        Arc::new(LsTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
        Arc::new(WriteTodosTool),
    ]
}

/// SubAgent task 도구 포함하여 모든 도구 반환
pub fn all_tools() -> Vec<DynTool> {
    let mut tools = default_tools();
    tools.push(Arc::new(TaskTool));
    tools
}
