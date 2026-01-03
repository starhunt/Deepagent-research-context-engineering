//! Tool implementations for DeepAgents
//!
//! This module provides tools for DeepAgents workflows:
//!
//! ## Core Tools (auto-injected by middleware)
//! - File operations: read_file, write_file, edit_file, ls, glob, grep
//! - Planning: write_todos
//! - Delegation: task (SubAgent)
//!
//! ## Domain Tools (optional, require configuration)
//! - Research: tavily_search (requires TAVILY_API_KEY)
//! - Reflection: think (explicit reasoning tool)

mod read_file;
mod write_file;
mod edit_file;
mod ls;
mod glob;
mod grep;
mod write_todos;
mod task;

// Domain tools
mod tavily;
mod think;

pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;
pub use edit_file::EditFileTool;
pub use ls::LsTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use write_todos::WriteTodosTool;
pub use task::TaskTool;

// Domain tool exports
pub use tavily::{TavilySearchTool, TavilyError, SearchDepth, Topic};
pub use think::ThinkTool;

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

/// Research tools (ThinkTool only - TavilySearchTool requires API key)
///
/// Use `research_tools_with_tavily` for full research capabilities.
pub fn research_tools() -> Vec<DynTool> {
    vec![Arc::new(ThinkTool)]
}

/// Research tools including Tavily search
///
/// # Arguments
/// * `tavily_api_key` - API key for Tavily Search
///
/// # Example
/// ```ignore
/// let tools = research_tools_with_tavily("your-api-key");
/// ```
pub fn research_tools_with_tavily(tavily_api_key: impl Into<String>) -> Vec<DynTool> {
    vec![
        Arc::new(TavilySearchTool::new(tavily_api_key)),
        Arc::new(ThinkTool),
    ]
}
