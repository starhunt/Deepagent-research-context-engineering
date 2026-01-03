//! FilesystemMiddleware - injects filesystem tools and usage guidance.
//!
//! Python Reference: deepagents.middleware.filesystem.FilesystemMiddleware

use std::sync::Arc;

use async_trait::async_trait;

use crate::middleware::{AgentMiddleware, DynTool};
use crate::tools::{EditFileTool, GlobTool, GrepTool, LsTool, ReadFileTool, WriteFileTool};

/// Default system prompt for filesystem tools.
pub const FILESYSTEM_SYSTEM_PROMPT: &str = "## Filesystem tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`\n\
You can access a filesystem with these tools. All file paths must start with `/`.\n\
- ls: list directory contents (absolute path required)\n\
- read_file: read file contents with optional pagination (offset/limit)\n\
- write_file: create a new file (avoid overwriting existing files)\n\
- edit_file: exact string replacement (read the file first)\n\
- glob: find files by pattern (e.g., \"**/*.rs\")\n\
- grep: literal text search within files";

/// Middleware that injects filesystem tools and prompt guidance.
pub struct FilesystemMiddleware {
    tools: Vec<DynTool>,
    system_prompt: String,
}

impl FilesystemMiddleware {
    /// Create a FilesystemMiddleware with default prompt.
    pub fn new() -> Self {
        Self::with_system_prompt(FILESYSTEM_SYSTEM_PROMPT)
    }

    /// Create a FilesystemMiddleware with a custom system prompt.
    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        Self {
            tools: vec![
                Arc::new(LsTool),
                Arc::new(ReadFileTool),
                Arc::new(WriteFileTool),
                Arc::new(EditFileTool),
                Arc::new(GlobTool),
                Arc::new(GrepTool),
            ],
            system_prompt: prompt.into(),
        }
    }
}

#[async_trait]
impl AgentMiddleware for FilesystemMiddleware {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn tools(&self) -> Vec<DynTool> {
        self.tools.clone()
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        if self.system_prompt.is_empty() {
            prompt
        } else {
            format!("{}\n\n{}", prompt, self.system_prompt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filesystem_injects_tools() {
        let middleware = FilesystemMiddleware::new();
        let names: std::collections::HashSet<_> = middleware
            .tools()
            .iter()
            .map(|tool| tool.definition().name)
            .collect();

        let expected = [
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        ];

        for name in expected {
            assert!(names.contains(name));
        }
    }

    #[test]
    fn test_filesystem_prompt_append() {
        let middleware = FilesystemMiddleware::new();
        let prompt = middleware.modify_system_prompt("Base prompt".to_string());
        assert!(prompt.contains("Base prompt"));
        assert!(prompt.contains("read_file"));
    }
}
