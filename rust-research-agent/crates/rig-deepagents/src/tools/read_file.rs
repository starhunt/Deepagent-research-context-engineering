//! read_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// read_file 도구
pub struct ReadFileTool;

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    file_path: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    2000
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read content from a file with optional line offset and limit.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed)",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                        "default": 2000
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: ReadFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        runtime.backend()
            .read(&args.file_path, args.offset, args.limit)
            .await
            .map_err(MiddlewareError::Backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::backends::Backend;
    use crate::state::AgentState;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_read_file_tool() {
        let backend = Arc::new(MemoryBackend::new());
        backend.write("/test.txt", "line1\nline2\nline3").await.unwrap();

        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);
        let tool = ReadFileTool;

        let result = tool.execute(
            serde_json::json!({"file_path": "/test.txt"}),
            &runtime,
        ).await.unwrap();

        assert!(result.contains("line1"));
        assert!(result.contains("line2"));
    }
}
