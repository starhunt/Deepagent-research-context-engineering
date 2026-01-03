//! write_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

use crate::error::MiddlewareError;
use crate::middleware::{StateUpdate, Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;
use crate::state::FileData;

/// write_file 도구
pub struct WriteFileTool;

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    file_path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file, creating it if it doesn't exist.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        let args: WriteFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .write(&args.file_path, &args.content)
            .await
            .map_err(MiddlewareError::Backend)?;

        if result.is_ok() {
            let mut tool_result =
                ToolResult::new(format!("Successfully wrote to {}", args.file_path));
            if let Some(files_update) = result.files_update {
                let updates: HashMap<String, Option<FileData>> = files_update
                    .into_iter()
                    .map(|(path, data)| (path, Some(data)))
                    .collect();
                tool_result = tool_result.with_update(StateUpdate::UpdateFiles(updates));
            }
            Ok(tool_result)
        } else {
            Err(MiddlewareError::ToolExecution(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_write_file_returns_state_update() {
        let tool = WriteFileTool;
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let args = json!({
            "file_path": "/test.txt",
            "content": "hello"
        });

        let result = tool.execute(args, &runtime).await.unwrap();
        assert_eq!(result.updates.len(), 1);

        match &result.updates[0] {
            StateUpdate::UpdateFiles(files) => {
                let file = files.get("/test.txt").and_then(|v| v.as_ref()).unwrap();
                assert_eq!(file.as_string(), "hello");
            }
            other => panic!("Unexpected update: {:?}", other),
        }
    }
}
