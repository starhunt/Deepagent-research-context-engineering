//! write_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

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
    ) -> Result<String, MiddlewareError> {
        let args: WriteFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .write(&args.file_path, &args.content)
            .await
            .map_err(MiddlewareError::Backend)?;

        if result.is_ok() {
            Ok(format!("Successfully wrote to {}", args.file_path))
        } else {
            Err(MiddlewareError::ToolExecution(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}
