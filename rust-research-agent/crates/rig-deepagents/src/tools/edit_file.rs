//! edit_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// edit_file 도구
pub struct EditFileTool;

#[derive(Debug, Deserialize)]
struct EditFileArgs {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Edit a file by replacing old_string with new_string.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)",
                        "default": false
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: EditFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .edit(&args.file_path, &args.old_string, &args.new_string, args.replace_all)
            .await
            .map_err(MiddlewareError::Backend)?;

        if result.is_ok() {
            let occurrences = result.occurrences.unwrap_or(1);
            Ok(format!("Replaced {} occurrence(s) in {}", occurrences, args.file_path))
        } else {
            Err(MiddlewareError::ToolExecution(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}
