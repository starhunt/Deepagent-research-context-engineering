//! edit_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

use crate::error::MiddlewareError;
use crate::middleware::{StateUpdate, Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;
use crate::state::FileData;

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
    ) -> Result<ToolResult, MiddlewareError> {
        let args: EditFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .edit(&args.file_path, &args.old_string, &args.new_string, args.replace_all)
            .await
            .map_err(MiddlewareError::Backend)?;

        if result.is_ok() {
            let occurrences = result.occurrences.unwrap_or(1);
            let mut tool_result = ToolResult::new(format!(
                "Replaced {} occurrence(s) in {}",
                occurrences,
                args.file_path
            ));
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
    use crate::backends::Backend;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_edit_file_returns_state_update() {
        let tool = EditFileTool;
        let backend = Arc::new(MemoryBackend::new());
        backend.write("/test.txt", "hello world").await.unwrap();
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let args = json!({
            "file_path": "/test.txt",
            "old_string": "world",
            "new_string": "there",
            "replace_all": false
        });

        let result = tool.execute(args, &runtime).await.unwrap();
        assert_eq!(result.updates.len(), 1);

        match &result.updates[0] {
            StateUpdate::UpdateFiles(files) => {
                let file = files.get("/test.txt").and_then(|v| v.as_ref()).unwrap();
                assert_eq!(file.as_string(), "hello there");
            }
            other => panic!("Unexpected update: {:?}", other),
        }
    }
}
