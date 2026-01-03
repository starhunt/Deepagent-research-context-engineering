//! ls 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

/// ls 도구
pub struct LsTool;

#[derive(Debug, Deserialize)]
struct LsArgs {
    #[serde(default = "default_path")]
    path: String,
}

fn default_path() -> String {
    "/".to_string()
}

#[async_trait]
impl Tool for LsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ls".to_string(),
            description: "List files and directories at the given path.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list",
                        "default": "/"
                    }
                }
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        let args: LsArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let files = runtime.backend()
            .ls(&args.path)
            .await
            .map_err(MiddlewareError::Backend)?;

        let output: Vec<String> = files.iter()
            .map(|f| {
                if f.is_dir {
                    format!("{}/ (dir)", f.path)
                } else {
                    format!("{} ({} bytes)", f.path, f.size.unwrap_or(0))
                }
            })
            .collect();

        if output.is_empty() {
            Ok(ToolResult::new("Directory is empty."))
        } else {
            Ok(ToolResult::new(output.join("\n")))
        }
    }
}
