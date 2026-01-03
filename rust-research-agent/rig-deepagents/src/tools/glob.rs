//! glob 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

/// glob 도구
pub struct GlobTool;

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    #[serde(default = "default_path")]
    base_path: String,
}

fn default_path() -> String {
    "/".to_string()
}

#[async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.rs', '*.txt')"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base path to search from",
                        "default": "/"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        let args: GlobArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let files = runtime.backend()
            .glob(&args.pattern, &args.base_path)
            .await
            .map_err(MiddlewareError::Backend)?;

        let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();

        if paths.is_empty() {
            Ok(ToolResult::new("No files found matching pattern."))
        } else {
            Ok(ToolResult::new(format!(
                "Found {} files:\n{}",
                paths.len(),
                paths.join("\n")
            )))
        }
    }
}
