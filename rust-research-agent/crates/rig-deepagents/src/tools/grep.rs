//! grep 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// grep 도구
pub struct GrepTool;

#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob_filter: Option<String>,
}

#[async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: "Search for a literal text pattern in files.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Literal text pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: /)"
                    },
                    "glob_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '**/*.rs')"
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
    ) -> Result<String, MiddlewareError> {
        let args: GrepArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let matches = runtime.backend()
            .grep(&args.pattern, args.path.as_deref(), args.glob_filter.as_deref())
            .await
            .map_err(MiddlewareError::Backend)?;

        if matches.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            let output: Vec<String> = matches.iter()
                .map(|m| format!("{}:{}: {}", m.path, m.line, m.text))
                .collect();
            Ok(format!("Found {} matches:\n{}", matches.len(), output.join("\n")))
        }
    }
}
