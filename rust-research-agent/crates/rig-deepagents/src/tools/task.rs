//! task 도구 구현 (SubAgent 위임)

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// task 도구 - SubAgent에 작업 위임
pub struct TaskTool;

#[derive(Debug, Deserialize)]
struct TaskArgs {
    subagent_type: String,
    prompt: String,
    #[serde(default)]
    description: Option<String>,
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task".to_string(),
            description: "Delegate a task to a sub-agent for specialized processing.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of sub-agent to use (e.g., 'researcher', 'explorer', 'synthesizer')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the sub-agent"
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the task"
                    }
                },
                "required": ["subagent_type", "prompt"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: TaskArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        // 재귀 한도 확인
        if runtime.is_recursion_limit_exceeded() {
            return Err(MiddlewareError::RecursionLimit(
                format!("Recursion limit exceeded. Cannot delegate to subagent '{}'", args.subagent_type)
            ));
        }

        // Note: 실제 SubAgent 실행은 executor에서 처리
        // 이 도구는 요청을 구조화하고 검증만 수행
        Ok(format!(
            "Task delegation requested:\n- Agent: {}\n- Description: {}\n- Prompt: {}",
            args.subagent_type,
            args.description.unwrap_or_else(|| "N/A".to_string()),
            args.prompt
        ))
    }
}
