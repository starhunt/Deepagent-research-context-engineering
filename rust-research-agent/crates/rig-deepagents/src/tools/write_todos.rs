//! write_todos 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;
use crate::state::{Todo, TodoStatus};

/// write_todos 도구
pub struct WriteTodosTool;

#[derive(Debug, Deserialize)]
struct TodoItem {
    content: String,
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct WriteTodosArgs {
    todos: Vec<TodoItem>,
}

#[async_trait]
impl Tool for WriteTodosTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_todos".to_string(),
            description: "Update the todo list with new items.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The todo item content"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "default": "pending"
                                }
                            },
                            "required": ["content"]
                        },
                        "description": "List of todo items"
                    }
                },
                "required": ["todos"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: WriteTodosArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let todos: Vec<Todo> = args.todos.iter()
            .map(|t| {
                let status = match t.status.as_str() {
                    "in_progress" => TodoStatus::InProgress,
                    "completed" => TodoStatus::Completed,
                    _ => TodoStatus::Pending,
                };
                Todo::with_status(&t.content, status)
            })
            .collect();

        // Note: 실제 상태 업데이트는 미들웨어 레벨에서 처리
        // 여기서는 검증 및 포맷만 수행
        Ok(format!("Updated {} todo items", todos.len()))
    }
}
