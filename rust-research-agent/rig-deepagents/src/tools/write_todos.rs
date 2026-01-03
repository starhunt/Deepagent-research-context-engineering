//! write_todos 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{StateUpdate, Tool, ToolDefinition, ToolResult};
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
    ) -> Result<ToolResult, MiddlewareError> {
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

        Ok(
            ToolResult::new(format!("Updated {} todo items", todos.len()))
                .with_update(StateUpdate::SetTodos(todos)),
        )
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
    async fn test_write_todos_returns_state_update() {
        let tool = WriteTodosTool;
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        let args = json!({
            "todos": [
                {"content": "First task", "status": "in_progress"},
                {"content": "Second task"}
            ]
        });

        let result = tool.execute(args, &runtime).await.unwrap();
        assert_eq!(result.updates.len(), 1);

        match &result.updates[0] {
            StateUpdate::SetTodos(todos) => {
                assert_eq!(todos.len(), 2);
                assert_eq!(todos[0].content, "First task");
                assert_eq!(todos[0].status, TodoStatus::InProgress);
                assert_eq!(todos[1].content, "Second task");
                assert_eq!(todos[1].status, TodoStatus::Pending);
            }
            other => panic!("Unexpected update: {:?}", other),
        }
    }
}
