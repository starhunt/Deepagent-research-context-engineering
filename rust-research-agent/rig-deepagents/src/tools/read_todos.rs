use async_trait::async_trait;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

pub struct ReadTodosTool;

#[async_trait]
impl Tool for ReadTodosTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_todos".to_string(),
            description: "Read the current todo list state.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    async fn execute(
        &self,
        _args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        let todos = &runtime.state().todos;
        let json = serde_json::to_string(todos)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Failed to serialize todos: {e}")))?;
        Ok(ToolResult::new(json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::{AgentState, Todo, TodoStatus};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_read_todos_returns_state() {
        let tool = ReadTodosTool;
        let backend = Arc::new(MemoryBackend::new());
        let mut state = AgentState::new();
        state.todos = vec![
            Todo::with_status("First", TodoStatus::Pending),
            Todo::with_status("Second", TodoStatus::Completed),
        ];
        let runtime = ToolRuntime::new(state, backend);

        let result = tool.execute(serde_json::json!({}), &runtime).await.unwrap();
        let todos: Vec<Todo> = serde_json::from_str(&result.message).unwrap();

        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0].content, "First");
        assert_eq!(todos[0].status, TodoStatus::Pending);
        assert_eq!(todos[1].content, "Second");
        assert_eq!(todos[1].status, TodoStatus::Completed);
    }
}
