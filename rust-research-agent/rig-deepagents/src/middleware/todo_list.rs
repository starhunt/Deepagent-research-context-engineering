//! TodoListMiddleware - injects write_todos and planning guidance.
//!
//! Python Reference: langchain.agents.middleware.todo (TodoListMiddleware)

use std::sync::Arc;

use async_trait::async_trait;

use crate::middleware::{AgentMiddleware, DynTool};
use crate::tools::{ReadTodosTool, WriteTodosTool};

/// Default system prompt for todo planning.
pub const TODO_SYSTEM_PROMPT: &str = "## Planning with `write_todos`\n\
Use `write_todos` for multi-step tasks (3+ steps).\n\
Each todo item has `content` and `status` (pending, in_progress, completed).\n\
Update the list as you work: mark items in_progress before starting and completed immediately after finishing.";

/// Middleware that injects the write_todos tool and planning guidance.
pub struct TodoListMiddleware {
    tools: Vec<DynTool>,
    system_prompt: String,
}

impl TodoListMiddleware {
    /// Create a TodoListMiddleware with default prompt.
    pub fn new() -> Self {
        Self::with_system_prompt(TODO_SYSTEM_PROMPT)
    }

    /// Create a TodoListMiddleware with a custom system prompt.
    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        Self {
            tools: vec![Arc::new(ReadTodosTool), Arc::new(WriteTodosTool)],
            system_prompt: prompt.into(),
        }
    }
}

#[async_trait]
impl AgentMiddleware for TodoListMiddleware {
    fn name(&self) -> &str {
        "todo_list"
    }

    fn tools(&self) -> Vec<DynTool> {
        self.tools.clone()
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        if self.system_prompt.is_empty() {
            prompt
        } else {
            format!("{}\n\n{}", prompt, self.system_prompt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_list_injects_tool() {
        let middleware = TodoListMiddleware::new();
        let tools = middleware.tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].definition().name, "read_todos");
        assert_eq!(tools[1].definition().name, "write_todos");
    }

    #[test]
    fn test_todo_list_prompt_append() {
        let middleware = TodoListMiddleware::new();
        let prompt = middleware.modify_system_prompt("Base prompt".to_string());
        assert!(prompt.contains("Base prompt"));
        assert!(prompt.contains("write_todos"));
    }
}
