//! Think Tool - Explicit reflection for agent reasoning
//!
//! Provides a structured way for agents to record their thinking process.
//! This tool has no side effects - it simply echoes the reflection back,
//! making the agent's reasoning explicit and traceable.
//!
//! # Production Considerations
//!
//! - Output is minimal to reduce prompt pollution
//! - No emojis or decorative formatting
//! - Integrates with ToolRuntime for tracing

use async_trait::async_trait;
use serde::Deserialize;
use tracing::debug;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

/// Think Tool for explicit agent reflection
///
/// This tool allows agents to pause and explicitly record their thinking
/// process. It's useful for:
/// - Making reasoning steps visible in traces
/// - Forcing deliberate analysis before decisions
/// - Improving agent reasoning through explicit reflection
///
/// # Example
/// ```ignore
/// let tool = ThinkTool;
/// let result = tool.execute(json!({
///     "reflection": "I've found 3 relevant sources. Let me analyze their credibility..."
/// }), &runtime).await?;
/// ```
pub struct ThinkTool;

/// Arguments for the think tool
#[derive(Debug, Deserialize)]
struct ThinkArgs {
    /// The reflection or thought to record
    reflection: String,
}

#[async_trait]
impl Tool for ThinkTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "think".to_string(),
            description: "Record your thinking process explicitly. Use this tool to pause and reflect on your reasoning, analyze information, or plan your next steps. The reflection is recorded and returned as confirmation.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "reflection": {
                        "type": "string",
                        "description": "Your thought process, analysis, or reasoning to record",
                        "minLength": 1
                    }
                },
                "required": ["reflection"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        let args: ThinkArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        // Log for tracing
        if let Some(tool_call_id) = runtime.tool_call_id() {
            debug!(
                tool_call_id,
                reflection_len = args.reflection.len(),
                "Think tool executed"
            );
        }

        // Minimal output to avoid prompt pollution
        // The reflection itself is the valuable content - we just acknowledge it
        Ok(ToolResult::new(format!(
            "[Reflection recorded: {} chars]",
            args.reflection.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use std::sync::Arc;

    fn create_test_runtime() -> ToolRuntime {
        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        ToolRuntime::new(state, backend)
    }

    #[test]
    fn test_think_tool_definition() {
        let tool = ThinkTool;
        let def = tool.definition();

        assert_eq!(def.name, "think");
        assert!(def.description.contains("thinking process"));

        // Verify required parameters
        let params = &def.parameters;
        let required = params["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("reflection")));

        // Verify additionalProperties is false
        assert_eq!(params["additionalProperties"], serde_json::json!(false));

        // Verify minLength constraint
        assert_eq!(params["properties"]["reflection"]["minLength"], 1);
    }

    #[tokio::test]
    async fn test_think_tool_execute() {
        let tool = ThinkTool;
        let runtime = create_test_runtime();

        let reflection = "I need to search for more sources on this topic.";
        let result = tool
            .execute(
                serde_json::json!({
                    "reflection": reflection
                }),
                &runtime,
            )
            .await
            .unwrap();

        // Should contain char count, not the full reflection (minimal output)
        assert!(result.message.contains("Reflection recorded"));
        assert!(result.message.contains(&format!("{} chars", reflection.len())));
    }

    #[tokio::test]
    async fn test_think_tool_no_emoji() {
        let tool = ThinkTool;
        let runtime = create_test_runtime();

        let result = tool
            .execute(
                serde_json::json!({
                    "reflection": "Test reflection"
                }),
                &runtime,
            )
            .await
            .unwrap();

        // Should not contain any emoji
        assert!(!result.message.contains("📝"));
        assert!(!result.message.contains("*"));
        assert!(!result.message.contains("---"));
    }

    #[tokio::test]
    async fn test_think_tool_empty_reflection() {
        let tool = ThinkTool;
        let runtime = create_test_runtime();

        // Empty reflection should still work (schema validation is LLM's job)
        let result = tool
            .execute(serde_json::json!({"reflection": ""}), &runtime)
            .await
            .unwrap();

        assert!(result.message.contains("0 chars"));
    }

    #[tokio::test]
    async fn test_think_tool_long_reflection() {
        let tool = ThinkTool;
        let runtime = create_test_runtime();

        let long_thought = "x".repeat(1000);
        let result = tool
            .execute(serde_json::json!({"reflection": long_thought}), &runtime)
            .await
            .unwrap();

        // Output should be minimal - just the char count
        assert!(result.message.contains("1000 chars"));
        assert!(!result.message.contains(&long_thought)); // Should NOT echo the full content
    }

    #[tokio::test]
    async fn test_think_tool_missing_reflection() {
        let tool = ThinkTool;
        let runtime = create_test_runtime();

        let result = tool.execute(serde_json::json!({}), &runtime).await;

        assert!(result.is_err());
    }
}
