//! Adapter for using Rig tools in rig-deepagents
//!
//! This module provides `RigToolAdapter` which wraps any Rig `Tool` implementation
//! to work with rig-deepagents' `Tool` trait.
//!
//! # Key Differences Bridged
//!
//! | Aspect | Rig Tool | rig-deepagents Tool |
//! |--------|----------|---------------------|
//! | Args | Typed `Self::Args` | `serde_json::Value` |
//! | Output | Typed `Self::Output` | `String` |
//! | Definition | Async with prompt | Sync |
//! | Runtime | None | `ToolRuntime` (ignored) |
//!
//! # Example
//!
//! ```rust,ignore
//! use rig::tools::think::ThinkTool;
//! use rig_deepagents::compat::RigToolAdapter;
//!
//! let adapter = RigToolAdapter::new(ThinkTool).await;
//! let result = adapter.execute(
//!     serde_json::json!({"thought": "Analyzing..."}),
//!     &runtime,
//! ).await?;
//! ```

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// Adapter that wraps a Rig `Tool` to implement rig-deepagents `Tool` trait.
///
/// This enables using Rig's built-in tools (like `ThinkTool`) and any custom
/// Rig tools within the rig-deepagents middleware system.
///
/// # Type Parameters
///
/// - `T`: The Rig tool type implementing `rig::tool::Tool`
///
/// # Notes
///
/// - The `ToolRuntime` parameter is ignored since Rig tools don't use runtime context
/// - Tool definition is cached at construction time for efficiency
/// - Errors are converted to `MiddlewareError::ToolExecution`
pub struct RigToolAdapter<T>
where
    T: rig::tool::Tool + Send + Sync,
{
    /// The wrapped Rig tool
    inner: T,
    /// Cached tool definition (computed once at construction)
    cached_definition: ToolDefinition,
    /// Phantom data to satisfy variance requirements
    _phantom: PhantomData<T>,
}

impl<T> RigToolAdapter<T>
where
    T: rig::tool::Tool + Send + Sync,
    T::Args: DeserializeOwned + Send + Sync,
    T::Output: Serialize + Send,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    /// Create a new adapter wrapping a Rig tool.
    ///
    /// This is an async constructor because it needs to call the tool's
    /// `definition()` method which is async in Rig.
    ///
    /// # Arguments
    ///
    /// * `tool` - The Rig tool to wrap
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rig::tools::think::ThinkTool;
    ///
    /// let adapter = RigToolAdapter::new(ThinkTool).await;
    /// ```
    pub async fn new(tool: T) -> Self {
        // Get the Rig tool definition (async)
        let rig_def = tool.definition(String::new()).await;

        // Convert to rig-deepagents ToolDefinition
        let cached_definition = ToolDefinition {
            name: rig_def.name,
            description: rig_def.description,
            parameters: rig_def.parameters,
        };

        Self {
            inner: tool,
            cached_definition,
            _phantom: PhantomData,
        }
    }

    /// Create adapter with a custom prompt for definition generation.
    ///
    /// Some Rig tools may tailor their definition based on the prompt context.
    ///
    /// # Arguments
    ///
    /// * `tool` - The Rig tool to wrap
    /// * `prompt` - Context prompt for definition generation
    pub async fn with_prompt(tool: T, prompt: impl Into<String>) -> Self {
        let rig_def = tool.definition(prompt.into()).await;

        let cached_definition = ToolDefinition {
            name: rig_def.name,
            description: rig_def.description,
            parameters: rig_def.parameters,
        };

        Self {
            inner: tool,
            cached_definition,
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the inner Rig tool.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Consume the adapter and return the inner Rig tool.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

#[async_trait]
impl<T> Tool for RigToolAdapter<T>
where
    T: rig::tool::Tool + Send + Sync + 'static,
    T::Args: DeserializeOwned + Send + Sync,
    T::Output: Serialize + Send,
    T::Error: std::error::Error + Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.cached_definition.clone()
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime, // Rig tools don't use runtime
    ) -> Result<String, MiddlewareError> {
        // Step 1: Deserialize JSON args to the tool's typed Args
        let typed_args: T::Args = serde_json::from_value(args).map_err(|e| {
            MiddlewareError::ToolExecution(format!(
                "Failed to deserialize args for tool '{}': {}",
                T::NAME,
                e
            ))
        })?;

        // Step 2: Call the Rig tool
        let result = self.inner.call(typed_args).await.map_err(|e| {
            MiddlewareError::ToolExecution(format!("Tool '{}' execution failed: {}", T::NAME, e))
        })?;

        // Step 3: Serialize the output to JSON string
        serde_json::to_string(&result).map_err(|e| {
            MiddlewareError::ToolExecution(format!(
                "Failed to serialize output from tool '{}': {}",
                T::NAME,
                e
            ))
        })
    }
}

impl<T> Debug for RigToolAdapter<T>
where
    T: rig::tool::Tool + Send + Sync + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RigToolAdapter")
            .field("tool_name", &T::NAME)
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    // =========================================================================
    // Test Tool Implementation (mimics Rig's Tool trait)
    // =========================================================================

    #[derive(Debug, Deserialize)]
    struct AddArgs {
        x: i32,
        y: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error: {0}")]
    struct MathError(String);

    #[derive(Debug, Clone, Deserialize, Serialize)]
    struct Adder;

    impl rig::tool::Tool for Adder {
        const NAME: &'static str = "add";

        type Error = MathError;
        type Args = AddArgs;
        type Output = i32;

        async fn definition(
            &self,
            _prompt: String,
        ) -> rig::completion::ToolDefinition {
            rig::completion::ToolDefinition {
                name: "add".to_string(),
                description: "Add two numbers together".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "First number"},
                        "y": {"type": "integer", "description": "Second number"}
                    },
                    "required": ["x", "y"]
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.x + args.y)
        }
    }

    // =========================================================================
    // Test Helper
    // =========================================================================

    fn create_test_runtime() -> ToolRuntime {
        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        ToolRuntime::new(state, backend)
    }

    // =========================================================================
    // Tests
    // =========================================================================

    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = RigToolAdapter::new(Adder).await;

        let def = adapter.definition();
        assert_eq!(def.name, "add");
        assert!(def.description.contains("Add"));
    }

    #[tokio::test]
    async fn test_adapter_execute_success() {
        let adapter = RigToolAdapter::new(Adder).await;
        let runtime = create_test_runtime();

        let result = adapter
            .execute(serde_json::json!({"x": 5, "y": 3}), &runtime)
            .await
            .unwrap();

        // Output should be JSON-serialized integer
        assert_eq!(result, "8");
    }

    #[tokio::test]
    async fn test_adapter_execute_invalid_args() {
        let adapter = RigToolAdapter::new(Adder).await;
        let runtime = create_test_runtime();

        // Missing required field "y"
        let result = adapter
            .execute(serde_json::json!({"x": 5}), &runtime)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("deserialize"));
    }

    #[tokio::test]
    async fn test_adapter_with_prompt() {
        let adapter = RigToolAdapter::with_prompt(Adder, "Custom context").await;

        let def = adapter.definition();
        assert_eq!(def.name, "add");
    }

    #[tokio::test]
    async fn test_adapter_inner_access() {
        let adapter = RigToolAdapter::new(Adder).await;

        // Test inner() reference
        let _ = adapter.inner();

        // Test into_inner() consumption
        let _tool = adapter.into_inner();
    }

    #[tokio::test]
    async fn test_adapter_debug() {
        let adapter = RigToolAdapter::new(Adder).await;

        let debug_str = format!("{:?}", adapter);
        assert!(debug_str.contains("RigToolAdapter"));
        assert!(debug_str.contains("add"));
    }

    // =========================================================================
    // Test with Rig's Built-in ThinkTool
    // =========================================================================

    #[tokio::test]
    async fn test_with_rig_think_tool() {
        use rig::tools::think::ThinkTool;

        let adapter = RigToolAdapter::new(ThinkTool).await;
        let runtime = create_test_runtime();

        let def = adapter.definition();
        assert_eq!(def.name, "think");
        assert!(def.description.contains("think"));

        // Execute ThinkTool
        let result = adapter
            .execute(
                serde_json::json!({"thought": "I need to analyze this carefully."}),
                &runtime,
            )
            .await
            .unwrap();

        // ThinkTool echoes the thought back
        assert!(result.contains("analyze"));
    }
}
