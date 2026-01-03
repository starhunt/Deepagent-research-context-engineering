//! SubAgent middleware system
//!
//! This module implements the SubAgent delegation pattern from Python DeepAgents,
//! enabling agents to delegate tasks to specialized sub-agents.
//!
//! # Architecture
//!
//! ```text
//! Orchestrator Agent
//!     │
//!     ├─> task("researcher", "Research quantum computing")
//!     │       │
//!     │       └─> TaskTool.execute()
//!     │               │
//!     │               ├─ Check recursion limit
//!     │               ├─ Lookup "researcher" in SubAgentRegistry
//!     │               ├─ Create IsolatedState (filter messages, todos)
//!     │               ├─ Execute subagent with isolated context
//!     │               │
//!     │               └─> Return SubAgentResult
//!     │
//!     └─> Receive ToolMessage with subagent's response
//! ```
//!
//! # Components
//!
//! - [`SubAgentSpec`]: Specification for dynamically-created subagents
//! - [`CompiledSubAgent`]: Pre-compiled subagents ready for execution
//! - [`SubAgentRegistry`]: Registry for looking up subagents by name
//! - [`IsolatedState`]: State isolation for subagent contexts
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_deepagents::middleware::subagent::*;
//!
//! // Create subagent specifications
//! let researcher = SubAgentSpec::builder("researcher")
//!     .description("Conducts web research on topics")
//!     .system_prompt("You are a research agent...")
//!     .build();
//!
//! let synthesizer = SubAgentSpec::builder("synthesizer")
//!     .description("Synthesizes research findings")
//!     .system_prompt("You are a synthesis agent...")
//!     .build();
//!
//! // Create registry
//! let registry = SubAgentRegistry::new()
//!     .with_agent(SubAgentKind::Spec(researcher))
//!     .with_agent(SubAgentKind::Spec(synthesizer));
//!
//! // Use with SubAgentMiddleware (see middleware implementation)
//! ```
//!
//! # State Isolation
//!
//! When a subagent is invoked, it receives isolated state:
//! - **Excluded**: messages, todos, structured_response
//! - **Included**: files (shared filesystem context)
//!
//! This prevents context contamination while allowing file sharing.
//!
//! # Python Reference
//!
//! This implementation is based on:
//! - `deepagents/middleware/subagents.py`
//! - SubAgent and CompiledSubAgent TypedDicts
//! - _EXCLUDED_STATE_KEYS pattern

pub mod spec;
pub mod state_isolation;
pub mod executor;
pub mod task_tool;
pub mod middleware;

// Re-export main types
pub use spec::{
    CompiledSubAgent, CompiledSubAgentExecutor, SubAgentKind, SubAgentRegistry, SubAgentResult,
    SubAgentSpec, SubAgentSpecBuilder,
};
pub use state_isolation::{IsolatedState, IsolatedStateBuilder, EXCLUDED_STATE_KEYS};
pub use executor::{
    SubAgentExecutorFactory, SubAgentExecutorConfig, DefaultSubAgentExecutorFactory,
};
pub use task_tool::{TaskTool, TaskArgs};
pub use middleware::{SubAgentMiddleware, SubAgentMiddlewareConfig, SubAgentMiddlewareBuilder};

/// System prompt addition for task tool usage
///
/// This prompt explains when and how to use the task tool
/// for delegating work to subagents.
pub const TASK_SYSTEM_PROMPT: &str = r#"
## Task Delegation

You have access to the `task` tool to delegate work to specialized sub-agents.
Each sub-agent operates in an isolated context and returns its findings to you.

### When to Use Sub-agents

Use sub-agents when:
- The task requires specialized expertise (research, synthesis, etc.)
- You want to parallelize independent tasks
- The task is complex enough to benefit from focused attention

### Guidelines

1. **Bias toward single sub-agent**: Most tasks should use one sub-agent at a time
2. **Parallelize only when beneficial**: Use multiple simultaneous tasks only for
   clearly independent work (e.g., comparing two topics)
3. **Provide clear descriptions**: Sub-agents only see your task description,
   not the full conversation history

### Example

```
task(
    subagent_type="researcher",
    description="Research the latest developments in quantum computing,
    focusing on error correction techniques and commercial applications."
)
```
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _spec = SubAgentSpec::new("test", "Test agent");
        let _isolated = IsolatedState::new();
        let _registry = SubAgentRegistry::new();
        let _result = SubAgentResult::success("Done");

        // Verify constants are accessible
        assert!(!EXCLUDED_STATE_KEYS.is_empty());
        assert!(!TASK_SYSTEM_PROMPT.is_empty());
    }
}
