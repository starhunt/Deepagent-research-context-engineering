//! 미들웨어 모듈
//!
//! AgentMiddleware 트레이트와 구현체들을 제공합니다.
//!
//! # Available Middleware
//!
//! - [`subagent`]: SubAgent delegation middleware (task tool, registry, execution)
//! - [`summarization`]: Token budget management and context summarization
//! - [`patch_tool_calls`]: Fix dangling tool calls in message history
//! - [`human_in_the_loop`]: Interrupt execution for human approval

pub mod traits;
pub mod stack;
pub mod filesystem;
pub mod todo_list;
pub mod subagent;
pub mod summarization;
pub mod patch_tool_calls;
pub mod human_in_the_loop;

// Core traits and types
pub use traits::{AgentMiddleware, DynTool, Tool, ToolDefinition, ToolRegistry, ToolResult, StateUpdate};
pub use stack::MiddlewareStack;
pub use filesystem::{FilesystemMiddleware, FILESYSTEM_SYSTEM_PROMPT};
pub use todo_list::{TodoListMiddleware, TODO_SYSTEM_PROMPT};

// Model hook types (Python Parity - NEW)
pub use traits::{
    ModelRequest, ModelResponse, ModelControl,
    InterruptRequest, ActionRequest, ReviewConfig, Decision,
};

// Summarization middleware
pub use summarization::{
    SummarizationMiddleware, SummarizationConfig, SummarizationConfigBuilder,
    TriggerCondition, KeepSize,
    count_tokens_approximately, get_chars_per_token, TokenCounterConfig,
    DEFAULT_CHARS_PER_TOKEN, CLAUDE_CHARS_PER_TOKEN, DEFAULT_SUMMARY_PROMPT,
};

// SubAgent types
pub use subagent::{
    SubAgentSpec, SubAgentSpecBuilder, CompiledSubAgent, SubAgentKind,
    SubAgentRegistry, SubAgentResult, IsolatedState, IsolatedStateBuilder,
    EXCLUDED_STATE_KEYS, TASK_SYSTEM_PROMPT,
    // Executor types
    SubAgentExecutorFactory, SubAgentExecutorConfig, DefaultSubAgentExecutorFactory,
    // Task tool
    TaskTool, TaskArgs,
    // Middleware
    SubAgentMiddleware, SubAgentMiddlewareConfig, SubAgentMiddlewareBuilder,
};

// PatchToolCalls middleware (Python Parity - NEW)
pub use patch_tool_calls::PatchToolCallsMiddleware;

// HumanInTheLoop middleware (Python Parity - NEW)
pub use human_in_the_loop::{HumanInTheLoopMiddleware, InterruptOnConfig};
