//! 미들웨어 모듈
//!
//! AgentMiddleware 트레이트와 구현체들을 제공합니다.
//!
//! # Available Middleware
//!
//! - [`subagent`]: SubAgent delegation middleware (task tool, registry, execution)
//! - [`summarization`]: Token budget management and context summarization

pub mod traits;
pub mod stack;
pub mod subagent;
pub mod summarization;

// Core traits and types
pub use traits::{AgentMiddleware, DynTool, Tool, ToolDefinition, ToolRegistry, StateUpdate};
pub use stack::MiddlewareStack;

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
