//! rig-deepagents: DeepAgents-style middleware system for Rig
//!
//! LangChain DeepAgents의 핵심 패턴을 Rust로 구현합니다.
//! - AgentMiddleware 트레이트: 도구 자동 주입, 프롬프트 수정
//! - Backend 트레이트: 파일시스템 추상화
//! - MiddlewareStack: 미들웨어 조합 및 실행
//! - AgentExecutor: LLM 호출 및 도구 실행 루프

pub mod error;
pub mod state;
pub mod backends;
pub mod middleware;
pub mod runtime;
pub mod executor;
pub mod tools;

// Re-exports for convenience
pub use error::{BackendError, MiddlewareError, DeepAgentError, WriteResult, EditResult};
pub use state::{AgentState, Message, Role, Todo, TodoStatus, FileData, ToolCall};
pub use backends::{Backend, FileInfo, GrepMatch, MemoryBackend, FilesystemBackend, CompositeBackend};
pub use middleware::{AgentMiddleware, MiddlewareStack, StateUpdate, Tool, ToolDefinition, DynTool};
pub use runtime::{ToolRuntime, RuntimeConfig};
pub use tools::{
    ReadFileTool, WriteFileTool, EditFileTool,
    LsTool, GlobTool, GrepTool,
    WriteTodosTool, TaskTool,
    default_tools, all_tools,
};
pub use executor::{AgentExecutor, LLMProvider};
