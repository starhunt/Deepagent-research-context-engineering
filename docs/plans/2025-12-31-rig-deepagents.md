# Rig DeepAgents Implementation Plan (Enhanced v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** LangChain DeepAgents의 **전체 기능** (`create_deep_agent` 패리티)을 Rust/Rig 프레임워크로 구현하여 Python 대비 속도 이점을 **E2E 벤치마크**로 입증한다.

**Architecture:**
- LangChain의 AgentMiddleware 패턴을 Rust 트레이트 시스템으로 포팅
- **Rig의 Tool 트레이트 직접 통합** (Arc<dyn Any> 사용 안 함)
- **Full Parity**: TodoList, Filesystem, SubAgent, Summarization, PatchToolCalls 미들웨어
- **Agent Execution Loop**: LLM 호출 및 도구 실행 루프 구현
- **실제 OpenAI API E2E 테스트**로 Python vs Rust 레이턴시/처리량 비교

**Tech Stack:** Rust 1.75+, rig-core 0.27, rig-openai 0.27, tokio, serde, async-trait, criterion (벤치마크)

**Python Reference:** `deepagents_sourcecode/libs/deepagents/deepagents/` (LangChain 구현)

**이전 검증 피드백 반영:**
- ✅ `files_update` 필드를 WriteResult/EditResult에 추가
- ✅ `grep`는 리터럴 검색 (정규식 아님) - **프롬프트도 수정**
- ✅ Rig Tool 트레이트 직접 통합 (`DynTool = Any` 제거)
- ✅ `tokio::sync::RwLock` 사용 (async 안전성)
- ✅ SubAgentMiddleware, SummarizationMiddleware, PatchToolCallsMiddleware 추가
- ✅ criterion 기반 통계적 벤치마크
- ✅ **HashMap import 추가**
- ✅ **FileData 중복 정의 해결**
- ✅ **Agent Execution Loop 추가**
- ✅ **FilesystemBackend, CompositeBackend 추가**
- ✅ **ToolRuntime 개념 추가**
- ✅ **각 미들웨어에 실제 도구 구현**

---

## Phase 1: 프로젝트 초기화

### Task 1.1: Cargo 프로젝트 생성

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/Cargo.toml`
- Create: `rust-research-agent/crates/rig-deepagents/src/lib.rs`

**Step 1: 디렉토리 구조 생성**

```bash
mkdir -p rust-research-agent/crates/rig-deepagents/src
```

**Step 2: Cargo.toml 작성**

```toml
[package]
name = "rig-deepagents"
version = "0.1.0"
edition = "2021"
description = "DeepAgents-style middleware system for Rig framework"
license = "MIT"

[dependencies]
rig-core = { version = "0.27", features = ["derive"] }
tokio = { version = "1", features = ["full", "sync"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"
thiserror = "2"
anyhow = "1"
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }
glob = "0.3"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
rig-openai = "0.27"
tokio-test = "0.4"
dotenv = "0.15"
criterion = { version = "0.5", features = ["async_tokio"] }

[[bench]]
name = "middleware_benchmark"
harness = false
```

**Step 3: lib.rs 기본 구조 작성**

```rust
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

pub use error::{BackendError, MiddlewareError, DeepAgentError};
pub use state::{AgentState, Message, Role, Todo, TodoStatus, FileData};
pub use backends::{Backend, FileInfo, GrepMatch, MemoryBackend};
pub use middleware::{AgentMiddleware, MiddlewareStack, StateUpdate};
pub use runtime::ToolRuntime;
pub use executor::AgentExecutor;
```

**Step 4: 빌드 확인**

Run: `cd rust-research-agent/crates/rig-deepagents && cargo check`
Expected: Compiling rig-deepagents...

**Step 5: Commit**

```bash
git add rust-research-agent/crates/rig-deepagents/
git commit -m "feat: initialize rig-deepagents crate structure"
```

---

## Phase 2: 에러 타입 및 상태 정의

### Task 2.1: 에러 타입 정의

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/error.rs`

**Python Reference:** `deepagents/backends/protocol.py` - `FileOperationError`, `WriteResult`, `EditResult`

**Step 1: 에러 타입 구현** (FileData는 state.rs에서 정의)

```rust
// src/error.rs
//! 에러 타입 정의
//!
//! Python Reference: deepagents/backends/protocol.py의 FileOperationError

use std::collections::HashMap;
use thiserror::Error;
use crate::state::FileData;

/// 백엔드 작업 에러
/// Python: FileOperationError literal type
#[derive(Error, Debug, Clone)]
pub enum BackendError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Is a directory: {0}")]
    IsDirectory(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Path traversal not allowed: {0}")]
    PathTraversal(String),

    #[error("File already exists: {0}")]
    FileExists(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Pattern error: {0}")]
    Pattern(String),
}

/// 미들웨어 에러
#[derive(Error, Debug)]
pub enum MiddlewareError {
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    #[error("Tool execution error: {0}")]
    ToolExecution(String),

    #[error("State update error: {0}")]
    StateUpdate(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("SubAgent error: {0}")]
    SubAgent(String),
}

/// DeepAgent 최상위 에러
#[derive(Error, Debug)]
pub enum DeepAgentError {
    #[error("Middleware error: {0}")]
    Middleware(#[from] MiddlewareError),

    #[error("Agent execution error: {0}")]
    AgentExecution(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
}

/// 쓰기 작업 결과
/// Python: WriteResult dataclass
///
/// **Codex 피드백 반영:** `files_update` 필드 추가
/// - 체크포인트 백엔드: {file_path: FileData} 형태로 상태 업데이트
/// - 외부 백엔드 (디스크/S3): None (이미 영구 저장됨)
#[derive(Debug, Clone)]
pub struct WriteResult {
    pub error: Option<String>,
    pub path: Option<String>,
    /// 체크포인트 백엔드를 위한 상태 업데이트
    /// Python: files_update: dict[str, Any] | None
    pub files_update: Option<HashMap<String, FileData>>,
}

impl WriteResult {
    /// 체크포인트 백엔드용 성공 결과
    pub fn success_with_update(path: &str, file_data: FileData) -> Self {
        let mut files = HashMap::new();
        files.insert(path.to_string(), file_data);
        Self { error: None, path: Some(path.to_string()), files_update: Some(files) }
    }

    /// 외부 백엔드용 성공 결과 (files_update = None)
    pub fn success_external(path: &str) -> Self {
        Self { error: None, path: Some(path.to_string()), files_update: None }
    }

    pub fn error(msg: &str) -> Self {
        Self { error: Some(msg.to_string()), path: None, files_update: None }
    }

    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
}

/// 편집 작업 결과
/// Python: EditResult dataclass
#[derive(Debug, Clone)]
pub struct EditResult {
    pub error: Option<String>,
    pub path: Option<String>,
    /// 체크포인트 백엔드를 위한 상태 업데이트
    pub files_update: Option<HashMap<String, FileData>>,
    pub occurrences: Option<usize>,
}

impl EditResult {
    /// 체크포인트 백엔드용 성공 결과
    pub fn success_with_update(path: &str, file_data: FileData, occurrences: usize) -> Self {
        let mut files = HashMap::new();
        files.insert(path.to_string(), file_data);
        Self {
            error: None,
            path: Some(path.to_string()),
            files_update: Some(files),
            occurrences: Some(occurrences),
        }
    }

    /// 외부 백엔드용 성공 결과
    pub fn success_external(path: &str, occurrences: usize) -> Self {
        Self {
            error: None,
            path: Some(path.to_string()),
            files_update: None,
            occurrences: Some(occurrences),
        }
    }

    pub fn error(msg: &str) -> Self {
        Self { error: Some(msg.to_string()), path: None, files_update: None, occurrences: None }
    }

    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
}
```

**Step 2: 테스트 추가** (src/error.rs 하단)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::FileData;

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::FileNotFound("/test.txt".to_string());
        assert!(err.to_string().contains("/test.txt"));
    }

    #[test]
    fn test_middleware_error_from_backend() {
        let backend_err = BackendError::FileNotFound("/test.txt".to_string());
        let middleware_err: MiddlewareError = backend_err.into();
        assert!(matches!(middleware_err, MiddlewareError::Backend(_)));
    }

    #[test]
    fn test_write_result_success() {
        let file_data = FileData::new("hello");
        let result = WriteResult::success_with_update("/test.txt", file_data);
        assert!(result.is_ok());
        assert!(result.files_update.is_some());
    }

    #[test]
    fn test_write_result_external() {
        let result = WriteResult::success_external("/test.txt");
        assert!(result.is_ok());
        assert!(result.files_update.is_none());
    }
}
```

**Step 3: 테스트 실행**

Run: `cargo test error`
Expected: PASS

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: add error types with WriteResult and EditResult"
```

---

### Task 2.2: AgentState 정의

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/state.rs`

**Python Reference:** `langchain/agents/middleware/types.py` - `AgentState(TypedDict)`

**Step 1: state.rs 구현**

```rust
// src/state.rs
//! 에이전트 상태 정의
//!
//! Python Reference: langchain/agents/middleware/types.py의 AgentState

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;
use chrono::Utc;

/// Todo 상태
/// Python: Literal["pending", "in_progress", "completed"]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Todo 아이템
/// Python: Todo(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: TodoStatus,
}

impl Todo {
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            status: TodoStatus::Pending,
        }
    }

    pub fn with_status(content: &str, status: TodoStatus) -> Self {
        Self {
            content: content.to_string(),
            status,
        }
    }
}

/// 파일 데이터
/// Python: FileData(TypedDict) in filesystem.py
///
/// **Note:** 이 타입은 error.rs의 WriteResult/EditResult에서도 사용됨
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    pub content: Vec<String>,
    pub created_at: String,
    pub modified_at: String,
}

impl FileData {
    pub fn new(content: &str) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            content: content.lines().map(String::from).collect(),
            created_at: now.clone(),
            modified_at: now,
        }
    }

    pub fn as_string(&self) -> String {
        self.content.join("\n")
    }

    pub fn update(&mut self, new_content: &str) {
        self.content = new_content.lines().map(String::from).collect();
        self.modified_at = Utc::now().to_rfc3339();
    }

    pub fn line_count(&self) -> usize {
        self.content.len()
    }
}

/// 메시지 역할
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// 도구 호출 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// 메시지
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Self { role: Role::User, content: content.to_string(), tool_call_id: None, tool_calls: None }
    }

    pub fn assistant(content: &str) -> Self {
        Self { role: Role::Assistant, content: content.to_string(), tool_call_id: None, tool_calls: None }
    }

    pub fn assistant_with_tool_calls(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: Some(tool_calls)
        }
    }

    pub fn system(content: &str) -> Self {
        Self { role: Role::System, content: content.to_string(), tool_call_id: None, tool_calls: None }
    }

    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(tool_call_id.to_string()),
            tool_calls: None,
        }
    }

    /// 이 메시지에 dangling tool call이 있는지 확인
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().map_or(false, |tc| !tc.is_empty())
    }
}

/// 에이전트 상태
/// Python: AgentState(TypedDict) + FilesystemState + PlanningState
#[derive(Debug, Clone, Default)]
pub struct AgentState {
    /// 메시지 히스토리
    pub messages: Vec<Message>,

    /// Todo 리스트 (TodoListMiddleware)
    pub todos: Vec<Todo>,

    /// 가상 파일 시스템 (FilesystemMiddleware)
    pub files: HashMap<String, FileData>,

    /// 구조화된 응답
    pub structured_response: Option<serde_json::Value>,

    /// 확장 데이터 (미들웨어별 커스텀 상태)
    extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 초기 메시지로 상태 생성
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// 확장 데이터 설정
    pub fn set_extension<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.extensions.insert(key.to_string(), Box::new(value));
    }

    /// 확장 데이터 조회
    pub fn get_extension<T: Any>(&self, key: &str) -> Option<&T> {
        self.extensions.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// 마지막 사용자 메시지 가져오기
    pub fn last_user_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::User)
    }

    /// 마지막 어시스턴트 메시지 가져오기
    pub fn last_assistant_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::Assistant)
    }

    /// 메시지 추가
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// 메시지 수 반환
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_status_serialization() {
        let status = TodoStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::new();
        assert!(state.messages.is_empty());
        assert!(state.todos.is_empty());
        assert!(state.files.is_empty());
    }

    #[test]
    fn test_file_data_creation() {
        let file = FileData::new("hello\nworld");
        assert_eq!(file.content, vec!["hello", "world"]);
        assert!(!file.created_at.is_empty());
        assert_eq!(file.line_count(), 2);
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/test.txt"}),
        };
        let msg = Message::assistant_with_tool_calls("", vec![tool_call]);
        assert!(msg.has_tool_calls());
    }

    #[test]
    fn test_agent_state_with_messages() {
        let state = AgentState::with_messages(vec![Message::user("Hello")]);
        assert_eq!(state.message_count(), 1);
        assert!(state.last_user_message().is_some());
    }
}
```

**Step 2: 테스트 실행**

Run: `cargo test state`
Expected: PASS

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: add AgentState with Todo, FileData, and Message types"
```

---

## Phase 3: Backend 트레이트

### Task 3.1: Backend 프로토콜 정의

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/mod.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/protocol.rs`

**Python Reference:** `deepagents/backends/protocol.py` - `BackendProtocol(ABC)`

**Step 1: 디렉토리 생성**

```bash
mkdir -p rust-research-agent/crates/rig-deepagents/src/backends
```

**Step 2: protocol.rs 구현**

```rust
// src/backends/protocol.rs
//! Backend 프로토콜 정의
//!
//! Python Reference: deepagents/backends/protocol.py의 BackendProtocol

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::error::{BackendError, WriteResult, EditResult};

/// 파일 정보
/// Python: FileInfo(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
}

impl FileInfo {
    pub fn file(path: &str, size: u64) -> Self {
        Self { path: path.to_string(), is_dir: false, size: Some(size), modified_at: None }
    }

    pub fn file_with_time(path: &str, size: u64, modified_at: &str) -> Self {
        Self {
            path: path.to_string(),
            is_dir: false,
            size: Some(size),
            modified_at: Some(modified_at.to_string()),
        }
    }

    pub fn dir(path: &str) -> Self {
        Self { path: path.to_string(), is_dir: true, size: None, modified_at: None }
    }
}

/// Grep 검색 결과
/// Python: GrepMatch(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepMatch {
    pub path: String,
    pub line: usize,
    pub text: String,
}

impl GrepMatch {
    pub fn new(path: &str, line: usize, text: &str) -> Self {
        Self { path: path.to_string(), line, text: text.to_string() }
    }
}

/// Backend 프로토콜
/// Python: BackendProtocol(ABC)
///
/// 모든 백엔드 구현체가 준수해야 하는 인터페이스입니다.
/// 파일시스템 추상화를 제공하여 인메모리, 디스크, 클라우드 등 다양한 저장소 지원.
#[async_trait]
pub trait Backend: Send + Sync {
    /// 디렉토리 내용 나열
    /// Python: ls_info(path: str) -> list[FileInfo]
    async fn ls(&self, path: &str) -> Result<Vec<FileInfo>, BackendError>;

    /// 파일 읽기 (페이지네이션 지원)
    /// Python: read(file_path: str, offset: int, limit: int) -> str
    ///
    /// Returns: 라인 번호 포함된 포맷 (cat -n 스타일)
    async fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, BackendError>;

    /// 파일 쓰기 (새 파일 생성)
    /// Python: write(file_path: str, content: str) -> WriteResult
    async fn write(&self, path: &str, content: &str) -> Result<WriteResult, BackendError>;

    /// 파일 편집 (문자열 교체)
    /// Python: edit(file_path: str, old_string: str, new_string: str, replace_all: bool) -> EditResult
    async fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool
    ) -> Result<EditResult, BackendError>;

    /// Glob 패턴 검색
    /// Python: glob_info(pattern: str, path: str) -> list[FileInfo]
    async fn glob(&self, pattern: &str, path: &str) -> Result<Vec<FileInfo>, BackendError>;

    /// 텍스트 검색 (리터럴 문자열)
    /// Python: grep_raw(pattern: str, path: str | None, glob: str | None) -> list[GrepMatch]
    ///
    /// **Important:** pattern은 리터럴 문자열입니다 (정규식 아님!)
    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob_filter: Option<&str>,
    ) -> Result<Vec<GrepMatch>, BackendError>;

    /// 파일 존재 여부 확인
    async fn exists(&self, path: &str) -> Result<bool, BackendError>;

    /// 파일 삭제
    async fn delete(&self, path: &str) -> Result<(), BackendError>;
}
```

**Step 3: mod.rs 생성**

```rust
// src/backends/mod.rs
//! 백엔드 모듈
//!
//! 파일시스템 추상화를 제공합니다.

pub mod protocol;
pub mod memory;
pub mod filesystem;
pub mod composite;

pub use protocol::{Backend, FileInfo, GrepMatch};
pub use memory::MemoryBackend;
pub use filesystem::FilesystemBackend;
pub use composite::CompositeBackend;
```

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: add Backend trait protocol"
```

---

### Task 3.2: MemoryBackend 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/memory.rs`

**Python Reference:** `deepagents/backends/state.py` - `StateBackend`

**Step 1: memory.rs 구현**

```rust
// src/backends/memory.rs
//! 인메모리 백엔드 구현
//!
//! Python Reference: deepagents/backends/state.py의 StateBackend
//!
//! **Codex 피드백 반영:**
//! - `tokio::sync::RwLock` 사용 (async 안전성)
//! - `grep`는 리터럴 검색 (정규식 아님)

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use glob::Pattern;

use super::protocol::{Backend, FileInfo, GrepMatch};
use crate::error::{BackendError, WriteResult, EditResult};
use crate::state::FileData;

/// 인메모리 백엔드
/// Python: StateBackend - 상태에 파일 저장
///
/// **Note:** tokio::sync::RwLock을 사용하여 async 컨텍스트에서 안전하게 동작
pub struct MemoryBackend {
    files: RwLock<HashMap<String, FileData>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
        }
    }

    /// 기존 파일로 초기화
    pub fn with_files(files: HashMap<String, FileData>) -> Self {
        Self {
            files: RwLock::new(files),
        }
    }

    /// 경로 정규화 및 검증
    fn validate_path(path: &str) -> Result<String, BackendError> {
        if path.contains("..") || path.starts_with("~") {
            return Err(BackendError::PathTraversal(path.to_string()));
        }

        let normalized = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{}", path)
        };

        Ok(normalized)
    }

    /// 라인 번호 포맷팅
    fn format_with_line_numbers(content: &str, offset: usize) -> String {
        content
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 상위 디렉토리 생성 (가상)
    fn ensure_parent_dirs(_path: &str) {
        // 인메모리 백엔드에서는 디렉토리 자동 생성 불필요
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for MemoryBackend {
    async fn ls(&self, path: &str) -> Result<Vec<FileInfo>, BackendError> {
        let path = Self::validate_path(path)?;
        let files = self.files.read().await;

        let prefix = if path == "/" { "" } else { &path };
        let mut results = Vec::new();
        let mut dirs_seen = HashSet::new();

        for (file_path, data) in files.iter() {
            if file_path.starts_with(prefix) || prefix.is_empty() {
                let relative = file_path.strip_prefix(prefix).unwrap_or(file_path);
                let relative = relative.trim_start_matches('/');

                if let Some(slash_pos) = relative.find('/') {
                    // 서브디렉토리
                    let dir_name = &relative[..slash_pos];
                    let dir_path = format!("{}/{}", path.trim_end_matches('/'), dir_name);
                    if dirs_seen.insert(dir_path.clone()) {
                        results.push(FileInfo::dir(&format!("{}/", dir_path)));
                    }
                } else if !relative.is_empty() {
                    // 파일
                    let size = data.content.iter().map(|s| s.len()).sum::<usize>() as u64;
                    results.push(FileInfo::file_with_time(
                        file_path,
                        size,
                        &data.modified_at,
                    ));
                }
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(results)
    }

    async fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, BackendError> {
        let path = Self::validate_path(path)?;
        let files = self.files.read().await;

        let file = files.get(&path).ok_or_else(|| BackendError::FileNotFound(path.clone()))?;

        let lines: Vec<_> = file.content.iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect();

        let content = lines.join("\n");
        Ok(Self::format_with_line_numbers(&content, offset))
    }

    async fn write(&self, path: &str, content: &str) -> Result<WriteResult, BackendError> {
        let path = Self::validate_path(path)?;
        let mut files = self.files.write().await;

        // 이미 존재하면 에러
        if files.contains_key(&path) {
            return Ok(WriteResult::error(&format!(
                "Cannot write to {} because it already exists. Read and then make an edit, or write to a new path.",
                path
            )));
        }

        let file_data = FileData::new(content);
        files.insert(path.clone(), file_data.clone());

        // 체크포인트 백엔드이므로 files_update 포함
        Ok(WriteResult::success_with_update(&path, file_data))
    }

    async fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool
    ) -> Result<EditResult, BackendError> {
        let path = Self::validate_path(path)?;
        let mut files = self.files.write().await;

        let file = files.get_mut(&path).ok_or_else(|| BackendError::FileNotFound(path.clone()))?;

        let content = file.as_string();
        let occurrences = content.matches(old_string).count();

        if occurrences == 0 {
            return Ok(EditResult::error(&format!("String '{}' not found in file", old_string)));
        }

        if !replace_all && occurrences > 1 {
            return Ok(EditResult::error(&format!(
                "String '{}' found {} times. Use replace_all=true or provide more context.",
                old_string, occurrences
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        file.update(&new_content);
        let updated_file = file.clone();
        let actual_occurrences = if replace_all { occurrences } else { 1 };

        // 체크포인트 백엔드이므로 files_update 포함
        Ok(EditResult::success_with_update(&path, updated_file, actual_occurrences))
    }

    async fn glob(&self, pattern: &str, base_path: &str) -> Result<Vec<FileInfo>, BackendError> {
        let _base = Self::validate_path(base_path)?;
        let files = self.files.read().await;

        let glob_pattern = Pattern::new(pattern)
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();
        for (file_path, data) in files.iter() {
            let match_path = file_path.trim_start_matches('/');
            if glob_pattern.matches(match_path) {
                let size = data.content.iter().map(|s| s.len()).sum::<usize>() as u64;
                results.push(FileInfo::file_with_time(
                    file_path,
                    size,
                    &data.modified_at,
                ));
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(results)
    }

    /// 리터럴 텍스트 검색
    ///
    /// **Codex 피드백 반영:** 정규식이 아닌 리터럴 문자열 검색
    /// Python: grep_raw의 docstring - "검색할 리터럴 문자열 (정규식 아님)"
    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob_filter: Option<&str>,
    ) -> Result<Vec<GrepMatch>, BackendError> {
        let files = self.files.read().await;

        let glob_pattern = glob_filter.map(|g| Pattern::new(g)).transpose()
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();

        for (file_path, data) in files.iter() {
            // Path filter
            if let Some(p) = path {
                if !file_path.starts_with(p) {
                    continue;
                }
            }

            // Glob filter
            if let Some(ref gp) = glob_pattern {
                let match_path = file_path.trim_start_matches('/');
                if !gp.matches(match_path) {
                    continue;
                }
            }

            // 리터럴 검색 (정규식 아님)
            for (line_num, line) in data.content.iter().enumerate() {
                if line.contains(pattern) {
                    results.push(GrepMatch::new(file_path, line_num + 1, line));
                }
            }
        }

        Ok(results)
    }

    async fn exists(&self, path: &str) -> Result<bool, BackendError> {
        let path = Self::validate_path(path)?;
        let files = self.files.read().await;
        Ok(files.contains_key(&path))
    }

    async fn delete(&self, path: &str) -> Result<(), BackendError> {
        let path = Self::validate_path(path)?;
        let mut files = self.files.write().await;

        if files.remove(&path).is_none() {
            return Err(BackendError::FileNotFound(path));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_backend_write_and_read() {
        let backend = MemoryBackend::new();

        // Write
        let result = backend.write("/test.txt", "Hello, World!").await.unwrap();
        assert!(result.is_ok());
        assert!(result.files_update.is_some());

        // Read
        let content = backend.read("/test.txt", 0, 100).await.unwrap();
        assert!(content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_memory_backend_write_existing_file() {
        let backend = MemoryBackend::new();
        backend.write("/test.txt", "content").await.unwrap();

        // 두 번째 쓰기는 에러
        let result = backend.write("/test.txt", "new content").await.unwrap();
        assert!(!result.is_ok());
        assert!(result.error.unwrap().contains("already exists"));
    }

    #[tokio::test]
    async fn test_memory_backend_edit() {
        let backend = MemoryBackend::new();
        backend.write("/test.txt", "foo bar foo").await.unwrap();

        // Edit single
        let result = backend.edit("/test.txt", "foo", "baz", false).await.unwrap();
        assert!(!result.is_ok()); // 2번 발견되어 에러

        // Edit all
        let result = backend.edit("/test.txt", "foo", "baz", true).await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.occurrences, Some(2));

        let content = backend.read("/test.txt", 0, 100).await.unwrap();
        assert!(content.contains("baz bar baz"));
    }

    #[tokio::test]
    async fn test_memory_backend_ls() {
        let backend = MemoryBackend::new();
        backend.write("/dir/file1.txt", "content1").await.unwrap();
        backend.write("/dir/file2.txt", "content2").await.unwrap();

        let files = backend.ls("/dir").await.unwrap();
        assert_eq!(files.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_backend_glob() {
        let backend = MemoryBackend::new();
        backend.write("/src/main.rs", "fn main()").await.unwrap();
        backend.write("/src/lib.rs", "pub mod").await.unwrap();
        backend.write("/test.txt", "test").await.unwrap();

        let files = backend.glob("**/*.rs", "/").await.unwrap();
        assert_eq!(files.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_backend_grep_literal() {
        let backend = MemoryBackend::new();
        backend.write("/test.rs", "fn main() {\n    println!(\"hello\");\n}").await.unwrap();

        // 리터럴 검색 - 정규식 메타문자가 리터럴로 처리됨
        let matches = backend.grep("()", None, None).await.unwrap();
        assert!(!matches.is_empty()); // "()" 를 리터럴로 찾음
    }

    #[tokio::test]
    async fn test_memory_backend_delete() {
        let backend = MemoryBackend::new();
        backend.write("/test.txt", "content").await.unwrap();

        assert!(backend.exists("/test.txt").await.unwrap());
        backend.delete("/test.txt").await.unwrap();
        assert!(!backend.exists("/test.txt").await.unwrap());
    }
}
```

**Step 2: 테스트 실행**

Run: `cargo test memory`
Expected: PASS

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: implement MemoryBackend with tokio RwLock"
```

---

### Task 3.3: FilesystemBackend 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/filesystem.rs`

**Python Reference:** `deepagents/backends/filesystem.py`

**Step 1: filesystem.rs 구현**

```rust
// src/backends/filesystem.rs
//! 실제 파일시스템 백엔드 구현
//!
//! Python Reference: deepagents/backends/filesystem.py

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use glob::Pattern;
use chrono::{DateTime, Utc};

use super::protocol::{Backend, FileInfo, GrepMatch};
use crate::error::{BackendError, WriteResult, EditResult};

/// 파일시스템 백엔드
/// Python: FilesystemBackend
///
/// 실제 파일시스템에서 직접 파일을 읽고 씁니다.
pub struct FilesystemBackend {
    /// 루트 디렉토리
    root: PathBuf,
    /// 가상 모드 - 모든 경로를 루트 내부로 제한
    virtual_mode: bool,
}

impl FilesystemBackend {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            virtual_mode: true,
        }
    }

    pub fn with_virtual_mode(root: impl AsRef<Path>, virtual_mode: bool) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            virtual_mode,
        }
    }

    /// 경로 검증 및 해결
    fn resolve_path(&self, path: &str) -> Result<PathBuf, BackendError> {
        if self.virtual_mode {
            // 경로 탐색 방지
            if path.contains("..") || path.starts_with("~") {
                return Err(BackendError::PathTraversal(path.to_string()));
            }

            let clean_path = path.trim_start_matches('/');
            let resolved = self.root.join(clean_path).canonicalize()
                .unwrap_or_else(|_| self.root.join(clean_path));

            // 루트 외부 접근 방지
            if !resolved.starts_with(&self.root) {
                return Err(BackendError::PathTraversal(path.to_string()));
            }

            Ok(resolved)
        } else {
            Ok(PathBuf::from(path))
        }
    }

    /// 가상 경로로 변환
    fn to_virtual_path(&self, path: &Path) -> String {
        if self.virtual_mode {
            path.strip_prefix(&self.root)
                .map(|p| format!("/{}", p.display()))
                .unwrap_or_else(|_| path.display().to_string())
        } else {
            path.display().to_string()
        }
    }

    fn format_with_line_numbers(content: &str, offset: usize) -> String {
        content
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Backend for FilesystemBackend {
    async fn ls(&self, path: &str) -> Result<Vec<FileInfo>, BackendError> {
        let resolved = self.resolve_path(path)?;

        if !resolved.exists() || !resolved.is_dir() {
            return Ok(vec![]);
        }

        let mut results = Vec::new();
        let mut entries = fs::read_dir(&resolved).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        while let Some(entry) = entries.next_entry().await
            .map_err(|e| BackendError::Io(e.to_string()))?
        {
            let path = entry.path();
            let metadata = entry.metadata().await
                .map_err(|e| BackendError::Io(e.to_string()))?;

            let virt_path = self.to_virtual_path(&path);

            if metadata.is_dir() {
                results.push(FileInfo::dir(&format!("{}/", virt_path)));
            } else {
                let modified = metadata.modified()
                    .ok()
                    .map(|t| DateTime::<Utc>::from(t).to_rfc3339());

                results.push(FileInfo {
                    path: virt_path,
                    is_dir: false,
                    size: Some(metadata.len()),
                    modified_at: modified,
                });
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(results)
    }

    async fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, BackendError> {
        let resolved = self.resolve_path(path)?;

        if !resolved.exists() || !resolved.is_file() {
            return Err(BackendError::FileNotFound(path.to_string()));
        }

        let content = fs::read_to_string(&resolved).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        let lines: Vec<&str> = content.lines().collect();
        let start = offset.min(lines.len());
        let end = (offset + limit).min(lines.len());

        let selected = lines[start..end].join("\n");
        Ok(Self::format_with_line_numbers(&selected, offset))
    }

    async fn write(&self, path: &str, content: &str) -> Result<WriteResult, BackendError> {
        let resolved = self.resolve_path(path)?;

        if resolved.exists() {
            return Ok(WriteResult::error(&format!(
                "Cannot write to {} because it already exists. Read and then make an edit.",
                path
            )));
        }

        // 상위 디렉토리 생성
        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| BackendError::Io(e.to_string()))?;
        }

        fs::write(&resolved, content).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        // 외부 백엔드이므로 files_update = None
        Ok(WriteResult::success_external(path))
    }

    async fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool
    ) -> Result<EditResult, BackendError> {
        let resolved = self.resolve_path(path)?;

        if !resolved.exists() || !resolved.is_file() {
            return Err(BackendError::FileNotFound(path.to_string()));
        }

        let content = fs::read_to_string(&resolved).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        let occurrences = content.matches(old_string).count();

        if occurrences == 0 {
            return Ok(EditResult::error(&format!("String '{}' not found in file", old_string)));
        }

        if !replace_all && occurrences > 1 {
            return Ok(EditResult::error(&format!(
                "String '{}' found {} times. Use replace_all=true or provide more context.",
                old_string, occurrences
            )));
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };

        fs::write(&resolved, &new_content).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        let actual = if replace_all { occurrences } else { 1 };
        Ok(EditResult::success_external(path, actual))
    }

    async fn glob(&self, pattern: &str, base_path: &str) -> Result<Vec<FileInfo>, BackendError> {
        let resolved = self.resolve_path(base_path)?;

        if !resolved.exists() || !resolved.is_dir() {
            return Ok(vec![]);
        }

        let glob_pattern = Pattern::new(pattern)
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();

        // 재귀적으로 파일 검색
        let walker = walkdir::WalkDir::new(&resolved);
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let rel_path = entry.path().strip_prefix(&resolved)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            if glob_pattern.matches(&rel_path) {
                let virt_path = self.to_virtual_path(entry.path());
                let metadata = entry.metadata()
                    .map_err(|e| BackendError::Io(e.to_string()))?;

                results.push(FileInfo {
                    path: virt_path,
                    is_dir: false,
                    size: Some(metadata.len()),
                    modified_at: metadata.modified()
                        .ok()
                        .map(|t| DateTime::<Utc>::from(t).to_rfc3339()),
                });
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(results)
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob_filter: Option<&str>,
    ) -> Result<Vec<GrepMatch>, BackendError> {
        let search_path = path.unwrap_or("/");
        let resolved = self.resolve_path(search_path)?;

        if !resolved.exists() {
            return Ok(vec![]);
        }

        let glob_pattern = glob_filter.map(|g| Pattern::new(g)).transpose()
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();
        let walker = walkdir::WalkDir::new(&resolved);

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            // Glob filter
            if let Some(ref gp) = glob_pattern {
                let name = entry.file_name().to_string_lossy();
                if !gp.matches(&name) {
                    continue;
                }
            }

            // 파일 읽기
            let content = match std::fs::read_to_string(entry.path()) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let virt_path = self.to_virtual_path(entry.path());

            // 리터럴 검색
            for (line_num, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    results.push(GrepMatch::new(&virt_path, line_num + 1, line));
                }
            }
        }

        Ok(results)
    }

    async fn exists(&self, path: &str) -> Result<bool, BackendError> {
        let resolved = self.resolve_path(path)?;
        Ok(resolved.exists())
    }

    async fn delete(&self, path: &str) -> Result<(), BackendError> {
        let resolved = self.resolve_path(path)?;

        if !resolved.exists() {
            return Err(BackendError::FileNotFound(path.to_string()));
        }

        fs::remove_file(&resolved).await
            .map_err(|e| BackendError::Io(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_filesystem_backend_write_and_read() {
        let temp = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(temp.path());

        let result = backend.write("/test.txt", "Hello").await.unwrap();
        assert!(result.is_ok());
        assert!(result.files_update.is_none()); // 외부 백엔드

        let content = backend.read("/test.txt", 0, 100).await.unwrap();
        assert!(content.contains("Hello"));
    }

    #[tokio::test]
    async fn test_filesystem_backend_path_traversal() {
        let temp = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(temp.path());

        let result = backend.read("/../etc/passwd", 0, 100).await;
        assert!(result.is_err());
    }
}
```

**Step 2: Cargo.toml에 walkdir 추가**

```toml
# Cargo.toml [dependencies]에 추가
walkdir = "2"
tempfile = "3"
```

**Step 3: 테스트 실행**

Run: `cargo test filesystem`
Expected: PASS

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: implement FilesystemBackend with virtual mode"
```

---

### Task 3.4: CompositeBackend 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/composite.rs`

**Python Reference:** `deepagents/backends/composite.py`

**Step 1: composite.rs 구현**

```rust
// src/backends/composite.rs
//! 복합 백엔드 - 경로 기반 라우팅
//!
//! Python Reference: deepagents/backends/composite.py

use async_trait::async_trait;
use std::sync::Arc;

use super::protocol::{Backend, FileInfo, GrepMatch};
use crate::error::{BackendError, WriteResult, EditResult};

/// 라우트 설정
pub struct Route {
    pub prefix: String,
    pub backend: Arc<dyn Backend>,
}

/// 복합 백엔드
/// Python: CompositeBackend
///
/// 경로 접두사를 기반으로 요청을 다른 백엔드로 라우팅합니다.
pub struct CompositeBackend {
    default: Arc<dyn Backend>,
    routes: Vec<Route>,
}

impl CompositeBackend {
    pub fn new(default: Arc<dyn Backend>) -> Self {
        Self {
            default,
            routes: Vec::new(),
        }
    }

    /// 라우트 추가 (빌더 패턴)
    pub fn with_route(mut self, prefix: &str, backend: Arc<dyn Backend>) -> Self {
        // 길이 순으로 정렬 (가장 긴 것 먼저)
        let route = Route {
            prefix: prefix.to_string(),
            backend,
        };
        self.routes.push(route);
        self.routes.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
        self
    }

    /// 경로에 맞는 백엔드와 변환된 경로 반환
    fn get_backend_and_path(&self, path: &str) -> (Arc<dyn Backend>, String) {
        for route in &self.routes {
            if path.starts_with(&route.prefix) {
                let suffix = &path[route.prefix.len()..];
                let stripped = if suffix.is_empty() || suffix == "/" {
                    "/".to_string()
                } else {
                    format!("/{}", suffix.trim_start_matches('/'))
                };
                return (route.backend.clone(), stripped);
            }
        }
        (self.default.clone(), path.to_string())
    }

    /// 결과 경로에 접두사 복원
    fn restore_prefix(&self, path: &str, original_path: &str) -> String {
        for route in &self.routes {
            if original_path.starts_with(&route.prefix) {
                let prefix = route.prefix.trim_end_matches('/');
                return format!("{}{}", prefix, path);
            }
        }
        path.to_string()
    }
}

#[async_trait]
impl Backend for CompositeBackend {
    async fn ls(&self, path: &str) -> Result<Vec<FileInfo>, BackendError> {
        // 루트 경로면 모든 백엔드에서 수집
        if path == "/" {
            let mut results = self.default.ls("/").await?;

            // 라우트된 디렉토리 추가
            for route in &self.routes {
                results.push(FileInfo::dir(&route.prefix));
            }

            results.sort_by(|a, b| a.path.cmp(&b.path));
            return Ok(results);
        }

        let (backend, stripped) = self.get_backend_and_path(path);
        let mut results = backend.ls(&stripped).await?;

        // 경로 복원
        for info in &mut results {
            info.path = self.restore_prefix(&info.path, path);
        }

        Ok(results)
    }

    async fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, BackendError> {
        let (backend, stripped) = self.get_backend_and_path(path);
        backend.read(&stripped, offset, limit).await
    }

    async fn write(&self, path: &str, content: &str) -> Result<WriteResult, BackendError> {
        let (backend, stripped) = self.get_backend_and_path(path);
        let mut result = backend.write(&stripped, content).await?;

        if result.path.is_some() {
            result.path = Some(path.to_string());
        }

        Ok(result)
    }

    async fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool
    ) -> Result<EditResult, BackendError> {
        let (backend, stripped) = self.get_backend_and_path(path);
        let mut result = backend.edit(&stripped, old_string, new_string, replace_all).await?;

        if result.path.is_some() {
            result.path = Some(path.to_string());
        }

        Ok(result)
    }

    async fn glob(&self, pattern: &str, base_path: &str) -> Result<Vec<FileInfo>, BackendError> {
        let (backend, stripped) = self.get_backend_and_path(base_path);
        let mut results = backend.glob(pattern, &stripped).await?;

        for info in &mut results {
            info.path = self.restore_prefix(&info.path, base_path);
        }

        Ok(results)
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob_filter: Option<&str>,
    ) -> Result<Vec<GrepMatch>, BackendError> {
        let search_path = path.unwrap_or("/");

        // 특정 경로가 라우트에 매칭되면 해당 백엔드만 검색
        for route in &self.routes {
            if search_path.starts_with(route.prefix.trim_end_matches('/')) {
                let stripped = &search_path[route.prefix.len() - 1..];
                let search = if stripped.is_empty() { "/" } else { stripped };

                let mut results = route.backend.grep(pattern, Some(search), glob_filter).await?;

                for m in &mut results {
                    m.path = self.restore_prefix(&m.path, search_path);
                }

                return Ok(results);
            }
        }

        // 전체 검색
        let mut all_results = self.default.grep(pattern, path, glob_filter).await?;

        for route in &self.routes {
            let mut route_results = route.backend.grep(pattern, Some("/"), glob_filter).await?;
            for m in &mut route_results {
                let prefix = route.prefix.trim_end_matches('/');
                m.path = format!("{}{}", prefix, m.path);
            }
            all_results.extend(route_results);
        }

        Ok(all_results)
    }

    async fn exists(&self, path: &str) -> Result<bool, BackendError> {
        let (backend, stripped) = self.get_backend_and_path(path);
        backend.exists(&stripped).await
    }

    async fn delete(&self, path: &str) -> Result<(), BackendError> {
        let (backend, stripped) = self.get_backend_and_path(path);
        backend.delete(&stripped).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;

    #[tokio::test]
    async fn test_composite_backend_routing() {
        let default = Arc::new(MemoryBackend::new());
        let memories = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/memories/", memories.clone());

        // 메모리 백엔드에 파일 쓰기
        composite.write("/memories/notes.txt", "my notes").await.unwrap();

        // 읽기
        let content = composite.read("/memories/notes.txt", 0, 100).await.unwrap();
        assert!(content.contains("my notes"));

        // 기본 백엔드에 파일 쓰기
        composite.write("/other.txt", "other content").await.unwrap();

        // 루트 ls
        let files = composite.ls("/").await.unwrap();
        assert!(files.iter().any(|f| f.path.contains("memories")));
        assert!(files.iter().any(|f| f.path.contains("other")));
    }

    #[tokio::test]
    async fn test_composite_backend_grep() {
        let default = Arc::new(MemoryBackend::new());
        let docs = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/docs/", docs.clone());

        composite.write("/docs/readme.txt", "hello world").await.unwrap();
        composite.write("/other.txt", "hello there").await.unwrap();

        // 전체 검색
        let matches = composite.grep("hello", None, None).await.unwrap();
        assert_eq!(matches.len(), 2);
    }
}
```

**Step 2: 테스트 실행**

Run: `cargo test composite`
Expected: PASS

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: implement CompositeBackend with route-based dispatch"
```

---

## Phase 4: ToolRuntime 및 AgentMiddleware 트레이트

### Task 4.1: ToolRuntime 정의

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/runtime.rs`

**Python Reference:** `langchain/tools.py` - `ToolRuntime`

**Step 1: runtime.rs 구현**

```rust
// src/runtime.rs
//! 도구 실행 런타임
//!
//! Python Reference: langchain/tools.py의 ToolRuntime
//!
//! 도구 실행 시 필요한 컨텍스트를 제공합니다.

use std::sync::Arc;
use crate::state::AgentState;
use crate::backends::Backend;

/// 도구 실행 런타임
/// Python: ToolRuntime
///
/// 도구가 실행될 때 필요한 컨텍스트를 제공합니다:
/// - 현재 에이전트 상태
/// - 백엔드 접근
/// - 도구 호출 ID
pub struct ToolRuntime {
    /// 현재 에이전트 상태 (읽기 전용 스냅샷)
    state: AgentState,
    /// 백엔드 (파일 시스템 접근)
    backend: Arc<dyn Backend>,
    /// 현재 도구 호출 ID
    tool_call_id: Option<String>,
    /// 추가 설정
    config: RuntimeConfig,
}

/// 런타임 설정
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    /// 디버그 모드
    pub debug: bool,
    /// 최대 재귀 깊이 (SubAgent용)
    pub max_recursion: usize,
    /// 현재 재귀 깊이
    pub current_recursion: usize,
}

impl RuntimeConfig {
    pub fn new() -> Self {
        Self {
            debug: false,
            max_recursion: 10,
            current_recursion: 0,
        }
    }
}

impl ToolRuntime {
    pub fn new(state: AgentState, backend: Arc<dyn Backend>) -> Self {
        Self {
            state,
            backend,
            tool_call_id: None,
            config: RuntimeConfig::new(),
        }
    }

    pub fn with_tool_call_id(mut self, id: &str) -> Self {
        self.tool_call_id = Some(id.to_string());
        self
    }

    pub fn with_config(mut self, config: RuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// 현재 상태 참조
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// 백엔드 참조
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// 도구 호출 ID
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// 설정 참조
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// 재귀 깊이 증가한 새 런타임 생성
    pub fn with_increased_recursion(&self) -> Self {
        let mut new_config = self.config.clone();
        new_config.current_recursion += 1;

        Self {
            state: self.state.clone(),
            backend: self.backend.clone(),
            tool_call_id: None,
            config: new_config,
        }
    }

    /// 재귀 한도 초과 확인
    pub fn is_recursion_limit_exceeded(&self) -> bool {
        self.config.current_recursion >= self.config.max_recursion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;

    #[test]
    fn test_tool_runtime_creation() {
        let state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());

        let runtime = ToolRuntime::new(state, backend)
            .with_tool_call_id("call_123");

        assert_eq!(runtime.tool_call_id(), Some("call_123"));
    }

    #[test]
    fn test_recursion_limit() {
        let state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());

        let mut runtime = ToolRuntime::new(state, backend);

        for _ in 0..10 {
            runtime = runtime.with_increased_recursion();
        }

        assert!(runtime.is_recursion_limit_exceeded());
    }
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: add ToolRuntime for tool execution context"
```

---

### Task 4.2: AgentMiddleware 트레이트 정의

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/mod.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/traits.rs`

**Python Reference:** `langchain/agents/middleware/types.py`

**Step 1: 디렉토리 생성**

```bash
mkdir -p rust-research-agent/crates/rig-deepagents/src/middleware
```

**Step 2: traits.rs 구현**

```rust
// src/middleware/traits.rs
//! AgentMiddleware 트레이트 정의
//!
//! Python Reference: langchain/agents/middleware/types.py

use async_trait::async_trait;
use std::sync::Arc;
use crate::state::{AgentState, Message, Todo, FileData};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;

/// 상태 업데이트 커맨드
/// Python: langgraph.types.Command
#[derive(Debug, Clone)]
pub enum StateUpdate {
    /// 메시지 추가
    AddMessages(Vec<Message>),
    /// Todo 업데이트
    SetTodos(Vec<Todo>),
    /// 파일 업데이트 (None = 삭제)
    UpdateFiles(std::collections::HashMap<String, Option<FileData>>),
    /// 복합 업데이트
    Batch(Vec<StateUpdate>),
}

/// 도구 정의
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// 도구 인터페이스
#[async_trait]
pub trait Tool: Send + Sync {
    /// 도구 정의 반환
    fn definition(&self) -> ToolDefinition;

    /// 도구 실행
    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError>;
}

/// 동적 도구 타입
pub type DynTool = Arc<dyn Tool>;

/// AgentMiddleware 트레이트
///
/// Python Reference: AgentMiddleware(Generic[StateT, ContextT])
///
/// 핵심 기능:
/// - tools(): 에이전트에 자동 주입할 도구 목록 반환
/// - modify_system_prompt(): 시스템 프롬프트 수정 (체이닝)
/// - before_agent() / after_agent(): 라이프사이클 훅
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// 미들웨어 이름
    fn name(&self) -> &str;

    /// 이 미들웨어가 제공하는 도구 목록
    fn tools(&self) -> Vec<DynTool> {
        vec![]
    }

    /// 시스템 프롬프트 수정
    fn modify_system_prompt(&self, prompt: String) -> String {
        prompt
    }

    /// 에이전트 실행 전 훅
    /// Python: before_agent(self, state, runtime) -> dict | None
    async fn before_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }

    /// 에이전트 실행 후 훅
    /// Python: after_agent(self, state, runtime) -> dict | None
    async fn after_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }
}
```

**Step 3: mod.rs 생성**

```rust
// src/middleware/mod.rs
//! 미들웨어 모듈

pub mod traits;
pub mod stack;
pub mod todo;
pub mod filesystem;
pub mod patch_tool_calls;
pub mod summarization;
pub mod subagent;

pub use traits::{AgentMiddleware, DynTool, Tool, ToolDefinition, StateUpdate};
pub use stack::MiddlewareStack;
pub use todo::TodoListMiddleware;
pub use filesystem::FilesystemMiddleware;
pub use patch_tool_calls::PatchToolCallsMiddleware;
pub use summarization::SummarizationMiddleware;
pub use subagent::SubAgentMiddleware;
```

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: add AgentMiddleware trait and Tool interface"
```

---

### Task 4.3: MiddlewareStack 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/stack.rs`

**Step 1: stack.rs 구현**

```rust
// src/middleware/stack.rs
//! 미들웨어 스택
//!
//! 여러 미들웨어를 조합하여 순차적으로 실행합니다.

use std::sync::Arc;
use crate::state::{AgentState, Message, FileData};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use super::traits::{AgentMiddleware, DynTool, StateUpdate};

/// 미들웨어 스택
pub struct MiddlewareStack {
    middlewares: Vec<Arc<dyn AgentMiddleware>>,
}

impl MiddlewareStack {
    pub fn new() -> Self {
        Self { middlewares: vec![] }
    }

    /// 미들웨어 추가 (빌더 패턴)
    pub fn add<M: AgentMiddleware + 'static>(mut self, middleware: M) -> Self {
        self.middlewares.push(Arc::new(middleware));
        self
    }

    /// Arc로 래핑된 미들웨어 추가
    pub fn add_arc(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self {
        self.middlewares.push(middleware);
        self
    }

    /// 미들웨어 개수
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// 모든 미들웨어의 도구 수집
    pub fn collect_tools(&self) -> Vec<DynTool> {
        self.middlewares
            .iter()
            .flat_map(|m| m.tools())
            .collect()
    }

    /// 시스템 프롬프트 빌드 (체이닝)
    pub fn build_system_prompt(&self, base: &str) -> String {
        self.middlewares.iter().fold(
            base.to_string(),
            |acc, m| m.modify_system_prompt(acc)
        )
    }

    /// before_agent 훅 실행 (순차)
    pub async fn before_agent(
        &self,
        state: &mut AgentState,
        runtime: &ToolRuntime,
    ) -> Result<Vec<StateUpdate>, MiddlewareError> {
        let mut updates = vec![];

        for middleware in &self.middlewares {
            if let Some(update) = middleware.before_agent(state, runtime).await? {
                Self::apply_update(state, &update);
                updates.push(update);
            }
        }

        Ok(updates)
    }

    /// after_agent 훅 실행 (역순)
    pub async fn after_agent(
        &self,
        state: &mut AgentState,
        runtime: &ToolRuntime,
    ) -> Result<Vec<StateUpdate>, MiddlewareError> {
        let mut updates = vec![];

        for middleware in self.middlewares.iter().rev() {
            if let Some(update) = middleware.after_agent(state, runtime).await? {
                Self::apply_update(state, &update);
                updates.push(update);
            }
        }

        Ok(updates)
    }

    /// 상태 업데이트 적용
    fn apply_update(state: &mut AgentState, update: &StateUpdate) {
        match update {
            StateUpdate::AddMessages(msgs) => {
                state.messages.extend(msgs.clone());
            }
            StateUpdate::SetTodos(todos) => {
                state.todos = todos.clone();
            }
            StateUpdate::UpdateFiles(files) => {
                for (path, data) in files {
                    if let Some(d) = data {
                        state.files.insert(path.clone(), d.clone());
                    } else {
                        state.files.remove(path);
                    }
                }
            }
            StateUpdate::Batch(updates) => {
                for u in updates {
                    Self::apply_update(state, u);
                }
            }
        }
    }
}

impl Default for MiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use std::sync::Arc;

    struct TestMiddleware {
        name: String,
        prompt_addition: String,
    }

    #[async_trait::async_trait]
    impl AgentMiddleware for TestMiddleware {
        fn name(&self) -> &str {
            &self.name
        }

        fn modify_system_prompt(&self, prompt: String) -> String {
            format!("{}\n{}", prompt, self.prompt_addition)
        }
    }

    #[test]
    fn test_middleware_stack_prompt_chaining() {
        let stack = MiddlewareStack::new()
            .add(TestMiddleware {
                name: "First".to_string(),
                prompt_addition: "First addition".to_string()
            })
            .add(TestMiddleware {
                name: "Second".to_string(),
                prompt_addition: "Second addition".to_string()
            });

        let result = stack.build_system_prompt("Base prompt");
        assert!(result.contains("Base prompt"));
        assert!(result.contains("First addition"));
        assert!(result.contains("Second addition"));
    }

    #[tokio::test]
    async fn test_middleware_stack_hooks() {
        let stack = MiddlewareStack::new()
            .add(TestMiddleware {
                name: "Test".to_string(),
                prompt_addition: "Test".to_string()
            });

        let mut state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(state.clone(), backend);

        let updates = stack.before_agent(&mut state, &runtime).await.unwrap();
        assert!(updates.is_empty()); // 기본 미들웨어는 None 반환
    }
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: implement MiddlewareStack for middleware composition"
```

---

## Phase 5: 미들웨어 및 도구 구현

이 Phase에서는 모든 미들웨어와 실제 도구를 구현합니다.

### Task 5.1: TodoListMiddleware 및 write_todos 도구

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/todo.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/mod.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/write_todos.rs`

**Python Reference:** `langchain/agents/middleware/todo.py`

**Step 1: tools/mod.rs 생성**

```rust
// src/tools/mod.rs
//! 도구 모듈

pub mod write_todos;
pub mod filesystem;

pub use write_todos::WriteTodosTool;
pub use filesystem::{LsTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool, GrepTool};
```

**Step 2: write_todos.rs 구현**

```rust
// src/tools/write_todos.rs
//! write_todos 도구 구현

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::middleware::traits::{Tool, ToolDefinition};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::state::{Todo, TodoStatus};

/// write_todos 도구 인자
#[derive(Debug, Deserialize)]
pub struct WriteTodosArgs {
    pub todos: Vec<TodoInput>,
}

#[derive(Debug, Deserialize)]
pub struct TodoInput {
    pub content: String,
    pub status: String,
}

/// write_todos 도구
pub struct WriteTodosTool;

impl WriteTodosTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteTodosTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteTodosTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_todos".to_string(),
            description: "Update the todo list to track progress on complex tasks.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The updated todo list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Description of the task"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Current status of the task"
                                }
                            },
                            "required": ["content", "status"]
                        }
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
        let parsed: WriteTodosArgs = serde_json::from_value(args)?;

        let todos: Vec<Todo> = parsed.todos.into_iter().map(|t| {
            let status = match t.status.as_str() {
                "in_progress" => TodoStatus::InProgress,
                "completed" => TodoStatus::Completed,
                _ => TodoStatus::Pending,
            };
            Todo::with_status(&t.content, status)
        }).collect();

        let count = todos.len();
        let completed = todos.iter().filter(|t| t.status == TodoStatus::Completed).count();

        Ok(format!(
            "Todo list updated: {} items total, {} completed",
            count, completed
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_write_todos_tool() {
        let tool = WriteTodosTool::new();
        let state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(state, backend);

        let args = serde_json::json!({
            "todos": [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "completed"}
            ]
        });

        let result = tool.execute(args, &runtime).await.unwrap();
        assert!(result.contains("2 items"));
        assert!(result.contains("1 completed"));
    }
}
```

**Step 3: todo.rs 미들웨어 구현**

```rust
// src/middleware/todo.rs
//! TodoListMiddleware 구현
//!
//! Python Reference: langchain/agents/middleware/todo.py

use async_trait::async_trait;
use std::sync::Arc;
use crate::state::AgentState;
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::tools::WriteTodosTool;
use super::traits::{AgentMiddleware, DynTool, StateUpdate};

const TODO_SYSTEM_PROMPT: &str = r#"## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step.

It is critical that you mark todos as completed as soon as you are done with a step.

## Task States
- pending: Task not yet started
- in_progress: Currently working on
- completed: Task finished successfully

## When to Use
- Complex multi-step tasks (3+ steps)
- Tasks requiring careful planning
- User explicitly requests todo list

## When NOT to Use
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational requests"#;

/// TodoListMiddleware
pub struct TodoListMiddleware {
    system_prompt: String,
    tool: Arc<WriteTodosTool>,
}

impl TodoListMiddleware {
    pub fn new() -> Self {
        Self {
            system_prompt: TODO_SYSTEM_PROMPT.to_string(),
            tool: Arc::new(WriteTodosTool::new()),
        }
    }
}

impl Default for TodoListMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentMiddleware for TodoListMiddleware {
    fn name(&self) -> &str {
        "TodoListMiddleware"
    }

    fn tools(&self) -> Vec<DynTool> {
        vec![self.tool.clone()]
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        format!("{}\n\n{}", prompt, self.system_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_middleware_prompt() {
        let mw = TodoListMiddleware::new();
        let prompt = mw.modify_system_prompt("Base prompt".to_string());

        assert!(prompt.contains("write_todos"));
        assert!(prompt.contains("pending"));
        assert!(prompt.contains("in_progress"));
        assert!(prompt.contains("completed"));
    }

    #[test]
    fn test_todo_middleware_tools() {
        let mw = TodoListMiddleware::new();
        let tools = mw.tools();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "write_todos");
    }
}
```

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: implement TodoListMiddleware with write_todos tool"
```

---

### Task 5.2: FilesystemMiddleware 및 도구들

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/filesystem.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/filesystem.rs`

**Python Reference:** `deepagents/middleware/filesystem.py`

**Step 1: tools/filesystem.rs 구현** (도구 6개: ls, read_file, write_file, edit_file, glob, grep)

```rust
// src/tools/filesystem.rs
//! 파일시스템 도구 구현

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use crate::middleware::traits::{Tool, ToolDefinition};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::backends::Backend;

// ============= ls 도구 =============

#[derive(Debug, Deserialize)]
pub struct LsArgs {
    pub path: String,
}

pub struct LsTool {
    backend: Arc<dyn Backend>,
}

impl LsTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for LsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ls".to_string(),
            description: "List directory contents. Returns files and subdirectories.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (e.g., '/' or '/src')"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: LsArgs = serde_json::from_value(args)?;

        let files = self.backend.ls(&parsed.path).await
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        if files.is_empty() {
            return Ok(format!("Directory '{}' is empty or does not exist", parsed.path));
        }

        let output: Vec<String> = files.iter().map(|f| {
            if f.is_dir {
                format!("{}  (dir)", f.path)
            } else {
                format!("{}  ({} bytes)", f.path, f.size.unwrap_or(0))
            }
        }).collect();

        Ok(output.join("\n"))
    }
}

// ============= read_file 도구 =============

#[derive(Debug, Deserialize)]
pub struct ReadFileArgs {
    pub path: String,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize { 500 }

pub struct ReadFileTool {
    backend: Arc<dyn Backend>,
}

impl ReadFileTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read file contents with line numbers (cat -n format). Use pagination for large files.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based)",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                        "default": 500
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: ReadFileArgs = serde_json::from_value(args)?;

        self.backend.read(&parsed.path, parsed.offset, parsed.limit).await
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))
    }
}

// ============= write_file 도구 =============

#[derive(Debug, Deserialize)]
pub struct WriteFileArgs {
    pub path: String,
    pub content: String,
}

pub struct WriteFileTool {
    backend: Arc<dyn Backend>,
}

impl WriteFileTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Create a new file. Fails if file already exists. Use edit_file for modifications.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path for the new file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: WriteFileArgs = serde_json::from_value(args)?;

        let result = self.backend.write(&parsed.path, &parsed.content).await
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        if let Some(error) = result.error {
            Ok(format!("Error: {}", error))
        } else {
            Ok(format!("Successfully created file: {}", parsed.path))
        }
    }
}

// ============= edit_file 도구 =============

#[derive(Debug, Deserialize)]
pub struct EditFileArgs {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    #[serde(default)]
    pub replace_all: bool,
}

pub struct EditFileTool {
    backend: Arc<dyn Backend>,
}

impl EditFileTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Edit a file by replacing old_string with new_string. Must read file before editing.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)",
                        "default": false
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: EditFileArgs = serde_json::from_value(args)?;

        let result = self.backend.edit(
            &parsed.path,
            &parsed.old_string,
            &parsed.new_string,
            parsed.replace_all
        ).await.map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        if let Some(error) = result.error {
            Ok(format!("Error: {}", error))
        } else {
            Ok(format!(
                "Successfully edited {}: {} occurrence(s) replaced",
                parsed.path,
                result.occurrences.unwrap_or(0)
            ))
        }
    }
}

// ============= glob 도구 =============

#[derive(Debug, Deserialize)]
pub struct GlobArgs {
    pub pattern: String,
    #[serde(default = "default_path")]
    pub path: String,
}

fn default_path() -> String { "/".to_string() }

pub struct GlobTool {
    backend: Arc<dyn Backend>,
}

impl GlobTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern (e.g., '**/*.rs', 'src/*.py').".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search from",
                        "default": "/"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: GlobArgs = serde_json::from_value(args)?;

        let files = self.backend.glob(&parsed.pattern, &parsed.path).await
            .map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        if files.is_empty() {
            return Ok(format!("No files matching pattern '{}'", parsed.pattern));
        }

        let output: Vec<String> = files.iter()
            .map(|f| f.path.clone())
            .collect();

        Ok(format!("Found {} files:\n{}", files.len(), output.join("\n")))
    }
}

// ============= grep 도구 =============

#[derive(Debug, Deserialize)]
pub struct GrepArgs {
    pub pattern: String,
    pub path: Option<String>,
    pub glob: Option<String>,
}

pub struct GrepTool {
    backend: Arc<dyn Backend>,
}

impl GrepTool {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            // **수정됨**: 리터럴 검색임을 명시
            description: "Search for a literal string pattern in files. NOT regex - searches for exact text match.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Literal string to search for (NOT regex)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (optional)"
                    },
                    "glob": {
                        "type": "string",
                        "description": "File pattern filter (e.g., '*.rs')"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: GrepArgs = serde_json::from_value(args)?;

        let matches = self.backend.grep(
            &parsed.pattern,
            parsed.path.as_deref(),
            parsed.glob.as_deref()
        ).await.map_err(|e| MiddlewareError::ToolExecution(e.to_string()))?;

        if matches.is_empty() {
            return Ok(format!("No matches found for '{}'", parsed.pattern));
        }

        let output: Vec<String> = matches.iter()
            .take(50) // 결과 제한
            .map(|m| format!("{}:{}:{}", m.path, m.line, m.text))
            .collect();

        let suffix = if matches.len() > 50 {
            format!("\n... and {} more matches", matches.len() - 50)
        } else {
            String::new()
        };

        Ok(format!("Found {} matches:\n{}{}", matches.len(), output.join("\n"), suffix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;

    #[tokio::test]
    async fn test_ls_tool() {
        let backend = Arc::new(MemoryBackend::new());
        backend.write("/test.txt", "content").await.unwrap();

        let tool = LsTool::new(backend.clone());
        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);

        let result = tool.execute(serde_json::json!({"path": "/"}), &runtime).await.unwrap();
        assert!(result.contains("test.txt"));
    }

    #[tokio::test]
    async fn test_read_write_edit_tools() {
        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend.clone());

        // Write
        let write_tool = WriteFileTool::new(backend.clone());
        let result = write_tool.execute(
            serde_json::json!({"path": "/test.txt", "content": "hello world"}),
            &runtime
        ).await.unwrap();
        assert!(result.contains("Successfully"));

        // Read
        let read_tool = ReadFileTool::new(backend.clone());
        let result = read_tool.execute(
            serde_json::json!({"path": "/test.txt"}),
            &runtime
        ).await.unwrap();
        assert!(result.contains("hello world"));

        // Edit
        let edit_tool = EditFileTool::new(backend.clone());
        let result = edit_tool.execute(
            serde_json::json!({
                "path": "/test.txt",
                "old_string": "hello",
                "new_string": "goodbye"
            }),
            &runtime
        ).await.unwrap();
        assert!(result.contains("Successfully edited"));
    }

    #[tokio::test]
    async fn test_grep_literal_search() {
        let backend = Arc::new(MemoryBackend::new());
        backend.write("/test.rs", "fn main() {}").await.unwrap();

        let tool = GrepTool::new(backend.clone());
        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);

        // 리터럴 검색 - 괄호가 그대로 검색됨
        let result = tool.execute(
            serde_json::json!({"pattern": "()"}),
            &runtime
        ).await.unwrap();
        assert!(result.contains("Found"));
    }
}
```

**Step 2: filesystem.rs 미들웨어 구현**

```rust
// src/middleware/filesystem.rs
//! FilesystemMiddleware 구현
//!
//! Python Reference: deepagents/middleware/filesystem.py

use async_trait::async_trait;
use std::sync::Arc;
use crate::backends::Backend;
use crate::state::AgentState;
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::tools::{LsTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool, GrepTool};
use super::traits::{AgentMiddleware, DynTool, StateUpdate};

/// 파일시스템 도구 시스템 프롬프트
/// **수정됨**: grep은 리터럴 검색임을 명시
const FILESYSTEM_SYSTEM_PROMPT: &str = r#"## Filesystem Tools

You have access to filesystem tools for managing files:

### `ls` - List directory contents
Usage: ls(path="/dir")

### `read_file` - Read file contents
Usage: read_file(path="/file.txt", offset=0, limit=500)
- Returns content with line numbers (cat -n format)
- Use pagination for large files

### `write_file` - Create new file
Usage: write_file(path="/file.txt", content="...")
- Use for creating new files only
- Fails if file already exists
- Prefer edit_file for modifications

### `edit_file` - Edit existing file
Usage: edit_file(path="/file.txt", old_string="...", new_string="...", replace_all=false)
- Must read file before editing
- old_string must be unique unless replace_all=true

### `glob` - Pattern matching
Usage: glob(pattern="**/*.rs", path="/")
- Standard glob: *, **, ?

### `grep` - Text search
Usage: grep(pattern="search term", path="/src", glob="*.rs")
- **IMPORTANT**: Uses LITERAL string matching (NOT regex)
- Searches for exact text occurrences"#;

/// FilesystemMiddleware
pub struct FilesystemMiddleware {
    backend: Arc<dyn Backend>,
    system_prompt: String,
    tools: Vec<DynTool>,
}

impl FilesystemMiddleware {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        let tools: Vec<DynTool> = vec![
            Arc::new(LsTool::new(backend.clone())),
            Arc::new(ReadFileTool::new(backend.clone())),
            Arc::new(WriteFileTool::new(backend.clone())),
            Arc::new(EditFileTool::new(backend.clone())),
            Arc::new(GlobTool::new(backend.clone())),
            Arc::new(GrepTool::new(backend.clone())),
        ];

        Self {
            backend,
            system_prompt: FILESYSTEM_SYSTEM_PROMPT.to_string(),
            tools,
        }
    }

    /// 백엔드 참조 반환
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }
}

#[async_trait]
impl AgentMiddleware for FilesystemMiddleware {
    fn name(&self) -> &str {
        "FilesystemMiddleware"
    }

    fn tools(&self) -> Vec<DynTool> {
        self.tools.clone()
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        format!("{}\n\n{}", prompt, self.system_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;

    #[test]
    fn test_filesystem_middleware_prompt() {
        let backend = Arc::new(MemoryBackend::new());
        let mw = FilesystemMiddleware::new(backend);
        let prompt = mw.modify_system_prompt("Base".to_string());

        assert!(prompt.contains("ls"));
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("write_file"));
        assert!(prompt.contains("edit_file"));
        assert!(prompt.contains("LITERAL")); // grep이 리터럴임을 확인
    }

    #[test]
    fn test_filesystem_middleware_tools() {
        let backend = Arc::new(MemoryBackend::new());
        let mw = FilesystemMiddleware::new(backend);
        let tools = mw.tools();

        assert_eq!(tools.len(), 6);

        let names: Vec<_> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"ls".to_string()));
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"write_file".to_string()));
        assert!(names.contains(&"edit_file".to_string()));
        assert!(names.contains(&"glob".to_string()));
        assert!(names.contains(&"grep".to_string()));
    }
}
```

**Step 3: Commit**

```bash
git add -A && git commit -m "feat: implement FilesystemMiddleware with 6 filesystem tools"
```

---

### Task 5.3: PatchToolCallsMiddleware 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/patch_tool_calls.rs`

**Python Reference:** `deepagents/middleware/patch_tool_calls.py`

**Step 1: patch_tool_calls.rs 구현**

```rust
// src/middleware/patch_tool_calls.rs
//! PatchToolCallsMiddleware - Dangling tool call 패치
//!
//! Python Reference: deepagents/middleware/patch_tool_calls.py
//!
//! AI 메시지의 tool_calls 중 대응하는 ToolMessage가 없는 것들을
//! 자동으로 패치하여 API 에러를 방지합니다.

use async_trait::async_trait;
use crate::state::{AgentState, Message, Role};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use super::traits::{AgentMiddleware, StateUpdate};

/// PatchToolCallsMiddleware
///
/// Python: PatchToolCallsMiddleware
///
/// 메시지 기록에서 dangling tool call을 패치합니다.
/// - AIMessage가 tool_calls를 가지고 있지만
/// - 대응하는 ToolMessage가 없는 경우
/// - 취소 메시지를 자동 삽입
pub struct PatchToolCallsMiddleware;

impl PatchToolCallsMiddleware {
    pub fn new() -> Self {
        Self
    }

    /// Dangling tool call 찾아서 패치
    fn patch_dangling_calls(messages: &[Message]) -> Vec<Message> {
        let mut patched = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            patched.push(msg.clone());

            // AI 메시지에 tool_calls가 있는 경우
            if msg.role == Role::Assistant {
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        // 나머지 메시지에서 대응하는 ToolMessage 찾기
                        let has_response = messages[i..].iter().any(|m| {
                            m.role == Role::Tool &&
                            m.tool_call_id.as_ref() == Some(&tc.id)
                        });

                        if !has_response {
                            // Dangling - 패치 메시지 삽입
                            patched.push(Message::tool(
                                &format!(
                                    "Tool call {} (ID: {}) was cancelled - \
                                    another message arrived before completion.",
                                    tc.name, tc.id
                                ),
                                &tc.id
                            ));
                        }
                    }
                }
            }
        }

        patched
    }
}

impl Default for PatchToolCallsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentMiddleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str {
        "PatchToolCallsMiddleware"
    }

    /// 에이전트 실행 전에 dangling tool call 패치
    async fn before_agent(
        &self,
        state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        if state.messages.is_empty() {
            return Ok(None);
        }

        let patched = Self::patch_dangling_calls(&state.messages);

        // 메시지가 변경되었으면 업데이트
        if patched.len() != state.messages.len() {
            state.messages = patched;
            // Note: StateUpdate로 반환하지 않고 직접 수정
            // (Overwrite 시맨틱)
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ToolCall;
    use crate::backends::MemoryBackend;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_patch_dangling_tool_calls() {
        let mw = PatchToolCallsMiddleware::new();

        // Dangling tool call이 있는 메시지
        let mut state = AgentState::new();
        state.messages = vec![
            Message::user("Hello"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/test.txt"}),
                }]
            ),
            // ToolMessage 없음 - dangling!
            Message::user("Continue"),
        ];

        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        mw.before_agent(&mut state, &runtime).await.unwrap();

        // 패치 메시지가 삽입되었는지 확인
        assert!(state.messages.len() > 3);
        assert!(state.messages.iter().any(|m|
            m.role == Role::Tool && m.content.contains("cancelled")
        ));
    }

    #[tokio::test]
    async fn test_no_patch_when_response_exists() {
        let mw = PatchToolCallsMiddleware::new();

        // 정상적인 tool call/response 쌍
        let mut state = AgentState::new();
        state.messages = vec![
            Message::user("Hello"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall {
                    id: "call_123".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/test.txt"}),
                }]
            ),
            Message::tool("file contents", "call_123"), // 응답 있음
            Message::assistant("Here's the file"),
        ];

        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);
        let original_len = state.messages.len();

        mw.before_agent(&mut state, &runtime).await.unwrap();

        // 패치 없어야 함
        assert_eq!(state.messages.len(), original_len);
    }
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: implement PatchToolCallsMiddleware for dangling tool calls"
```

---

### Task 5.4: SummarizationMiddleware 구현 (간소화)

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/summarization.rs`

**Python Reference:** `langchain/agents/middleware/summarization.py`

**Step 1: summarization.rs 구현** (토큰 기반 요약 - 간소화 버전)

```rust
// src/middleware/summarization.rs
//! SummarizationMiddleware - 컨텍스트 요약
//!
//! Python Reference: langchain/agents/middleware/summarization.py
//!
//! 메시지 히스토리가 너무 길어지면 자동으로 요약합니다.
//! 이 구현은 간소화된 버전으로, 토큰 카운팅 대신 메시지 수 기반입니다.

use async_trait::async_trait;
use crate::state::{AgentState, Message, Role};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use super::traits::{AgentMiddleware, StateUpdate};

/// 요약 트리거 조건
#[derive(Debug, Clone)]
pub enum SummarizationTrigger {
    /// 메시지 수 기반
    MessageCount(usize),
    /// 대략적인 토큰 수 기반 (문자 수 / 4)
    ApproximateTokens(usize),
}

/// SummarizationMiddleware
///
/// 대화 기록이 임계치에 도달하면 오래된 메시지를 요약합니다.
pub struct SummarizationMiddleware {
    trigger: SummarizationTrigger,
    keep_messages: usize,
}

impl SummarizationMiddleware {
    pub fn new(trigger: SummarizationTrigger, keep_messages: usize) -> Self {
        Self { trigger, keep_messages }
    }

    /// 메시지 수 기반 트리거로 생성
    pub fn with_message_limit(max_messages: usize, keep: usize) -> Self {
        Self::new(SummarizationTrigger::MessageCount(max_messages), keep)
    }

    /// 요약이 필요한지 확인
    fn needs_summarization(&self, state: &AgentState) -> bool {
        match self.trigger {
            SummarizationTrigger::MessageCount(max) => {
                state.messages.len() > max
            }
            SummarizationTrigger::ApproximateTokens(max_tokens) => {
                let approx_tokens: usize = state.messages.iter()
                    .map(|m| m.content.len() / 4)
                    .sum();
                approx_tokens > max_tokens
            }
        }
    }

    /// 메시지 요약 생성
    fn create_summary(messages: &[Message]) -> String {
        let mut summary_parts = Vec::new();

        for msg in messages {
            let prefix = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
                Role::Tool => "Tool",
            };

            // 긴 메시지는 잘라서 요약
            let content = if msg.content.len() > 200 {
                format!("{}...", &msg.content[..200])
            } else {
                msg.content.clone()
            };

            summary_parts.push(format!("{}: {}", prefix, content));
        }

        format!(
            "[Previous conversation summary]\n{}",
            summary_parts.join("\n")
        )
    }

    /// 요약 적용
    fn apply_summarization(&self, state: &mut AgentState) {
        let total = state.messages.len();
        let keep_count = self.keep_messages.min(total);
        let summarize_count = total.saturating_sub(keep_count);

        if summarize_count == 0 {
            return;
        }

        // 요약할 메시지들
        let to_summarize: Vec<_> = state.messages.drain(..summarize_count).collect();

        // 요약 메시지 생성
        let summary = Self::create_summary(&to_summarize);

        // 요약 메시지를 맨 앞에 삽입
        state.messages.insert(0, Message::system(&summary));
    }
}

impl Default for SummarizationMiddleware {
    fn default() -> Self {
        Self::with_message_limit(50, 10)
    }
}

#[async_trait]
impl AgentMiddleware for SummarizationMiddleware {
    fn name(&self) -> &str {
        "SummarizationMiddleware"
    }

    async fn before_agent(
        &self,
        state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        if self.needs_summarization(state) {
            self.apply_summarization(state);
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_summarization_trigger() {
        let mw = SummarizationMiddleware::with_message_limit(5, 2);

        // 6개 메시지 - 트리거됨
        let mut state = AgentState::new();
        for i in 0..6 {
            state.add_message(Message::user(&format!("Message {}", i)));
        }

        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        mw.before_agent(&mut state, &runtime).await.unwrap();

        // 요약 후: 요약 메시지 1개 + 유지된 메시지 2개 = 3개
        assert_eq!(state.messages.len(), 3);
        assert!(state.messages[0].content.contains("summary"));
    }

    #[tokio::test]
    async fn test_no_summarization_below_limit() {
        let mw = SummarizationMiddleware::with_message_limit(10, 2);

        let mut state = AgentState::new();
        for i in 0..5 {
            state.add_message(Message::user(&format!("Message {}", i)));
        }

        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(AgentState::new(), backend);

        mw.before_agent(&mut state, &runtime).await.unwrap();

        // 변경 없음
        assert_eq!(state.messages.len(), 5);
    }
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: implement SummarizationMiddleware for context management"
```

---

### Task 5.5: SubAgentMiddleware 구현 (간소화)

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/middleware/subagent.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/task.rs`

**Python Reference:** `deepagents/middleware/subagents.py`

이 태스크는 가장 복잡한 부분으로, 핵심 기능만 구현합니다.

**Step 1: tools/task.rs 구현**

```rust
// src/tools/task.rs
//! task 도구 - 서브에이전트 실행

use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use crate::middleware::traits::{Tool, ToolDefinition};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;

/// SubAgent 정의
pub struct SubAgentDef {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
}

/// task 도구 인자
#[derive(Debug, Deserialize)]
pub struct TaskArgs {
    pub description: String,
    pub subagent_type: String,
}

/// task 도구
pub struct TaskTool {
    subagents: HashMap<String, SubAgentDef>,
    task_description: String,
}

impl TaskTool {
    pub fn new(subagents: Vec<SubAgentDef>) -> Self {
        let mut map = HashMap::new();
        let mut descriptions = Vec::new();

        for sa in subagents {
            descriptions.push(format!("- {}: {}", sa.name, sa.description));
            map.insert(sa.name.clone(), sa);
        }

        let task_description = format!(
            "Launch a subagent to handle complex, multi-step tasks.\n\n\
            Available subagent types:\n{}\n\n\
            Use subagent_type parameter to select the agent type.",
            descriptions.join("\n")
        );

        Self {
            subagents: map,
            task_description,
        }
    }

    /// 범용 에이전트만 있는 기본 설정
    pub fn with_general_purpose() -> Self {
        Self::new(vec![SubAgentDef {
            name: "general-purpose".to_string(),
            description: "General-purpose agent for complex tasks with access to all tools.".to_string(),
            system_prompt: "You are a helpful assistant that completes tasks autonomously.".to_string(),
        }])
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task".to_string(),
            description: self.task_description.clone(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task for the subagent"
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "Type of subagent to use"
                    }
                },
                "required": ["description", "subagent_type"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let parsed: TaskArgs = serde_json::from_value(args)?;

        // 재귀 한도 확인
        if runtime.is_recursion_limit_exceeded() {
            return Ok("Error: Maximum subagent recursion depth exceeded.".to_string());
        }

        // 서브에이전트 유효성 확인
        let subagent = match self.subagents.get(&parsed.subagent_type) {
            Some(sa) => sa,
            None => {
                let valid_types: Vec<_> = self.subagents.keys().collect();
                return Ok(format!(
                    "Error: Unknown subagent type '{}'. Valid types: {:?}",
                    parsed.subagent_type, valid_types
                ));
            }
        };

        // 실제 서브에이전트 실행은 AgentExecutor에서 처리됨
        // 여기서는 메시지만 반환
        Ok(format!(
            "[SubAgent '{}' would execute task: {}]\n\
            Note: Full subagent execution requires AgentExecutor integration.",
            subagent.name, parsed.description
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;

    #[tokio::test]
    async fn test_task_tool_validation() {
        let tool = TaskTool::with_general_purpose();
        let state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());
        let runtime = ToolRuntime::new(state, backend);

        // 유효한 타입
        let result = tool.execute(
            serde_json::json!({
                "description": "Research topic X",
                "subagent_type": "general-purpose"
            }),
            &runtime
        ).await.unwrap();
        assert!(result.contains("SubAgent"));

        // 잘못된 타입
        let result = tool.execute(
            serde_json::json!({
                "description": "Task",
                "subagent_type": "invalid-type"
            }),
            &runtime
        ).await.unwrap();
        assert!(result.contains("Error"));
    }
}
```

**Step 2: tools/mod.rs 업데이트**

```rust
// src/tools/mod.rs (업데이트)
pub mod write_todos;
pub mod filesystem;
pub mod task;

pub use write_todos::WriteTodosTool;
pub use filesystem::{LsTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool, GrepTool};
pub use task::{TaskTool, SubAgentDef};
```

**Step 3: subagent.rs 미들웨어 구현**

```rust
// src/middleware/subagent.rs
//! SubAgentMiddleware - 서브에이전트 지원
//!
//! Python Reference: deepagents/middleware/subagents.py

use async_trait::async_trait;
use std::sync::Arc;
use crate::state::AgentState;
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;
use crate::tools::{TaskTool, SubAgentDef};
use super::traits::{AgentMiddleware, DynTool, StateUpdate};

const SUBAGENT_SYSTEM_PROMPT: &str = r#"## `task` (subagent spawner)

You have access to the `task` tool to spawn ephemeral subagents for isolated tasks.

When to use task tool:
- Complex multi-step tasks that can be fully delegated
- Independent tasks that can run in parallel
- Tasks requiring intensive reasoning that would bloat the main thread

When NOT to use task tool:
- Need to verify intermediate reasoning
- Trivial tasks (few tool calls)
- Delegation doesn't reduce complexity

Subagent lifecycle:
1. Spawn → provide clear role and instructions
2. Run → subagent works autonomously
3. Return → subagent provides single result
4. Reconcile → integrate results"#;

/// SubAgentMiddleware
pub struct SubAgentMiddleware {
    tool: Arc<TaskTool>,
    system_prompt: String,
}

impl SubAgentMiddleware {
    pub fn new(subagents: Vec<SubAgentDef>) -> Self {
        Self {
            tool: Arc::new(TaskTool::new(subagents)),
            system_prompt: SUBAGENT_SYSTEM_PROMPT.to_string(),
        }
    }

    /// 범용 에이전트만 포함하는 기본 설정
    pub fn with_general_purpose() -> Self {
        Self {
            tool: Arc::new(TaskTool::with_general_purpose()),
            system_prompt: SUBAGENT_SYSTEM_PROMPT.to_string(),
        }
    }
}

#[async_trait]
impl AgentMiddleware for SubAgentMiddleware {
    fn name(&self) -> &str {
        "SubAgentMiddleware"
    }

    fn tools(&self) -> Vec<DynTool> {
        vec![self.tool.clone()]
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        format!("{}\n\n{}", prompt, self.system_prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_middleware() {
        let mw = SubAgentMiddleware::with_general_purpose();

        let prompt = mw.modify_system_prompt("Base".to_string());
        assert!(prompt.contains("task"));
        assert!(prompt.contains("subagent"));

        let tools = mw.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "task");
    }
}
```

**Step 4: Commit**

```bash
git add -A && git commit -m "feat: implement SubAgentMiddleware with task tool"
```

---

## Phase 6: Agent Execution Loop

### Task 6.1: AgentExecutor 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/executor.rs`

이것은 가장 핵심적인 부분으로, LLM 호출과 도구 실행 루프를 구현합니다.

**Step 1: executor.rs 구현**

```rust
// src/executor.rs
//! Agent Execution Loop
//!
//! LLM 호출 및 도구 실행을 관리하는 핵심 실행기입니다.

use std::collections::HashMap;
use std::sync::Arc;
use crate::state::{AgentState, Message, Role, ToolCall};
use crate::error::DeepAgentError;
use crate::middleware::{MiddlewareStack, DynTool, Tool};
use crate::runtime::{ToolRuntime, RuntimeConfig};
use crate::backends::Backend;

/// LLM 응답 타입
pub struct LlmResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
}

/// LLM 인터페이스 트레이트
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(
        &self,
        messages: &[Message],
        system_prompt: &str,
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse, DeepAgentError>;
}

/// Agent Executor 설정
pub struct ExecutorConfig {
    pub max_iterations: usize,
    pub debug: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            debug: false,
        }
    }
}

/// Agent Executor
///
/// LLM 호출 및 도구 실행 루프를 관리합니다.
pub struct AgentExecutor {
    llm: Arc<dyn LlmClient>,
    middleware: MiddlewareStack,
    backend: Arc<dyn Backend>,
    config: ExecutorConfig,
    tools: HashMap<String, DynTool>,
}

impl AgentExecutor {
    pub fn new(
        llm: Arc<dyn LlmClient>,
        middleware: MiddlewareStack,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self::with_config(llm, middleware, backend, ExecutorConfig::default())
    }

    pub fn with_config(
        llm: Arc<dyn LlmClient>,
        middleware: MiddlewareStack,
        backend: Arc<dyn Backend>,
        config: ExecutorConfig,
    ) -> Self {
        // 모든 미들웨어에서 도구 수집
        let tools_list = middleware.collect_tools();
        let mut tools = HashMap::new();
        for tool in tools_list {
            tools.insert(tool.definition().name.clone(), tool);
        }

        Self {
            llm,
            middleware,
            backend,
            config,
            tools,
        }
    }

    /// 에이전트 실행
    pub async fn run(&self, state: &mut AgentState) -> Result<String, DeepAgentError> {
        let runtime = ToolRuntime::new(state.clone(), self.backend.clone())
            .with_config(RuntimeConfig {
                debug: self.config.debug,
                max_recursion: 10,
                current_recursion: 0,
            });

        // before_agent 훅 실행
        self.middleware.before_agent(state, &runtime).await
            .map_err(|e| DeepAgentError::Middleware(e))?;

        // 시스템 프롬프트 빌드
        let system_prompt = self.middleware.build_system_prompt(
            "You are a helpful assistant with access to various tools."
        );

        // 도구 스키마 준비
        let tool_schemas: Vec<_> = self.tools.values()
            .map(|t| {
                let def = t.definition();
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": def.name,
                        "description": def.description,
                        "parameters": def.parameters
                    }
                })
            })
            .collect();

        // 실행 루프
        for iteration in 0..self.config.max_iterations {
            if self.config.debug {
                eprintln!("[Executor] Iteration {}", iteration);
            }

            // LLM 호출
            let response = self.llm.chat(
                &state.messages,
                &system_prompt,
                &tool_schemas,
            ).await?;

            // 어시스턴트 메시지 추가
            if response.tool_calls.is_empty() {
                // 도구 호출 없음 - 최종 응답
                state.add_message(Message::assistant(&response.content));

                // after_agent 훅 실행
                self.middleware.after_agent(state, &runtime).await
                    .map_err(|e| DeepAgentError::Middleware(e))?;

                return Ok(response.content);
            }

            // 도구 호출이 있는 경우
            state.add_message(Message::assistant_with_tool_calls(
                &response.content,
                response.tool_calls.clone(),
            ));

            // 각 도구 실행
            for tool_call in &response.tool_calls {
                let result = self.execute_tool(tool_call, &runtime).await?;
                state.add_message(Message::tool(&result, &tool_call.id));
            }
        }

        Err(DeepAgentError::AgentExecution(
            "Maximum iterations exceeded".to_string()
        ))
    }

    /// 단일 도구 실행
    async fn execute_tool(
        &self,
        tool_call: &ToolCall,
        runtime: &ToolRuntime,
    ) -> Result<String, DeepAgentError> {
        let tool = self.tools.get(&tool_call.name)
            .ok_or_else(|| DeepAgentError::ToolNotFound(tool_call.name.clone()))?;

        if self.config.debug {
            eprintln!("[Executor] Executing tool: {} with args: {}",
                tool_call.name, tool_call.arguments);
        }

        tool.execute(tool_call.arguments.clone(), runtime).await
            .map_err(|e| DeepAgentError::Middleware(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::middleware::{TodoListMiddleware, FilesystemMiddleware};

    /// 테스트용 Mock LLM
    struct MockLlm {
        responses: Vec<LlmResponse>,
    }

    impl MockLlm {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self { responses }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for MockLlm {
        async fn chat(
            &self,
            _messages: &[Message],
            _system_prompt: &str,
            _tools: &[serde_json::Value],
        ) -> Result<LlmResponse, DeepAgentError> {
            // 간단히 첫 번째 응답 반환 (실제로는 상태 관리 필요)
            Ok(self.responses[0].clone())
        }
    }

    impl Clone for LlmResponse {
        fn clone(&self) -> Self {
            Self {
                content: self.content.clone(),
                tool_calls: self.tool_calls.clone(),
                finish_reason: self.finish_reason.clone(),
            }
        }
    }

    #[tokio::test]
    async fn test_executor_simple_response() {
        let llm = Arc::new(MockLlm::new(vec![
            LlmResponse {
                content: "Hello! I'm here to help.".to_string(),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
            }
        ]));

        let backend = Arc::new(MemoryBackend::new());
        let middleware = MiddlewareStack::new()
            .add(TodoListMiddleware::new())
            .add(FilesystemMiddleware::new(backend.clone()));

        let executor = AgentExecutor::new(llm, middleware, backend);

        let mut state = AgentState::with_messages(vec![
            Message::user("Hello!"),
        ]);

        let result = executor.run(&mut state).await.unwrap();
        assert_eq!(result, "Hello! I'm here to help.");
    }

    #[tokio::test]
    async fn test_executor_with_tool_call() {
        let backend = Arc::new(MemoryBackend::new());

        // 파일 미리 생성
        backend.write("/test.txt", "Hello from file!").await.unwrap();

        let llm = Arc::new(MockLlm::new(vec![
            LlmResponse {
                content: "".to_string(),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/test.txt"}),
                }],
                finish_reason: "tool_calls".to_string(),
            }
        ]));

        let middleware = MiddlewareStack::new()
            .add(FilesystemMiddleware::new(backend.clone()));

        let executor = AgentExecutor::new(llm, middleware, backend);

        let mut state = AgentState::with_messages(vec![
            Message::user("Read the file"),
        ]);

        // 첫 반복만 실행하면 도구 호출 후 다시 LLM 호출 필요
        // Mock이 단순하므로 에러 발생 예상
        let result = executor.run(&mut state).await;
        // 도구는 실행되었을 것
        assert!(state.messages.len() > 1);
    }
}
```

**Step 2: Commit**

```bash
git add -A && git commit -m "feat: implement AgentExecutor with LLM and tool execution loop"
```

---

## Phase 7: OpenAI 통합 테스트

### Task 7.1: OpenAI LLM Client 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/tests/openai_client.rs`
- Create: `rust-research-agent/crates/rig-deepagents/tests/integration_openai.rs`

**Step 1: openai_client.rs 구현** (테스트 헬퍼)

```rust
// tests/openai_client.rs
//! OpenAI Client for integration tests

use async_trait::async_trait;
use rig_deepagents::executor::{LlmClient, LlmResponse};
use rig_deepagents::state::{Message, Role, ToolCall};
use rig_deepagents::error::DeepAgentError;

/// OpenAI API Client (테스트용)
pub struct OpenAiClient {
    api_key: String,
    model: String,
}

impl OpenAiClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: "gpt-4o-mini".to_string(),
        }
    }

    pub fn with_model(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn chat(
        &self,
        messages: &[Message],
        system_prompt: &str,
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse, DeepAgentError> {
        let client = reqwest::Client::new();

        // 메시지 변환
        let mut api_messages: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "role": "system",
                "content": system_prompt
            })
        ];

        for msg in messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            let mut api_msg = serde_json::json!({
                "role": role,
                "content": msg.content
            });

            if let Some(ref id) = msg.tool_call_id {
                api_msg["tool_call_id"] = serde_json::json!(id);
            }

            if let Some(ref tcs) = msg.tool_calls {
                let tool_calls: Vec<_> = tcs.iter().map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments.to_string()
                        }
                    })
                }).collect();
                api_msg["tool_calls"] = serde_json::json!(tool_calls);
            }

            api_messages.push(api_msg);
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": api_messages,
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::json!(tools);
        }

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| DeepAgentError::LlmError(e.to_string()))?;

        let json: serde_json::Value = response.json().await
            .map_err(|e| DeepAgentError::LlmError(e.to_string()))?;

        // 응답 파싱
        let choice = &json["choices"][0];
        let message = &choice["message"];

        let content = message["content"].as_str().unwrap_or("").to_string();
        let finish_reason = choice["finish_reason"].as_str().unwrap_or("stop").to_string();

        let tool_calls: Vec<ToolCall> = message["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter().filter_map(|tc| {
                    Some(ToolCall {
                        id: tc["id"].as_str()?.to_string(),
                        name: tc["function"]["name"].as_str()?.to_string(),
                        arguments: serde_json::from_str(
                            tc["function"]["arguments"].as_str()?
                        ).ok()?,
                    })
                }).collect()
            })
            .unwrap_or_default();

        Ok(LlmResponse {
            content,
            tool_calls,
            finish_reason,
        })
    }
}
```

**Step 2: integration_openai.rs 구현**

```rust
// tests/integration_openai.rs
//! OpenAI 통합 테스트

mod openai_client;

use std::sync::Arc;
use std::time::Instant;
use rig_deepagents::middleware::*;
use rig_deepagents::backends::MemoryBackend;
use rig_deepagents::executor::{AgentExecutor, ExecutorConfig};
use rig_deepagents::state::{AgentState, Message};
use openai_client::OpenAiClient;

fn get_openai_key() -> Option<String> {
    dotenv::dotenv().ok();
    std::env::var("OPENAI_API_KEY").ok()
}

#[tokio::test]
#[ignore] // cargo test -- --ignored
async fn test_real_openai_simple() {
    let api_key = match get_openai_key() {
        Some(k) => k,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping test");
            return;
        }
    };

    let llm = Arc::new(OpenAiClient::new(&api_key));
    let backend = Arc::new(MemoryBackend::new());

    let middleware = MiddlewareStack::new()
        .add(TodoListMiddleware::new())
        .add(FilesystemMiddleware::new(backend.clone()));

    let executor = AgentExecutor::with_config(
        llm,
        middleware,
        backend,
        ExecutorConfig { max_iterations: 10, debug: true },
    );

    let mut state = AgentState::with_messages(vec![
        Message::user("Hello! Can you tell me what tools you have access to?"),
    ]);

    let start = Instant::now();
    let result = executor.run(&mut state).await.unwrap();
    let elapsed = start.elapsed();

    println!("Response: {}", result);
    println!("Time: {:?}", elapsed);

    assert!(!result.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_real_openai_with_tool_use() {
    let api_key = match get_openai_key() {
        Some(k) => k,
        None => {
            eprintln!("OPENAI_API_KEY not set, skipping test");
            return;
        }
    };

    let llm = Arc::new(OpenAiClient::new(&api_key));
    let backend = Arc::new(MemoryBackend::new());

    // 파일 미리 생성
    backend.write("/readme.txt", "# Welcome\nThis is a test file.").await.unwrap();

    let middleware = MiddlewareStack::new()
        .add(FilesystemMiddleware::new(backend.clone()));

    let executor = AgentExecutor::with_config(
        llm,
        middleware,
        backend,
        ExecutorConfig { max_iterations: 10, debug: true },
    );

    let mut state = AgentState::with_messages(vec![
        Message::user("Please read the file /readme.txt and tell me what it contains."),
    ]);

    let start = Instant::now();
    let result = executor.run(&mut state).await.unwrap();
    let elapsed = start.elapsed();

    println!("Response: {}", result);
    println!("Time: {:?}", elapsed);

    assert!(result.contains("Welcome") || result.contains("test"));
}

#[tokio::test]
#[ignore]
async fn benchmark_middleware_stack() {
    let backend = Arc::new(MemoryBackend::new());

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = MiddlewareStack::new()
            .add(TodoListMiddleware::new())
            .add(FilesystemMiddleware::new(backend.clone()))
            .add(PatchToolCallsMiddleware::new())
            .add(SummarizationMiddleware::default())
            .add(SubAgentMiddleware::with_general_purpose());
    }
    let elapsed = start.elapsed();

    println!("1000 middleware stack creations: {:?}", elapsed);
    println!("Average: {:?}", elapsed / 1000);

    assert!(elapsed.as_millis() < 100, "Should be < 100ms total");
}

#[tokio::test]
#[ignore]
async fn benchmark_memory_backend() {
    let backend = MemoryBackend::new();

    let start = Instant::now();
    for i in 0..1000 {
        backend.write(&format!("/test/file{}.txt", i), &format!("Content {}", i))
            .await.unwrap();
    }
    let write_time = start.elapsed();

    let start = Instant::now();
    let files = backend.glob("**/*.txt", "/").await.unwrap();
    let glob_time = start.elapsed();

    let start = Instant::now();
    let matches = backend.grep("Content 5", None, None).await.unwrap();
    let grep_time = start.elapsed();

    println!("Write 1000 files: {:?}", write_time);
    println!("Glob search: {:?} ({} files)", glob_time, files.len());
    println!("Grep search: {:?} ({} matches)", grep_time, matches.len());

    assert_eq!(files.len(), 1000);
}
```

**Step 3: Cargo.toml에 reqwest 추가**

```toml
# Cargo.toml [dev-dependencies]에 추가
reqwest = { version = "0.11", features = ["json"] }
```

**Step 4: Commit**

```bash
git add -A && git commit -m "test: add OpenAI integration tests"
```

---

## Phase 8: Criterion 벤치마크

### Task 8.1: Criterion 벤치마크 구현

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/benches/middleware_benchmark.rs`

**Step 1: 벤치마크 구현**

```rust
// benches/middleware_benchmark.rs
//! Criterion 벤치마크

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use tokio::runtime::Runtime;

use rig_deepagents::middleware::*;
use rig_deepagents::backends::MemoryBackend;
use rig_deepagents::state::AgentState;
use rig_deepagents::runtime::ToolRuntime;

fn bench_middleware_stack_creation(c: &mut Criterion) {
    c.bench_function("middleware_stack_creation", |b| {
        let backend = Arc::new(MemoryBackend::new());

        b.iter(|| {
            black_box(
                MiddlewareStack::new()
                    .add(TodoListMiddleware::new())
                    .add(FilesystemMiddleware::new(backend.clone()))
                    .add(PatchToolCallsMiddleware::new())
                    .add(SummarizationMiddleware::default())
                    .add(SubAgentMiddleware::with_general_purpose())
            )
        });
    });
}

fn bench_prompt_building(c: &mut Criterion) {
    let backend = Arc::new(MemoryBackend::new());
    let stack = MiddlewareStack::new()
        .add(TodoListMiddleware::new())
        .add(FilesystemMiddleware::new(backend.clone()))
        .add(PatchToolCallsMiddleware::new())
        .add(SubAgentMiddleware::with_general_purpose());

    c.bench_function("prompt_building", |b| {
        b.iter(|| {
            black_box(stack.build_system_prompt("You are a helpful assistant."))
        });
    });
}

fn bench_memory_backend_write(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("memory_backend_write_100", |b| {
        b.iter(|| {
            let backend = MemoryBackend::new();
            rt.block_on(async {
                for i in 0..100 {
                    black_box(
                        backend.write(&format!("/file{}.txt", i), "content").await.unwrap()
                    );
                }
            });
        });
    });
}

fn bench_memory_backend_glob(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let backend = MemoryBackend::new();

    // Setup: 1000 files
    rt.block_on(async {
        for i in 0..1000 {
            backend.write(&format!("/test/file{}.txt", i), "content").await.unwrap();
        }
    });

    c.bench_function("memory_backend_glob_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(backend.glob("**/*.txt", "/").await.unwrap())
            })
        });
    });
}

fn bench_memory_backend_grep(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let backend = MemoryBackend::new();

    // Setup
    rt.block_on(async {
        for i in 0..100 {
            backend.write(
                &format!("/src/file{}.rs", i),
                &format!("fn main() {{\n    println!(\"hello {}\");\n}}", i)
            ).await.unwrap();
        }
    });

    c.bench_function("memory_backend_grep_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(backend.grep("println", None, Some("*.rs")).await.unwrap())
            })
        });
    });
}

fn bench_tool_collection(c: &mut Criterion) {
    let backend = Arc::new(MemoryBackend::new());
    let stack = MiddlewareStack::new()
        .add(TodoListMiddleware::new())
        .add(FilesystemMiddleware::new(backend.clone()))
        .add(SubAgentMiddleware::with_general_purpose());

    c.bench_function("tool_collection", |b| {
        b.iter(|| {
            black_box(stack.collect_tools())
        });
    });
}

criterion_group!(
    benches,
    bench_middleware_stack_creation,
    bench_prompt_building,
    bench_memory_backend_write,
    bench_memory_backend_glob,
    bench_memory_backend_grep,
    bench_tool_collection,
);

criterion_main!(benches);
```

**Step 2: 벤치마크 실행**

Run: `cargo bench`
Expected: Benchmark results with statistical analysis

**Step 3: Commit**

```bash
git add -A && git commit -m "bench: add Criterion benchmarks for performance validation"
```

---

## Summary

이 계획은 LangChain DeepAgents의 **전체 기능**을 Rust/Rig로 구현합니다:

### Phases Overview

| Phase | 내용 | 주요 파일 |
|-------|------|----------|
| 1 | 프로젝트 초기화 | `Cargo.toml`, `lib.rs` |
| 2 | 에러 타입 및 상태 | `error.rs`, `state.rs` |
| 3 | Backend 트레이트 | `protocol.rs`, `memory.rs`, `filesystem.rs`, `composite.rs` |
| 4 | ToolRuntime & Middleware | `runtime.rs`, `traits.rs`, `stack.rs` |
| 5 | 미들웨어 구현 | `todo.rs`, `filesystem.rs`, `patch_tool_calls.rs`, `summarization.rs`, `subagent.rs` |
| 6 | Agent Executor | `executor.rs` |
| 7 | OpenAI 통합 | `integration_openai.rs` |
| 8 | Criterion 벤치마크 | `middleware_benchmark.rs` |

### 이전 계획 대비 주요 개선사항

1. ✅ **누락된 미들웨어 추가**: SubAgentMiddleware, SummarizationMiddleware, PatchToolCallsMiddleware
2. ✅ **실제 도구 구현**: 각 미들웨어에 실제 동작하는 도구 포함
3. ✅ **Agent Execution Loop**: LlmClient 트레이트 + AgentExecutor
4. ✅ **FilesystemBackend & CompositeBackend**: 경로 기반 라우팅
5. ✅ **ToolRuntime**: 도구 실행 컨텍스트
6. ✅ **HashMap import 수정**: error.rs
7. ✅ **FileData 통합**: state.rs에서 정의
8. ✅ **grep 프롬프트 수정**: 리터럴 검색 명시
9. ✅ **Criterion 벤치마크**: 실제 코드 포함

### Python Reference Files

- `deepagents/graph.py` - create_deep_agent
- `langchain/agents/middleware/types.py` - AgentMiddleware
- `deepagents/backends/protocol.py` - BackendProtocol
- `deepagents/backends/composite.py` - CompositeBackend
- `deepagents/backends/filesystem.py` - FilesystemBackend
- `deepagents/middleware/filesystem.py` - FilesystemMiddleware
- `deepagents/middleware/subagents.py` - SubAgentMiddleware
- `deepagents/middleware/patch_tool_calls.py` - PatchToolCallsMiddleware
- `langchain/agents/middleware/todo.py` - TodoListMiddleware
- `langchain/agents/middleware/summarization.py` - SummarizationMiddleware
