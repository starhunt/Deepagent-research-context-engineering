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

    #[error("Recursion limit exceeded: {0}")]
    RecursionLimit(String),
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
