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
