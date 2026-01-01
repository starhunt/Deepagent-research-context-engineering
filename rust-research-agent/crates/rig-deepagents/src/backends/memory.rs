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
use super::path_utils::{normalize_path, is_under_path};
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

    /// 라인 번호 포맷팅
    fn format_with_line_numbers(content: &str, offset: usize) -> String {
        content
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n")
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
        let path = normalize_path(path)?;
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
        let path = normalize_path(path)?;
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
        let path = normalize_path(path)?;
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
        let path = normalize_path(path)?;
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
        let base = normalize_path(base_path)?;
        let files = self.files.read().await;

        let glob_pattern = Pattern::new(pattern)
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();
        for (file_path, data) in files.iter() {
            // base_path 하위 파일만 검색 - use is_under_path for consistency
            if !is_under_path(file_path, &base) {
                continue;
            }

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

        let glob_pattern = glob_filter.map(Pattern::new).transpose()
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();

        for (file_path, data) in files.iter() {
            // Path filter - use is_under_path for proper boundary checking
            if let Some(p) = path {
                if !is_under_path(file_path, p) {
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
        let path = normalize_path(path)?;
        let files = self.files.read().await;
        Ok(files.contains_key(&path))
    }

    async fn delete(&self, path: &str) -> Result<(), BackendError> {
        let path = normalize_path(path)?;
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

    #[tokio::test]
    async fn test_memory_backend_glob_respects_base_path() {
        let backend = MemoryBackend::new();
        backend.write("/src/main.rs", "fn main()").await.unwrap();
        backend.write("/src/lib.rs", "pub mod").await.unwrap();
        backend.write("/docs/readme.md", "# Readme").await.unwrap();
        backend.write("/tests/test.rs", "test code").await.unwrap();

        // /src 하위에서만 검색해야 함
        let files = backend.glob("**/*.rs", "/src").await.unwrap();

        // /src 하위의 .rs 파일만 포함되어야 함
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.path.starts_with("/src")));

        // /tests/test.rs는 포함되면 안 됨
        assert!(!files.iter().any(|f| f.path.contains("/tests/")));
    }
}
