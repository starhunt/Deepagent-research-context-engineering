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
    ///
    /// # Security: 심볼릭 링크를 통한 루트 탈출 방지
    ///
    /// 부모 디렉토리가 심볼릭 링크인 경우 이를 canonicalize하여
    /// 루트 외부로의 탈출을 차단합니다.
    fn resolve_path(&self, path: &str) -> Result<PathBuf, BackendError> {
        if self.virtual_mode {
            // 경로 탐색 방지
            if path.contains("..") || path.starts_with("~") {
                return Err(BackendError::PathTraversal(path.to_string()));
            }

            let clean_path = path.trim_start_matches('/');

            // 루트 경로 ("/") 또는 빈 경로는 루트 디렉토리 자체
            if clean_path.is_empty() {
                return Ok(self.root.clone());
            }

            let target = self.root.join(clean_path);

            // 루트를 canonicalize
            let canonical_root = self.root.canonicalize()
                .unwrap_or_else(|_| self.root.clone());

            // 부모 디렉토리가 존재하면 canonicalize하여 symlink 해석
            // (루트 자체는 제외 - 루트 경로는 위에서 이미 처리됨)
            if let Some(parent) = target.parent() {
                if parent.exists() && parent != self.root {
                    let canonical_parent = parent.canonicalize()
                        .map_err(|e| BackendError::Io(e.to_string()))?;

                    // 부모가 루트 외부이면 차단
                    if !canonical_parent.starts_with(&canonical_root) {
                        return Err(BackendError::PathTraversal(
                            format!("Symlink escape detected: {}", path)
                        ));
                    }
                }
            }

            // 존재하는 경로는 canonicalize해서 최종 확인
            if target.exists() {
                let resolved = target.canonicalize()
                    .map_err(|e| BackendError::Io(e.to_string()))?;

                if !resolved.starts_with(&canonical_root) {
                    return Err(BackendError::PathTraversal(path.to_string()));
                }
            }

            Ok(target)
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

        // glob 패턴 정규화: **로 시작하지 않으면 **/ 접두사 추가
        let glob_pattern = glob_filter.map(|g| {
            let normalized = if g.starts_with("**/") || g.starts_with("/") {
                g.to_string()
            } else {
                format!("**/{}", g)
            };
            Pattern::new(&normalized)
        }).transpose()
            .map_err(|e| BackendError::Pattern(e.to_string()))?;

        let mut results = Vec::new();
        let walker = walkdir::WalkDir::new(&resolved);

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            // Glob filter - 전체 상대 경로에 대해 매칭
            if let Some(ref gp) = glob_pattern {
                let relative_path = entry.path()
                    .strip_prefix(&resolved)
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| entry.path().to_string_lossy().to_string());

                // 전체 경로 또는 파일명 매칭 (어느 하나라도 통과하면 OK)
                let filename = entry.file_name().to_string_lossy();
                if !gp.matches(&relative_path) && !gp.matches(&filename) {
                    continue;
                }
            }

            // 파일 읽기 (async)
            let content = match fs::read_to_string(entry.path()).await {
                Ok(c) => c,
                Err(e) => {
                    tracing::debug!(path = ?entry.path(), error = %e, "Skipping file in grep due to read error");
                    continue;
                }
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
    #[cfg(unix)]
    async fn test_filesystem_backend_symlink_traversal_prevention() {
        use std::os::unix::fs::symlink;
        use tempfile::tempdir;

        // 테스트용 디렉토리 생성
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();

        // 루트 외부에 파일 생성
        let outside_file = outside.path().join("secret.txt");
        std::fs::write(&outside_file, "secret data").unwrap();

        // 루트 내부에 외부를 가리키는 심볼릭 링크 생성
        let symlink_path = root.path().join("escape");
        symlink(outside.path(), &symlink_path).unwrap();

        let backend = FilesystemBackend::new(root.path());

        // 심볼릭 링크를 통한 읽기 시도 - 차단되어야 함
        let result = backend.read("/escape/secret.txt", 0, 100).await;
        assert!(result.is_err(), "Should block symlink traversal on read");

        // 심볼릭 링크를 통한 쓰기 시도 - 차단되어야 함
        let result = backend.write("/escape/malicious.txt", "pwned").await;
        assert!(
            result.is_err() || result.as_ref().map(|r| !r.is_ok()).unwrap_or(false),
            "Should block write via symlink"
        );
    }

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

    #[tokio::test]
    async fn test_filesystem_backend_grep_path_glob() {
        let temp = TempDir::new().unwrap();

        // 중첩 디렉토리에 파일 생성
        let src_dir = temp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("main.rs"), "fn main() { println!(\"hello\"); }").unwrap();
        std::fs::write(src_dir.join("lib.rs"), "pub fn hello() {}").unwrap();
        std::fs::write(temp.path().join("README.md"), "# Hello").unwrap();

        let backend = FilesystemBackend::new(temp.path());

        // **/*.rs 패턴으로 검색 - .rs 파일만 매칭
        let results = backend.grep("fn", None, Some("**/*.rs")).await.unwrap();

        assert!(!results.is_empty(), "Should find matches in .rs files");
        assert!(
            results.iter().all(|m| m.path.ends_with(".rs")),
            "All matches should be from .rs files"
        );

        // *.rs 패턴도 작동해야 함 (자동으로 **/ 접두사 추가)
        let results2 = backend.grep("fn", None, Some("*.rs")).await.unwrap();
        assert!(!results2.is_empty(), "*.rs pattern should also work");
    }
}
