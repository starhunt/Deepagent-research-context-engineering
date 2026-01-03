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
        // 경로 정규화 (후행 슬래시 제거)
        let normalized_path = path.trim_end_matches('/');

        for route in &self.routes {
            let route_prefix = route.prefix.trim_end_matches('/');

            // 정확히 일치하거나 route_prefix/ 로 시작하는 경우
            if normalized_path == route_prefix ||
               normalized_path.starts_with(&format!("{}/", route_prefix)) {
                let suffix = if normalized_path == route_prefix {
                    ""
                } else {
                    &normalized_path[route_prefix.len()..]
                };

                let stripped = if suffix.is_empty() {
                    "/".to_string()
                } else {
                    suffix.to_string()
                };
                return (route.backend.clone(), stripped);
            }
        }
        (self.default.clone(), path.to_string())
    }

    /// 결과 경로에 접두사 복원
    fn restore_prefix(&self, path: &str, original_path: &str) -> String {
        let normalized_original = original_path.trim_end_matches('/');
        for route in &self.routes {
            let normalized_prefix = route.prefix.trim_end_matches('/');
            // Check both exact match and prefix match with boundary
            if normalized_original == normalized_prefix ||
               normalized_original.starts_with(&format!("{}/", normalized_prefix)) {
                return format!("{}{}", normalized_prefix, path);
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

        // 경로 복원
        if result.path.is_some() {
            result.path = Some(path.to_string());
        }

        // files_update 키도 복원
        if let Some(ref mut files_update) = result.files_update {
            let restored: std::collections::HashMap<String, crate::state::FileData> = files_update
                .drain()
                .map(|(k, v)| (self.restore_prefix(&k, path), v))
                .collect();
            *files_update = restored;
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

        // 경로 복원
        if result.path.is_some() {
            result.path = Some(path.to_string());
        }

        // files_update 키도 복원
        if let Some(ref mut files_update) = result.files_update {
            let restored: std::collections::HashMap<String, crate::state::FileData> = files_update
                .drain()
                .map(|(k, v)| (self.restore_prefix(&k, path), v))
                .collect();
            *files_update = restored;
        }

        Ok(result)
    }

    async fn glob(&self, pattern: &str, base_path: &str) -> Result<Vec<FileInfo>, BackendError> {
        // 특정 라우트 경로인 경우 해당 백엔드만 검색
        for route in &self.routes {
            let route_prefix = route.prefix.trim_end_matches('/');
            if base_path.starts_with(route_prefix) &&
               (base_path.len() == route_prefix.len() || base_path[route_prefix.len()..].starts_with('/')) {
                let (backend, stripped) = self.get_backend_and_path(base_path);
                let mut results = backend.glob(pattern, &stripped).await?;

                for info in &mut results {
                    info.path = self.restore_prefix(&info.path, base_path);
                }
                return Ok(results);
            }
        }

        // 루트 또는 라우트되지 않은 경로 - 모든 백엔드에서 집계
        let mut all_results = self.default.glob(pattern, base_path).await?;

        for route in &self.routes {
            let mut route_results = route.backend.glob(pattern, "/").await?;
            for info in &mut route_results {
                let prefix = route.prefix.trim_end_matches('/');
                info.path = format!("{}{}", prefix, info.path);
            }
            all_results.extend(route_results);
        }

        all_results.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(all_results)
    }

    async fn grep(
        &self,
        pattern: &str,
        path: Option<&str>,
        glob_filter: Option<&str>,
    ) -> Result<Vec<GrepMatch>, BackendError> {
        let search_path = path.unwrap_or("/");

        // Use get_backend_and_path for consistent routing logic
        let (backend, stripped) = self.get_backend_and_path(search_path);

        // Check if this is a routed path (not default backend)
        let is_routed = !std::ptr::eq(
            Arc::as_ptr(&backend) as *const (),
            Arc::as_ptr(&self.default) as *const ()
        );

        if is_routed {
            // 특정 경로가 라우트에 매칭되면 해당 백엔드만 검색
            let mut results = backend.grep(pattern, Some(&stripped), glob_filter).await?;

            for m in &mut results {
                m.path = self.restore_prefix(&m.path, search_path);
            }

            return Ok(results);
        }

        // 전체 검색 (루트 또는 default backend 경로)
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

    #[tokio::test]
    async fn test_composite_backend_write_files_update_path_restoration() {
        let default = Arc::new(MemoryBackend::new());
        let memories = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/memories/", memories.clone());

        // routed backend에 파일 쓰기
        let result = composite.write("/memories/notes.txt", "my notes").await.unwrap();

        assert!(result.is_ok());
        assert_eq!(result.path, Some("/memories/notes.txt".to_string()));

        // files_update가 있다면 키도 복원되어야 함
        if let Some(files_update) = &result.files_update {
            // 키가 /memories/notes.txt 여야 함 (stripped된 /notes.txt가 아님)
            assert!(files_update.contains_key("/memories/notes.txt"),
                "files_update key should be '/memories/notes.txt', got keys: {:?}",
                files_update.keys().collect::<Vec<_>>());
        }
    }

    #[tokio::test]
    async fn test_composite_backend_edit_files_update_path_restoration() {
        let default = Arc::new(MemoryBackend::new());
        let memories = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/memories/", memories.clone());

        // 먼저 파일 생성
        composite.write("/memories/notes.txt", "hello world").await.unwrap();

        // 편집
        let result = composite.edit("/memories/notes.txt", "hello", "goodbye", false).await.unwrap();

        assert!(result.is_ok());

        // files_update가 있다면 키도 복원되어야 함
        if let Some(files_update) = &result.files_update {
            assert!(files_update.contains_key("/memories/notes.txt"),
                "files_update key should be '/memories/notes.txt', got keys: {:?}",
                files_update.keys().collect::<Vec<_>>());
        }
    }

    #[tokio::test]
    async fn test_composite_backend_glob_aggregates_all_backends() {
        let default = Arc::new(MemoryBackend::new());
        let docs = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/docs/", docs.clone());

        // 각 백엔드에 파일 생성
        composite.write("/src/main.rs", "fn main()").await.unwrap();
        composite.write("/docs/guide.md", "# Guide").await.unwrap();
        composite.write("/docs/api.md", "# API").await.unwrap();

        // 루트에서 모든 .md 파일 검색 - 모든 백엔드에서 집계해야 함
        let files = composite.glob("**/*.md", "/").await.unwrap();

        // docs 백엔드의 2개 파일이 모두 포함되어야 함
        assert_eq!(files.len(), 2, "Expected 2 .md files, got: {:?}", files);
        assert!(files.iter().any(|f| f.path.contains("guide.md")));
        assert!(files.iter().any(|f| f.path.contains("api.md")));
    }

    #[tokio::test]
    async fn test_composite_backend_route_matching_without_trailing_slash() {
        let default = Arc::new(MemoryBackend::new());
        let memories = Arc::new(MemoryBackend::new());

        let composite = CompositeBackend::new(default.clone())
            .with_route("/memories/", memories.clone());

        // 후행 슬래시 없이도 라우트되어야 함
        composite.write("/memories/notes.txt", "my notes").await.unwrap();

        // /memories 경로로 읽기 (후행 슬래시 없음)
        let files = composite.ls("/memories").await.unwrap();
        assert!(!files.is_empty(), "Should find files under /memories route");
    }
}
