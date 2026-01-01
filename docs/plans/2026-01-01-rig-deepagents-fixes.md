# Rig-DeepAgents Code Review 수정 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Codex 및 Code Review에서 식별된 HIGH/MEDIUM 이슈를 수정하여 Python DeepAgents와의 패리티를 확보하고 프로덕션 준비 상태를 달성한다.

**Architecture:**
- MemoryBackend와 AgentState.files의 이중 소스 문제는 `files_update` 반환 패턴을 통해 해결 (Python 패턴 유지)
- 경로 처리는 중앙화된 `normalize_path` 유틸리티로 통일
- CompositeBackend의 glob/write/edit 결과 집계 및 경로 복원 로직 보강

**Tech Stack:** Rust 1.75+, tokio, async-trait, glob crate

**검증 피드백 출처:**
- Codex CLI (gpt-5.2-codex): 55,431 토큰 분석
- Code Reviewer Subagent: 독립 검증

---

## Phase 1: HIGH Severity 수정 (필수)

### Task 1.1: MemoryBackend glob() base_path 필터 적용

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/memory.rs:191-213`
- Test: 동일 파일 하단 tests 모듈

**문제:** `glob()`이 `base_path`를 검증만 하고 실제 필터링에 사용하지 않아 모든 파일이 검색됨

**Step 1: 실패하는 테스트 작성**

`memory.rs` tests 모듈에 추가:

```rust
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
```

**Step 2: 테스트 실패 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_memory_backend_glob_respects_base_path`
Expected: FAIL - 현재 구현은 모든 .rs 파일(4개)을 반환

**Step 3: glob() 구현 수정**

`memory.rs` 191-213 라인을 다음으로 교체:

```rust
async fn glob(&self, pattern: &str, base_path: &str) -> Result<Vec<FileInfo>, BackendError> {
    let base = Self::validate_path(base_path)?;
    let files = self.files.read().await;

    let glob_pattern = Pattern::new(pattern)
        .map_err(|e| BackendError::Pattern(e.to_string()))?;

    let mut results = Vec::new();
    for (file_path, data) in files.iter() {
        // base_path 하위 파일만 검색
        let normalized_base = base.trim_end_matches('/');
        if !file_path.starts_with(normalized_base) {
            continue;
        }
        // base_path와 정확히 같거나 base_path/ 로 시작해야 함
        if file_path.len() > normalized_base.len()
            && !file_path[normalized_base.len()..].starts_with('/') {
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
```

**Step 4: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_memory_backend_glob_respects_base_path`
Expected: PASS

**Step 5: 전체 테스트 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 모든 테스트 PASS

---

### Task 1.2: CompositeBackend files_update 키 경로 복원

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/composite.rs:107-133`
- Test: 동일 파일 하단 tests 모듈

**문제:** routed backend에서 반환된 `files_update`의 키가 stripped 경로로 남아있어 상태 업데이트 시 경로 오류 발생

**Step 1: 실패하는 테스트 작성**

`composite.rs` tests 모듈에 추가:

```rust
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
```

**Step 2: 테스트 실패 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_composite_backend_write_files_update`
Expected: FAIL - 키가 `/notes.txt`로 되어있음

**Step 3: write() 및 edit() 구현 수정**

`composite.rs`의 write() 메서드 (107-116 라인)를 다음으로 교체:

```rust
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
```

`composite.rs`의 edit() 메서드 (118-133 라인)를 다음으로 교체:

```rust
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
```

**Step 4: 상단에 import 추가 확인**

`composite.rs` 상단에 `use crate::state::FileData;` 가 있는지 확인. 없으면 추가.

**Step 5: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_composite_backend`
Expected: 모든 composite 테스트 PASS

---

### Task 1.3: CompositeBackend glob() 집계 로직 추가

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/composite.rs:135-144`
- Test: 동일 파일 하단 tests 모듈

**문제:** glob()이 단일 백엔드만 쿼리하여 Python처럼 모든 백엔드 결과를 집계하지 않음

**Step 1: 실패하는 테스트 작성**

```rust
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
```

**Step 2: 테스트 실패 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_composite_backend_glob_aggregates`
Expected: FAIL - 현재는 default 백엔드만 검색

**Step 3: glob() 구현 수정**

`composite.rs`의 glob() 메서드 (135-144 라인)를 다음으로 교체:

```rust
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
```

**Step 4: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_composite_backend_glob`
Expected: PASS

---

### Task 1.4: AgentState.clone() extensions 경고 로그 추가

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/state.rs:169-180`
- Modify: `rust-research-agent/crates/rig-deepagents/Cargo.toml` (tracing 의존성 확인)

**문제:** clone() 시 extensions가 빈 HashMap으로 초기화되어 미들웨어 상태 손실

**Step 1: 현재 Clone 구현 확인**

`state.rs`의 Clone 구현 확인:

```rust
impl Clone for AgentState {
    fn clone(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            todos: self.todos.clone(),
            files: self.files.clone(),
            structured_response: self.structured_response.clone(),
            extensions: HashMap::new(),  // 데이터 손실!
        }
    }
}
```

**Step 2: tracing import 추가**

`state.rs` 상단에 추가:

```rust
use tracing::warn;
```

**Step 3: Clone 구현에 경고 로그 추가**

`state.rs`의 Clone 구현을 다음으로 교체:

```rust
impl Clone for AgentState {
    fn clone(&self) -> Self {
        // extensions가 비어있지 않으면 경고
        if !self.extensions.is_empty() {
            warn!(
                extension_count = self.extensions.len(),
                extension_keys = ?self.extensions.keys().collect::<Vec<_>>(),
                "AgentState.clone() called with non-empty extensions - extensions will be lost"
            );
        }

        Self {
            messages: self.messages.clone(),
            todos: self.todos.clone(),
            files: self.files.clone(),
            structured_response: self.structured_response.clone(),
            // extensions는 Box<dyn Any>를 clone할 수 없어서 빈 상태로 시작
            // 향후 Arc<RwLock<_>> 패턴으로 개선 고려
            extensions: HashMap::new(),
        }
    }
}
```

**Step 4: 컴파일 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo check`
Expected: 성공 (경고만 있을 수 있음)

---

## Phase 2: MEDIUM Severity 수정

### Task 2.1: 경로 정규화 유틸리티 추가 및 적용

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/backends/path_utils.rs`
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/mod.rs`
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/memory.rs`

**Step 1: path_utils.rs 생성**

```rust
// src/backends/path_utils.rs
//! 경로 정규화 유틸리티
//!
//! 모든 백엔드에서 일관된 경로 처리를 위한 헬퍼 함수들

use crate::error::BackendError;

/// 경로 정규화
/// - 앞에 `/` 추가
/// - 연속된 슬래시 제거 (`//` -> `/`)
/// - 후행 슬래시 제거 (루트 제외)
/// - 경로 순회 공격 방지
pub fn normalize_path(path: &str) -> Result<String, BackendError> {
    // 경로 순회 공격 방지
    if path.contains("..") || path.starts_with("~") {
        return Err(BackendError::PathTraversal(path.to_string()));
    }

    // 빈 경로는 루트로
    if path.is_empty() {
        return Ok("/".to_string());
    }

    // 연속된 슬래시 제거
    let parts: Vec<&str> = path.split('/')
        .filter(|p| !p.is_empty())
        .collect();

    if parts.is_empty() {
        return Ok("/".to_string());
    }

    Ok(format!("/{}", parts.join("/")))
}

/// 경로가 base_path 하위에 있는지 확인
/// `/dir`은 `/dir2`와 매칭되지 않음 (정확한 디렉토리 경계 확인)
pub fn is_under_path(path: &str, base_path: &str) -> bool {
    let normalized_base = base_path.trim_end_matches('/');

    if normalized_base.is_empty() || normalized_base == "/" {
        return true;  // 루트는 모든 경로 포함
    }

    if path == normalized_base {
        return true;  // 정확히 같은 경로
    }

    // base_path + "/" 로 시작해야 함
    path.starts_with(&format!("{}/", normalized_base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_basic() {
        assert_eq!(normalize_path("/test.txt").unwrap(), "/test.txt");
        assert_eq!(normalize_path("test.txt").unwrap(), "/test.txt");
        assert_eq!(normalize_path("/dir/file.txt").unwrap(), "/dir/file.txt");
    }

    #[test]
    fn test_normalize_path_double_slashes() {
        assert_eq!(normalize_path("/dir//file.txt").unwrap(), "/dir/file.txt");
        assert_eq!(normalize_path("//dir///file.txt").unwrap(), "/dir/file.txt");
    }

    #[test]
    fn test_normalize_path_trailing_slash() {
        assert_eq!(normalize_path("/dir/").unwrap(), "/dir");
        assert_eq!(normalize_path("/").unwrap(), "/");
    }

    #[test]
    fn test_normalize_path_traversal_attack() {
        assert!(normalize_path("../etc/passwd").is_err());
        assert!(normalize_path("/dir/../etc/passwd").is_err());
        assert!(normalize_path("~/.ssh/id_rsa").is_err());
    }

    #[test]
    fn test_is_under_path() {
        assert!(is_under_path("/dir/file.txt", "/dir"));
        assert!(is_under_path("/dir/sub/file.txt", "/dir"));
        assert!(is_under_path("/dir", "/dir"));
        assert!(is_under_path("/anything", "/"));

        // /dir 은 /dir2 에 포함되지 않음
        assert!(!is_under_path("/dir2/file.txt", "/dir"));
        assert!(!is_under_path("/directory/file.txt", "/dir"));
    }
}
```

**Step 2: mod.rs에 모듈 추가**

`backends/mod.rs` 수정:

```rust
// src/backends/mod.rs
//! 백엔드 모듈
//!
//! 파일시스템 추상화를 제공합니다.

pub mod protocol;
pub mod memory;
pub mod filesystem;
pub mod composite;
pub mod path_utils;

pub use protocol::{Backend, FileInfo, GrepMatch};
pub use memory::MemoryBackend;
pub use filesystem::FilesystemBackend;
pub use composite::CompositeBackend;
pub use path_utils::{normalize_path, is_under_path};
```

**Step 3: MemoryBackend에서 path_utils 사용**

`memory.rs` 상단 import 추가:

```rust
use super::path_utils::{normalize_path, is_under_path};
```

`memory.rs`의 `validate_path` 함수를 제거하고 `normalize_path` 사용으로 교체:

기존:
```rust
fn validate_path(path: &str) -> Result<String, BackendError> { ... }
```

모든 `Self::validate_path(path)?` 호출을 `normalize_path(path)?`로 교체.

**Step 4: 테스트 실행**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test path_utils`
Expected: PASS

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 모든 테스트 PASS

---

### Task 2.2: CompositeBackend 라우트 매칭 후행 슬래시 정규화

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/composite.rs:48-61`

**문제:** `/memories` 경로가 `/memories/` 라우트와 매칭되지 않음

**Step 1: 실패하는 테스트 작성**

```rust
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
```

**Step 2: get_backend_and_path() 수정**

`composite.rs`의 `get_backend_and_path` 메서드를 다음으로 교체:

```rust
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
```

**Step 3: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_composite_backend_route_matching`
Expected: PASS

---

### Task 2.3: FilesystemBackend grep에서 tokio::fs 사용

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/filesystem.rs:283-300`

**문제:** async 함수 내에서 sync `std::fs::read_to_string` 사용으로 런타임 블로킹

**Step 1: grep() 내부 파일 읽기를 async로 변경**

`filesystem.rs`의 grep 메서드에서 파일 읽기 부분을 수정.

기존:
```rust
let content = match std::fs::read_to_string(entry.path()) {
    Ok(c) => c,
    Err(_) => continue,
};
```

수정:
```rust
let content = match fs::read_to_string(entry.path()).await {
    Ok(c) => c,
    Err(e) => {
        tracing::debug!(path = ?entry.path(), error = %e, "Skipping file in grep due to read error");
        continue;
    }
};
```

**Step 2: filesystem.rs 상단에 tracing import 추가**

```rust
use tracing;
```

**Step 3: 컴파일 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo check`
Expected: 성공

---

### Task 2.4: max_recursion 설정 가능하게 변경

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/runtime.rs:41-48`

**Step 1: RuntimeConfig::new()에 기본값을 Python과 동일하게 수정**

```rust
impl RuntimeConfig {
    pub fn new() -> Self {
        Self {
            debug: false,
            max_recursion: 100,  // Python 기본값에 가깝게 조정 (1000은 너무 높음)
            current_recursion: 0,
        }
    }

    /// 커스텀 재귀 제한으로 생성
    pub fn with_max_recursion(max_recursion: usize) -> Self {
        Self {
            debug: false,
            max_recursion,
            current_recursion: 0,
        }
    }
}
```

**Step 2: 테스트 수정**

`runtime.rs`의 기존 테스트 `test_recursion_limit`를 수정:

```rust
#[test]
fn test_recursion_limit() {
    let state = AgentState::new();
    let backend = Arc::new(MemoryBackend::new());

    // 커스텀 재귀 제한 사용
    let config = RuntimeConfig::with_max_recursion(10);
    let mut runtime = ToolRuntime::new(state, backend).with_config(config);

    for _ in 0..10 {
        runtime = runtime.with_increased_recursion();
    }

    assert!(runtime.is_recursion_limit_exceeded());
}

#[test]
fn test_default_recursion_limit() {
    let state = AgentState::new();
    let backend = Arc::new(MemoryBackend::new());
    let runtime = ToolRuntime::new(state, backend);

    // 기본 제한은 100
    assert_eq!(runtime.config().max_recursion, 100);
}
```

**Step 3: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test runtime`
Expected: PASS

---

## Phase 3: 최종 검증

### Task 3.1: 전체 테스트 실행 및 검증

**Step 1: 전체 테스트 실행**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 모든 테스트 PASS (30개 이상)

**Step 2: Clippy 린트 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo clippy -- -D warnings`
Expected: 경고 없음 (또는 허용 가능한 경고만)

**Step 3: 문서 주석 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo doc --no-deps`
Expected: 문서 생성 성공

---

### Task 3.2: 변경사항 커밋

**Step 1: 변경 파일 확인**

Run: `git status`
Expected: 수정된 Rust 파일들 표시

**Step 2: 커밋**

```bash
git add rust-research-agent/crates/rig-deepagents/
git commit -m "fix: address HIGH and MEDIUM severity issues from Codex/Code Review

HIGH fixes:
- glob() now respects base_path filtering (security fix)
- CompositeBackend write/edit restores files_update key paths
- CompositeBackend glob aggregates results from all backends
- AgentState.clone() logs warning when extensions lost

MEDIUM fixes:
- Add centralized path_utils for consistent path normalization
- CompositeBackend route matching normalizes trailing slashes
- FilesystemBackend grep uses async tokio::fs
- RuntimeConfig.max_recursion increased to 100 (was 10)

Verified by: Codex CLI (gpt-5.2-codex), Code Reviewer subagent"
```

---

## 수정 우선순위 요약

| 우선순위 | Task | 예상 시간 | 위험도 |
|---------|------|----------|--------|
| 1 | Task 1.1: glob base_path 필터 | 5분 | Low |
| 2 | Task 1.2: files_update 키 복원 | 5분 | Low |
| 3 | Task 1.3: glob 집계 로직 | 10분 | Medium |
| 4 | Task 1.4: clone extensions 경고 | 5분 | Low |
| 5 | Task 2.1: path_utils 유틸리티 | 15분 | Low |
| 6 | Task 2.2: 라우트 매칭 정규화 | 5분 | Low |
| 7 | Task 2.3: async grep | 5분 | Low |
| 8 | Task 2.4: max_recursion 설정 | 5분 | Low |
| 9 | Task 3.1: 최종 검증 | 5분 | N/A |
| 10 | Task 3.2: 커밋 | 2분 | N/A |

**총 예상 시간:** 약 60분
