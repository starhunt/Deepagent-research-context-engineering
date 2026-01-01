# Rig-DeepAgents Phase 5-6 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the Rig-DeepAgents Rust port by fixing security issues, aligning backend behaviors with Python, implementing all tools, and building the executor loop for full feature parity.

**Architecture:**
- Phase 1 fixes security vulnerabilities identified by Codex CLI (symlink traversal)
- Phase 2 aligns backend behaviors with Python reference (ls boundary, glob patterns, grep)
- Phase 3 implements tool layer connecting Backend trait methods to AgentMiddleware
- Phase 4 builds executor loop with LLM integration via Rig framework
- Phase 5 adds SubAgent system for task delegation

**Tech Stack:** Rust 1.75+, tokio, async-trait, rig-core, serde_json, tracing

**Verification Sources:**
- Codex CLI (gpt-5.2-codex): 164,210 tokens of analysis
- Qwen CLI: SubAgent/Tools gap analysis
- Python Reference: `deepagents_sourcecode/libs/deepagents/deepagents/backends/`

---

## Phase 1: Security Fixes (HIGH Priority)

### Task 1.1: Fix Symlink Path Traversal in FilesystemBackend

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/filesystem.rs:42-66`
- Test: 동일 파일 하단 tests 모듈

**문제:** `resolve_path`가 존재하지 않는 경로에 대해 canonicalize 없이 `target`을 반환하여, 부모 디렉토리가 루트 외부로의 심볼릭 링크인 경우 경로 탈출이 가능함.

**Step 1: 실패하는 테스트 작성**

`filesystem.rs` tests 모듈에 추가:

```rust
#[tokio::test]
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

    // 심볼릭 링크를 통한 접근 시도 - 차단되어야 함
    let result = backend.read("/escape/secret.txt", 0, 100).await;
    assert!(result.is_err(), "Should block symlink traversal");

    // 심볼릭 링크를 통한 쓰기 시도 - 차단되어야 함
    let result = backend.write("/escape/malicious.txt", "pwned").await;
    assert!(result.is_err() || result.unwrap().is_err(), "Should block write via symlink");
}
```

**Step 2: 테스트 실패 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_filesystem_backend_symlink_traversal`
Expected: FAIL - 현재는 symlink를 통한 접근이 허용됨

**Step 3: Cargo.toml에 tempfile 의존성 추가**

`Cargo.toml`의 `[dev-dependencies]` 섹션에:

```toml
[dev-dependencies]
tempfile = "3"
```

**Step 4: resolve_path 구현 수정**

`filesystem.rs`의 `resolve_path` 메서드를 다음으로 교체:

```rust
/// 경로 검증 및 해결
/// Security: 심볼릭 링크를 통한 루트 탈출 방지
fn resolve_path(&self, path: &str) -> Result<PathBuf, BackendError> {
    if self.virtual_mode {
        // 경로 탐색 방지
        if path.contains("..") || path.starts_with("~") {
            return Err(BackendError::PathTraversal(path.to_string()));
        }

        let clean_path = path.trim_start_matches('/');
        let target = self.root.join(clean_path);

        // 루트를 canonicalize
        let canonical_root = self.root.canonicalize()
            .unwrap_or_else(|_| self.root.clone());

        // 부모 디렉토리가 존재하면 canonicalize하여 symlink 해석
        if let Some(parent) = target.parent() {
            if parent.exists() {
                let canonical_parent = parent.canonicalize()
                    .map_err(|e| BackendError::IoError(e.to_string()))?;

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
                .map_err(|e| BackendError::IoError(e.to_string()))?;

            if !resolved.starts_with(&canonical_root) {
                return Err(BackendError::PathTraversal(path.to_string()));
            }
        }

        Ok(target)
    } else {
        Ok(PathBuf::from(path))
    }
}
```

**Step 5: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_filesystem_backend_symlink`
Expected: PASS

**Step 6: 전체 테스트 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 모든 테스트 PASS

---

## Phase 2: Backend Behavioral Fixes (MEDIUM Priority)

### Task 2.1: Fix MemoryBackend::ls Boundary Check

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/memory.rs:61-94`
- Test: 동일 파일 하단 tests 모듈

**문제:** `starts_with` 검사가 경계를 확인하지 않아 `/dir`이 `/directory`의 파일도 매칭함

**Step 1: 실패하는 테스트 작성**

```rust
#[tokio::test]
async fn test_memory_backend_ls_boundary_check() {
    let backend = MemoryBackend::new();
    backend.write("/dir/file.txt", "in dir").await.unwrap();
    backend.write("/directory/other.txt", "in directory").await.unwrap();

    // /dir 에서 ls 하면 /directory 파일은 보이지 않아야 함
    let files = backend.ls("/dir").await.unwrap();

    assert_eq!(files.len(), 1, "Should only find files under /dir");
    assert!(files[0].path.contains("/dir/"), "File should be under /dir");
    assert!(!files.iter().any(|f| f.path.contains("/directory")),
        "Should not include /directory files");
}
```

**Step 2: 테스트 실패 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_memory_backend_ls_boundary_check`
Expected: FAIL

**Step 3: ls() 구현 수정**

`memory.rs`의 ls() 메서드에서 매칭 로직 수정:

```rust
async fn ls(&self, path: &str) -> Result<Vec<FileInfo>, BackendError> {
    let path = normalize_path(path)?;
    let files = self.files.read().await;

    let normalized_prefix = if path == "/" {
        "/".to_string()
    } else {
        format!("{}/", path.trim_end_matches('/'))
    };

    let mut results = Vec::new();
    let mut dirs_seen = HashSet::new();

    for (file_path, data) in files.iter() {
        // 정확한 디렉토리 경계 확인
        let matches = if path == "/" {
            true
        } else {
            file_path.starts_with(&normalized_prefix) || file_path == &path
        };

        if matches {
            let prefix_to_strip = if path == "/" { "/" } else { &normalized_prefix };
            let relative = file_path.strip_prefix(prefix_to_strip)
                .unwrap_or(file_path.strip_prefix(&path).unwrap_or(file_path));

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
```

**Step 4: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_memory_backend_ls_boundary`
Expected: PASS

---

### Task 2.2: Fix normalize_path to Handle "." Segments

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/path_utils.rs:32-52`
- Test: 동일 파일 하단 tests 모듈

**문제:** `/./file`이 `/file`과 다르게 처리됨

**Step 1: 실패하는 테스트 작성**

```rust
#[test]
fn test_normalize_path_dot_segments() {
    assert_eq!(normalize_path("/./file.txt").unwrap(), "/file.txt");
    assert_eq!(normalize_path("/dir/./sub/file.txt").unwrap(), "/dir/sub/file.txt");
    assert_eq!(normalize_path("./file.txt").unwrap(), "/file.txt");
    assert_eq!(normalize_path("/dir/.").unwrap(), "/dir");
}
```

**Step 2: normalize_path 수정**

```rust
pub fn normalize_path(path: &str) -> Result<String, BackendError> {
    // 경로 순회 공격 방지 (..는 차단, .은 허용)
    if path.contains("..") || path.starts_with("~") {
        return Err(BackendError::PathTraversal(path.to_string()));
    }

    // 빈 경로는 루트로
    if path.is_empty() {
        return Ok("/".to_string());
    }

    // 연속된 슬래시 제거 및 "." 세그먼트 필터링
    let parts: Vec<&str> = path.split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();

    if parts.is_empty() {
        return Ok("/".to_string());
    }

    Ok(format!("/{}", parts.join("/")))
}
```

**Step 3: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_normalize_path_dot`
Expected: PASS

---

### Task 2.3: Fix FilesystemBackend::grep glob_filter to Match Full Path

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/filesystem.rs:280-310`

**문제:** `glob_filter`가 파일명만 매칭하여 `**/*.rs` 같은 패턴이 작동하지 않음

**Step 1: 실패하는 테스트 작성**

```rust
#[tokio::test]
async fn test_filesystem_backend_grep_path_glob() {
    use tempfile::tempdir;

    let root = tempdir().unwrap();

    // 중첩 디렉토리에 파일 생성
    let src_dir = root.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("main.rs"), "fn main() { println!(\"hello\"); }").unwrap();
    std::fs::write(src_dir.join("lib.rs"), "pub fn hello() {}").unwrap();
    std::fs::write(root.path().join("README.md"), "# Hello").unwrap();

    let backend = FilesystemBackend::new(root.path());

    // **/*.rs 패턴으로 검색
    let results = backend.grep("fn", None, Some("**/*.rs")).await.unwrap();

    assert!(!results.is_empty(), "Should find matches in .rs files");
    assert!(results.iter().all(|m| m.path.ends_with(".rs")),
        "All matches should be from .rs files");
}
```

**Step 2: grep 구현 수정**

```rust
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

    let glob_pattern = glob_filter.map(|g| {
        // 패턴이 **로 시작하면 그대로, 아니면 앞에 **/ 추가
        let normalized = if g.starts_with("**/") || g.starts_with("/") {
            g.to_string()
        } else {
            format!("**/{}", g)
        };
        glob::Pattern::new(&normalized)
    }).transpose()
        .map_err(|e| BackendError::Pattern(e.to_string()))?;

    let mut results = Vec::new();
    let walker = walkdir::WalkDir::new(&resolved);

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }

        // Glob filter - 전체 경로에 대해 매칭
        if let Some(ref gp) = glob_pattern {
            let relative_path = entry.path()
                .strip_prefix(&resolved)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| entry.path().to_string_lossy().to_string());

            if !gp.matches(&relative_path) && !gp.matches(&entry.file_name().to_string_lossy()) {
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
```

**Step 3: 테스트 통과 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test test_filesystem_backend_grep_path_glob`
Expected: PASS

---

### Task 2.4: Document Grep Literal vs Regex Design Decision

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/protocol.rs:60-70` (문서 주석)
- Modify: `rust-research-agent/crates/rig-deepagents/src/backends/memory.rs:1-10` (모듈 문서)

**Step 1: protocol.rs에 grep 설계 결정 문서화**

`protocol.rs`의 `grep` 메서드 문서에 추가:

```rust
/// 파일 내용에서 패턴 검색
///
/// # Design Decision: Literal Search
///
/// Rust 구현은 **리터럴 문자열 검색**을 사용합니다 (Python의 regex와 다름).
/// 이유:
/// - 보안: 정규식 패턴 주입 공격 방지
/// - 성능: 정규식 컴파일 오버헤드 없음
/// - 단순성: LLM 에이전트가 이해하기 쉬움
///
/// 정규식이 필요한 경우:
/// - `regex` crate를 사용하는 `grep_regex` 메서드 추가 고려
/// - 또는 Backend 구현체에서 regex 기능 확장
async fn grep(
    &self,
    pattern: &str,
    path: Option<&str>,
    glob_filter: Option<&str>,
) -> Result<Vec<GrepMatch>, BackendError>;
```

**Step 2: 커밋**

```bash
git add rust-research-agent/crates/rig-deepagents/
git commit -m "docs: document grep literal search design decision

Rust grep uses literal substring matching (not regex like Python).
This is intentional for security, performance, and simplicity."
```

---

## Phase 3: Tool Implementations (CRITICAL)

### Task 3.1: Create Tool Module Structure

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/read_file.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/write_file.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/edit_file.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/ls.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/glob.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/grep.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/write_todos.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/task.rs`
- Modify: `rust-research-agent/crates/rig-deepagents/src/tools/mod.rs`

**Step 1: mod.rs 업데이트**

```rust
//! Tool implementations for DeepAgents
//!
//! This module provides the 8 core tools auto-injected by middleware:
//! - File operations: read_file, write_file, edit_file, ls, glob, grep
//! - Planning: write_todos
//! - Delegation: task (SubAgent)

mod read_file;
mod write_file;
mod edit_file;
mod ls;
mod glob;
mod grep;
mod write_todos;
mod task;

pub use read_file::ReadFileTool;
pub use write_file::WriteFileTool;
pub use edit_file::EditFileTool;
pub use ls::LsTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use write_todos::WriteTodosTool;
pub use task::TaskTool;

use crate::middleware::DynTool;
use std::sync::Arc;

/// 모든 기본 도구 반환
pub fn default_tools() -> Vec<DynTool> {
    vec![
        Arc::new(ReadFileTool),
        Arc::new(WriteFileTool),
        Arc::new(EditFileTool),
        Arc::new(LsTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
        Arc::new(WriteTodosTool),
    ]
}

/// SubAgent task 도구 포함하여 모든 도구 반환
pub fn all_tools() -> Vec<DynTool> {
    let mut tools = default_tools();
    tools.push(Arc::new(TaskTool));
    tools
}
```

---

### Task 3.2: Implement ReadFileTool

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/read_file.rs`

**Step 1: 파일 생성**

```rust
//! read_file 도구 구현

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// read_file 도구
pub struct ReadFileTool;

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    file_path: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    2000
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read content from a file with optional line offset and limit.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed)",
                        "default": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                        "default": 2000
                    }
                },
                "required": ["file_path"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: ReadFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        runtime.backend()
            .read(&args.file_path, args.offset, args.limit)
            .await
            .map_err(|e| MiddlewareError::Backend(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_read_file_tool() {
        let backend = Arc::new(MemoryBackend::new());
        backend.write("/test.txt", "line1\nline2\nline3").await.unwrap();

        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);
        let tool = ReadFileTool;

        let result = tool.execute(
            serde_json::json!({"file_path": "/test.txt"}),
            &runtime,
        ).await.unwrap();

        assert!(result.contains("line1"));
        assert!(result.contains("line2"));
    }
}
```

---

### Task 3.3: Implement WriteFileTool

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/write_file.rs`

**Step 1: 파일 생성**

```rust
//! write_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// write_file 도구
pub struct WriteFileTool;

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    file_path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file, creating it if it doesn't exist.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: WriteFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .write(&args.file_path, &args.content)
            .await
            .map_err(|e| MiddlewareError::Backend(e))?;

        if result.is_ok() {
            Ok(format!("Successfully wrote to {}", args.file_path))
        } else {
            Err(MiddlewareError::ToolExecution(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}
```

---

### Task 3.4: Implement EditFileTool

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/edit_file.rs`

**Step 1: 파일 생성**

```rust
//! edit_file 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// edit_file 도구
pub struct EditFileTool;

#[derive(Debug, Deserialize)]
struct EditFileArgs {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Edit a file by replacing old_string with new_string.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The string to find and replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)",
                        "default": false
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: EditFileArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let result = runtime.backend()
            .edit(&args.file_path, &args.old_string, &args.new_string, args.replace_all)
            .await
            .map_err(|e| MiddlewareError::Backend(e))?;

        if result.is_ok() {
            let occurrences = result.occurrences.unwrap_or(1);
            Ok(format!("Replaced {} occurrence(s) in {}", occurrences, args.file_path))
        } else {
            Err(MiddlewareError::ToolExecution(
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            ))
        }
    }
}
```

---

### Task 3.5: Implement LsTool, GlobTool, GrepTool

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/ls.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/glob.rs`
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/grep.rs`

**Step 1: ls.rs 생성**

```rust
//! ls 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

pub struct LsTool;

#[derive(Debug, Deserialize)]
struct LsArgs {
    #[serde(default = "default_path")]
    path: String,
}

fn default_path() -> String {
    "/".to_string()
}

#[async_trait]
impl Tool for LsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ls".to_string(),
            description: "List files and directories at the given path.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list",
                        "default": "/"
                    }
                }
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: LsArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let files = runtime.backend()
            .ls(&args.path)
            .await
            .map_err(|e| MiddlewareError::Backend(e))?;

        let output: Vec<String> = files.iter()
            .map(|f| {
                if f.is_dir {
                    format!("{}/ (dir)", f.path)
                } else {
                    format!("{} ({} bytes)", f.path, f.size.unwrap_or(0))
                }
            })
            .collect();

        Ok(output.join("\n"))
    }
}
```

**Step 2: glob.rs 생성**

```rust
//! glob 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

pub struct GlobTool;

#[derive(Debug, Deserialize)]
struct GlobArgs {
    pattern: String,
    #[serde(default = "default_path")]
    base_path: String,
}

fn default_path() -> String {
    "/".to_string()
}

#[async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_string(),
            description: "Find files matching a glob pattern.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.rs', '*.txt')"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base path to search from",
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
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: GlobArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let files = runtime.backend()
            .glob(&args.pattern, &args.base_path)
            .await
            .map_err(|e| MiddlewareError::Backend(e))?;

        let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();

        if paths.is_empty() {
            Ok("No files found matching pattern.".to_string())
        } else {
            Ok(format!("Found {} files:\n{}", paths.len(), paths.join("\n")))
        }
    }
}
```

**Step 3: grep.rs 생성**

```rust
//! grep 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

pub struct GrepTool;

#[derive(Debug, Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob_filter: Option<String>,
}

#[async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_string(),
            description: "Search for a literal text pattern in files.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Literal text pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: /)"
                    },
                    "glob_filter": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '**/*.rs')"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: GrepArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let matches = runtime.backend()
            .grep(&args.pattern, args.path.as_deref(), args.glob_filter.as_deref())
            .await
            .map_err(|e| MiddlewareError::Backend(e))?;

        if matches.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            let output: Vec<String> = matches.iter()
                .map(|m| format!("{}:{}: {}", m.path, m.line_number, m.content))
                .collect();
            Ok(format!("Found {} matches:\n{}", matches.len(), output.join("\n")))
        }
    }
}
```

---

### Task 3.6: Implement WriteTodosTool

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/write_todos.rs`

**Step 1: 파일 생성**

```rust
//! write_todos 도구 구현

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;
use crate::state::{Todo, TodoStatus};

pub struct WriteTodosTool;

#[derive(Debug, Deserialize)]
struct TodoItem {
    content: String,
    #[serde(default)]
    status: String,
}

#[derive(Debug, Deserialize)]
struct WriteTodosArgs {
    todos: Vec<TodoItem>,
}

#[async_trait]
impl Tool for WriteTodosTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_todos".to_string(),
            description: "Update the todo list with new items.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The todo item content"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "default": "pending"
                                }
                            },
                            "required": ["content"]
                        },
                        "description": "List of todo items"
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
        let args: WriteTodosArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let todos: Vec<Todo> = args.todos.iter()
            .map(|t| {
                let status = match t.status.as_str() {
                    "in_progress" => TodoStatus::InProgress,
                    "completed" => TodoStatus::Completed,
                    _ => TodoStatus::Pending,
                };
                Todo::with_status(&t.content, status)
            })
            .collect();

        // Note: 실제 상태 업데이트는 미들웨어 레벨에서 처리
        // 여기서는 검증 및 포맷만 수행
        Ok(format!("Updated {} todo items", todos.len()))
    }
}
```

---

### Task 3.7: Implement TaskTool (SubAgent Delegation)

**Files:**
- Create: `rust-research-agent/crates/rig-deepagents/src/tools/task.rs`

**Step 1: 파일 생성**

```rust
//! task 도구 구현 (SubAgent 위임)

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

pub struct TaskTool;

#[derive(Debug, Deserialize)]
struct TaskArgs {
    subagent_type: String,
    prompt: String,
    #[serde(default)]
    description: Option<String>,
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task".to_string(),
            description: "Delegate a task to a sub-agent for specialized processing.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "The type of sub-agent to use (e.g., 'researcher', 'explorer', 'synthesizer')"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt for the sub-agent"
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the task"
                    }
                },
                "required": ["subagent_type", "prompt"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: TaskArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        // 재귀 한도 확인
        if runtime.is_recursion_limit_exceeded() {
            return Err(MiddlewareError::RecursionLimit(
                format!("Recursion limit exceeded. Cannot delegate to subagent '{}'", args.subagent_type)
            ));
        }

        // Note: 실제 SubAgent 실행은 executor에서 처리
        // 이 도구는 요청을 구조화하고 검증만 수행
        Ok(format!(
            "Task delegation requested:\n- Agent: {}\n- Description: {}\n- Prompt: {}",
            args.subagent_type,
            args.description.unwrap_or_else(|| "N/A".to_string()),
            args.prompt
        ))
    }
}
```

---

### Task 3.8: Update lib.rs Exports

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/lib.rs`

**Step 1: tools 모듈 export 추가**

```rust
// lib.rs 상단에 추가
pub mod tools;

// pub use 추가
pub use tools::{
    ReadFileTool, WriteFileTool, EditFileTool,
    LsTool, GlobTool, GrepTool,
    WriteTodosTool, TaskTool,
    default_tools, all_tools,
};
```

---

### Task 3.9: 전체 테스트 실행

**Step 1: 컴파일 확인**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo check`
Expected: 성공

**Step 2: 테스트 실행**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 모든 테스트 PASS

**Step 3: 커밋**

```bash
git add rust-research-agent/crates/rig-deepagents/
git commit -m "feat(tools): implement all 8 core tools

Tools implemented:
- read_file: Read file content with offset/limit
- write_file: Create/overwrite files
- edit_file: String replacement in files
- ls: List directory contents
- glob: Pattern-based file search
- grep: Content search (literal)
- write_todos: Todo list management
- task: SubAgent delegation request

All tools connect to Backend trait for operations."
```

---

## Phase 4: Executor Loop (CRITICAL)

### Task 4.1: Create Executor Module Structure

**Files:**
- Modify: `rust-research-agent/crates/rig-deepagents/src/executor.rs`

**Step 1: Executor 구조체 정의**

```rust
//! Agent executor - 메시지 처리 및 도구 실행 루프
//!
//! Python Reference: deepagents/graph.py

use std::sync::Arc;
use async_trait::async_trait;

use crate::backends::Backend;
use crate::error::{DeepAgentError, MiddlewareError};
use crate::middleware::{MiddlewareStack, DynTool, StateUpdate};
use crate::runtime::ToolRuntime;
use crate::state::{AgentState, Message, Role, ToolCall};

/// LLM 인터페이스 트레이트
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// 메시지로부터 응답 생성
    async fn generate(
        &self,
        messages: &[Message],
        tools: &[crate::middleware::ToolDefinition],
    ) -> Result<Message, DeepAgentError>;
}

/// Agent Executor
pub struct AgentExecutor<L: LLMProvider> {
    llm: L,
    middleware: MiddlewareStack,
    backend: Arc<dyn Backend>,
    max_iterations: usize,
}

impl<L: LLMProvider> AgentExecutor<L> {
    pub fn new(
        llm: L,
        middleware: MiddlewareStack,
        backend: Arc<dyn Backend>,
    ) -> Self {
        Self {
            llm,
            middleware,
            backend,
            max_iterations: 50,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// 에이전트 실행
    pub async fn run(&self, initial_state: AgentState) -> Result<AgentState, DeepAgentError> {
        let mut state = initial_state;
        let runtime = ToolRuntime::new(state.clone(), self.backend.clone());

        // Before hooks 실행
        if let Some(update) = self.middleware.before_agent(&mut state, &runtime).await? {
            self.apply_update(&mut state, update);
        }

        // 도구 수집
        let tools = self.middleware.collect_tools();
        let tool_definitions: Vec<_> = tools.iter()
            .map(|t| t.definition())
            .collect();

        // 메인 실행 루프
        for iteration in 0..self.max_iterations {
            tracing::debug!(iteration, "Agent iteration");

            // LLM 호출
            let response = self.llm.generate(&state.messages, &tool_definitions).await?;
            state.add_message(response.clone());

            // 도구 호출이 없으면 종료
            if !response.has_tool_calls() {
                tracing::debug!("No tool calls, finishing");
                break;
            }

            // 도구 호출 처리
            if let Some(tool_calls) = &response.tool_calls {
                for call in tool_calls {
                    let result = self.execute_tool_call(call, &tools, &runtime).await;
                    let tool_message = Message::tool(&result, &call.id);
                    state.add_message(tool_message);
                }
            }
        }

        // After hooks 실행
        if let Some(update) = self.middleware.after_agent(&mut state, &runtime).await? {
            self.apply_update(&mut state, update);
        }

        Ok(state)
    }

    /// 도구 호출 실행
    async fn execute_tool_call(
        &self,
        call: &ToolCall,
        tools: &[DynTool],
        runtime: &ToolRuntime,
    ) -> String {
        let tool = tools.iter().find(|t| t.definition().name == call.name);

        match tool {
            Some(t) => {
                match t.execute(call.arguments.clone(), runtime).await {
                    Ok(result) => result,
                    Err(e) => format!("Tool error: {}", e),
                }
            }
            None => format!("Unknown tool: {}", call.name),
        }
    }

    /// 상태 업데이트 적용
    fn apply_update(&self, state: &mut AgentState, update: StateUpdate) {
        match update {
            StateUpdate::AddMessages(msgs) => {
                for msg in msgs {
                    state.add_message(msg);
                }
            }
            StateUpdate::SetTodos(todos) => {
                state.todos = todos;
            }
            StateUpdate::UpdateFiles(files) => {
                for (path, data) in files {
                    if let Some(file_data) = data {
                        state.files.insert(path, file_data);
                    } else {
                        state.files.remove(&path);
                    }
                }
            }
            StateUpdate::Batch(updates) => {
                for u in updates {
                    self.apply_update(state, u);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;

    struct MockLLM;

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn generate(
            &self,
            _messages: &[Message],
            _tools: &[crate::middleware::ToolDefinition],
        ) -> Result<Message, DeepAgentError> {
            Ok(Message::assistant("Hello! I'm a mock assistant."))
        }
    }

    #[tokio::test]
    async fn test_executor_basic() {
        let llm = MockLLM;
        let backend = Arc::new(MemoryBackend::new());
        let middleware = MiddlewareStack::new();

        let executor = AgentExecutor::new(llm, middleware, backend);

        let initial_state = AgentState::with_messages(vec![
            Message::user("Hello!")
        ]);

        let result = executor.run(initial_state).await.unwrap();

        assert!(result.messages.len() >= 2);
        assert!(result.last_assistant_message().is_some());
    }
}
```

---

## Phase 5: 최종 검증 및 문서화

### Task 5.1: 전체 테스트 및 Clippy

**Step 1: 전체 테스트**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo test`
Expected: 40+ 테스트 PASS

**Step 2: Clippy**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo clippy -- -D warnings`
Expected: 경고 없음

**Step 3: 문서 생성**

Run: `source ~/.cargo/env && cd rust-research-agent/crates/rig-deepagents && cargo doc --no-deps --open`
Expected: 문서 생성 성공

---

### Task 5.2: 최종 커밋 및 태그

```bash
git add rust-research-agent/
git commit -m "feat: complete Phase 5-6 implementation

Phase 1 (Security):
- Fix symlink path traversal in FilesystemBackend

Phase 2 (Behavioral):
- Fix MemoryBackend ls boundary check
- Fix normalize_path to handle . segments
- Fix grep glob_filter to match full paths
- Document grep literal vs regex decision

Phase 3 (Tools):
- Implement all 8 core tools
- Add default_tools() and all_tools() helpers

Phase 4 (Executor):
- Add LLMProvider trait
- Implement AgentExecutor with tool execution loop
- Add state update application

Test coverage: 45+ tests passing
All Clippy warnings resolved"

git tag v0.2.0-alpha
```

---

## 수정 우선순위 요약

| 우선순위 | Phase | Task 수 | 예상 시간 |
|----------|-------|---------|-----------|
| 🔴 HIGH | Phase 1 (Security) | 1 | 30분 |
| 🟡 MEDIUM | Phase 2 (Behavioral) | 4 | 1시간 |
| 🔴 CRITICAL | Phase 3 (Tools) | 9 | 2시간 |
| 🔴 CRITICAL | Phase 4 (Executor) | 1 | 1시간 |
| 🟢 LOW | Phase 5 (Verification) | 2 | 30분 |

**총 예상 시간:** 약 5시간

---

## 의존성 그래프

```
Phase 1 (Security)
    ↓
Phase 2 (Behavioral) ──→ Phase 3 (Tools)
                              ↓
                         Phase 4 (Executor)
                              ↓
                         Phase 5 (Verification)
```

Phase 1과 2는 독립적으로 진행 가능. Phase 3는 2 완료 후, Phase 4는 3 완료 후 진행.
