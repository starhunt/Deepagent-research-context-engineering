// src/backends/path_utils.rs
//! 경로 정규화 유틸리티
//!
//! 모든 백엔드에서 일관된 경로 처리를 위한 헬퍼 함수들
//!
//! # Security Model
//!
//! This module provides path normalization for the virtual filesystem backends.
//!
//! ## Security Guarantees
//! - Prevents `..` path traversal attacks
//! - Blocks home directory expansion (`~`)
//! - Normalizes multiple slashes (`//` → `/`) and trailing slashes
//! - Ensures proper directory boundary checking (`/dir` won't match `/directory`)
//!
//! ## Limitations
//! - Does not handle URL-encoded path segments (assumed pre-decoded)
//! - Case-sensitivity depends on the underlying backend (MemoryBackend is case-sensitive)
//! - Assumes UTF-8 valid paths (Rust strings are always valid UTF-8)
//!
//! ## Usage
//! All backend methods should use `normalize_path()` for path validation and
//! `is_under_path()` for directory containment checks.

use crate::error::BackendError;

/// 경로 정규화
/// - 앞에 `/` 추가
/// - 연속된 슬래시 제거 (`//` -> `/`)
/// - `.` 세그먼트 제거 (`/./` -> `/`)
/// - 후행 슬래시 제거 (루트 제외)
/// - 경로 순회 공격 방지 (..는 차단, .은 허용)
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
    fn test_normalize_path_dot_segments() {
        assert_eq!(normalize_path("/./file.txt").unwrap(), "/file.txt");
        assert_eq!(normalize_path("/dir/./sub/file.txt").unwrap(), "/dir/sub/file.txt");
        assert_eq!(normalize_path("./file.txt").unwrap(), "/file.txt");
        assert_eq!(normalize_path("/dir/.").unwrap(), "/dir");
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
