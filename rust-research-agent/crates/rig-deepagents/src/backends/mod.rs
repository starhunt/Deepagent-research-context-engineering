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
