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
            max_recursion: 100,  // Python 기본값에 가깝게 조정
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
}
