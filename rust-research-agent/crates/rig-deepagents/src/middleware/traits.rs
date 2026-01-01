// src/middleware/traits.rs
//! AgentMiddleware 트레이트 정의
//!
//! Python Reference: langchain/agents/middleware/types.py

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::state::{AgentState, Message, Todo, FileData};
use crate::error::MiddlewareError;
use crate::runtime::ToolRuntime;

/// 상태 업데이트 커맨드
/// Python: langgraph.types.Command
#[derive(Debug, Clone)]
pub enum StateUpdate {
    /// 메시지 추가
    AddMessages(Vec<Message>),
    /// Todo 업데이트
    SetTodos(Vec<Todo>),
    /// 파일 업데이트 (None = 삭제)
    UpdateFiles(HashMap<String, Option<FileData>>),
    /// 복합 업데이트
    Batch(Vec<StateUpdate>),
}

/// 도구 정의
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// 도구 인터페이스
#[async_trait]
pub trait Tool: Send + Sync {
    /// 도구 정의 반환
    fn definition(&self) -> ToolDefinition;

    /// 도구 실행
    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError>;
}

/// 동적 도구 타입
pub type DynTool = Arc<dyn Tool>;

/// AgentMiddleware 트레이트
///
/// Python Reference: AgentMiddleware(Generic[StateT, ContextT])
///
/// 핵심 기능:
/// - tools(): 에이전트에 자동 주입할 도구 목록 반환
/// - modify_system_prompt(): 시스템 프롬프트 수정 (체이닝)
/// - before_agent() / after_agent(): 라이프사이클 훅
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// 미들웨어 이름
    fn name(&self) -> &str;

    /// 이 미들웨어가 제공하는 도구 목록
    fn tools(&self) -> Vec<DynTool> {
        vec![]
    }

    /// 시스템 프롬프트 수정
    fn modify_system_prompt(&self, prompt: String) -> String {
        prompt
    }

    /// 에이전트 실행 전 훅
    /// Python: before_agent(self, state, runtime) -> dict | None
    async fn before_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }

    /// 에이전트 실행 후 훅
    /// Python: after_agent(self, state, runtime) -> dict | None
    async fn after_agent(
        &self,
        _state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mock_tool".to_string(),
                description: "A mock tool for testing".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            }
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _runtime: &ToolRuntime,
        ) -> Result<String, MiddlewareError> {
            Ok("mock result".to_string())
        }
    }

    struct MockMiddleware;

    #[async_trait]
    impl AgentMiddleware for MockMiddleware {
        fn name(&self) -> &str {
            "mock"
        }

        fn tools(&self) -> Vec<DynTool> {
            vec![Arc::new(MockTool)]
        }

        fn modify_system_prompt(&self, prompt: String) -> String {
            format!("{}\n\nMock middleware addition.", prompt)
        }
    }

    #[test]
    fn test_middleware_tools() {
        let middleware = MockMiddleware;
        let tools = middleware.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "mock_tool");
    }

    #[test]
    fn test_middleware_prompt_modification() {
        let middleware = MockMiddleware;
        let result = middleware.modify_system_prompt("Base prompt".to_string());
        assert!(result.contains("Base prompt"));
        assert!(result.contains("Mock middleware addition"));
    }
}
