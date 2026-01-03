//! HumanInTheLoopMiddleware - 인간 승인 인터럽트
//!
//! 특정 도구 호출에 대해 인간 승인을 요청하는 미들웨어입니다.
//! LLM 응답에 승인이 필요한 도구 호출이 포함되면 실행을 인터럽트합니다.
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_deepagents::middleware::{HumanInTheLoopMiddleware, InterruptOnConfig, Decision};
//! use std::collections::HashMap;
//!
//! // shell 도구에 대해 인터럽트 설정
//! let mut interrupt_on = HashMap::new();
//! interrupt_on.insert("shell".to_string(), InterruptOnConfig::default());
//!
//! let middleware = HumanInTheLoopMiddleware::new(interrupt_on);
//! ```
//!
//! # Interrupt Flow
//!
//! 1. LLM이 응답 생성
//! 2. `after_model` 훅에서 tool_calls 검사
//! 3. 승인 필요한 도구가 있으면 `ModelControl::Interrupt` 반환
//! 4. AgentExecutor가 `DeepAgentError::Interrupt` 반환
//! 5. 외부 시스템이 사용자 결정 수집
//! 6. 결정과 함께 실행 재개

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::MiddlewareError;
use crate::middleware::{
    AgentMiddleware, ModelControl, ModelResponse,
    InterruptRequest, ActionRequest, ReviewConfig, Decision,
};
use crate::runtime::ToolRuntime;
use crate::state::AgentState;

/// 도구별 인터럽트 설정
#[derive(Debug, Clone)]
pub struct InterruptOnConfig {
    /// 인터럽트 활성화 여부
    pub enabled: bool,
    /// 허용되는 결정 유형
    pub allowed_decisions: Vec<Decision>,
    /// 설명 생성 함수 (선택)
    description_fn: Option<fn(&serde_json::Value) -> String>,
}

impl Default for InterruptOnConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_decisions: vec![Decision::Approve, Decision::Reject],
            description_fn: None,
        }
    }
}

impl InterruptOnConfig {
    /// 새 설정 생성
    pub fn new(enabled: bool) -> Self {
        Self { enabled, ..Default::default() }
    }

    /// 허용 결정 설정
    pub fn with_decisions(mut self, decisions: Vec<Decision>) -> Self {
        self.allowed_decisions = decisions;
        self
    }

    /// 모든 결정 허용 (Approve, Reject, Edit)
    pub fn allow_all() -> Self {
        Self {
            enabled: true,
            allowed_decisions: vec![Decision::Approve, Decision::Reject, Decision::Edit],
            description_fn: None,
        }
    }

    /// 승인/거부만 허용
    pub fn approve_reject_only() -> Self {
        Self::default()
    }

    /// 설명 생성 함수 설정
    pub fn with_description_fn(mut self, f: fn(&serde_json::Value) -> String) -> Self {
        self.description_fn = Some(f);
        self
    }
}

/// 인간 승인을 요청하는 미들웨어
///
/// 특정 도구 호출에 대해 실행을 인터럽트하고 인간 승인을 요청합니다.
pub struct HumanInTheLoopMiddleware {
    /// 도구별 인터럽트 설정
    interrupt_on: HashMap<String, InterruptOnConfig>,
}

impl HumanInTheLoopMiddleware {
    /// 새 미들웨어 생성
    pub fn new(interrupt_on: HashMap<String, InterruptOnConfig>) -> Self {
        Self { interrupt_on }
    }

    /// bool 맵으로부터 생성 (tool_name -> interrupt?)
    pub fn from_bool_map(map: HashMap<String, bool>) -> Self {
        let interrupt_on = map
            .into_iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(name, _)| (name, InterruptOnConfig::default()))
            .collect();
        Self { interrupt_on }
    }

    /// 단일 도구에 대한 인터럽트 설정
    pub fn for_tool(tool_name: impl Into<String>) -> Self {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert(tool_name.into(), InterruptOnConfig::default());
        Self { interrupt_on }
    }

    /// 여러 도구에 대해 동일 설정
    pub fn for_tools(tool_names: Vec<String>, config: InterruptOnConfig) -> Self {
        let interrupt_on = tool_names
            .into_iter()
            .map(|name| (name, config.clone()))
            .collect();
        Self { interrupt_on }
    }

    /// 도구가 인터럽트 필요한지 확인
    fn should_interrupt(&self, tool_name: &str) -> Option<&InterruptOnConfig> {
        self.interrupt_on
            .get(tool_name)
            .filter(|c| c.enabled)
    }

    /// ActionRequest 생성
    fn create_action_request(
        &self,
        id: &str,
        name: &str,
        args: &serde_json::Value,
        config: &InterruptOnConfig,
    ) -> ActionRequest {
        let description = config.description_fn
            .map(|f| f(args))
            .unwrap_or_else(|| {
                format!(
                    "Tool '{}' requires approval.\n\nArguments:\n{}",
                    name,
                    serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string())
                )
            });

        ActionRequest::new(id, name, args.clone())
            .with_description(description)
    }
}

#[async_trait]
impl AgentMiddleware for HumanInTheLoopMiddleware {
    fn name(&self) -> &str {
        "human_in_the_loop"
    }

    async fn after_model(
        &self,
        response: &ModelResponse,
        _state: &AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        // tool_calls 확인
        let tool_calls = match &response.message.tool_calls {
            Some(tc) if !tc.is_empty() => tc,
            _ => return Ok(ModelControl::Continue),
        };

        let mut action_requests = Vec::new();
        let mut review_configs = Vec::new();

        for tc in tool_calls {
            if let Some(config) = self.should_interrupt(&tc.name) {
                let action = self.create_action_request(
                    &tc.id,
                    &tc.name,
                    &tc.arguments,
                    config,
                );

                let review = ReviewConfig::new(
                    tc.name.clone(),
                    config.allowed_decisions.clone(),
                );

                action_requests.push(action);
                review_configs.push(review);
            }
        }

        if action_requests.is_empty() {
            return Ok(ModelControl::Continue);
        }

        tracing::info!(
            interrupt_count = action_requests.len(),
            tools = ?action_requests.iter().map(|a| &a.name).collect::<Vec<_>>(),
            "Interrupting for human approval"
        );

        Ok(ModelControl::Interrupt(InterruptRequest::new(
            action_requests,
            review_configs,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::{Message, ToolCall};
    use std::sync::Arc;

    fn create_runtime() -> ToolRuntime {
        let state = AgentState::new();
        let backend = Arc::new(MemoryBackend::new());
        ToolRuntime::new(state, backend)
    }

    #[tokio::test]
    async fn test_no_tool_calls_continues() {
        let middleware = HumanInTheLoopMiddleware::new(HashMap::new());
        let runtime = create_runtime();
        let state = AgentState::new();

        let response = ModelResponse::new(Message::assistant("Hello"));

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();
        assert!(matches!(result, ModelControl::Continue));
    }

    #[tokio::test]
    async fn test_unconfigured_tool_continues() {
        let middleware = HumanInTheLoopMiddleware::new(HashMap::new());
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/test"}),
        };

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("Reading file", vec![tool_call])
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();
        assert!(matches!(result, ModelControl::Continue));
    }

    #[tokio::test]
    async fn test_configured_tool_interrupts() {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert("shell".to_string(), InterruptOnConfig::default());

        let middleware = HumanInTheLoopMiddleware::new(interrupt_on);
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({"command": "rm -rf /"}),
        };

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("Running command", vec![tool_call])
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();

        match result {
            ModelControl::Interrupt(req) => {
                assert_eq!(req.action_requests.len(), 1);
                assert_eq!(req.action_requests[0].name, "shell");
                assert_eq!(req.action_requests[0].id, "call_1");
            }
            _ => panic!("Expected Interrupt"),
        }
    }

    #[tokio::test]
    async fn test_disabled_config_continues() {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert("shell".to_string(), InterruptOnConfig::new(false));

        let middleware = HumanInTheLoopMiddleware::new(interrupt_on);
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({}),
        };

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("", vec![tool_call])
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();
        assert!(matches!(result, ModelControl::Continue));
    }

    #[tokio::test]
    async fn test_multiple_tools_partial_interrupt() {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert("shell".to_string(), InterruptOnConfig::default());
        // read_file은 설정 안 함

        let middleware = HumanInTheLoopMiddleware::new(interrupt_on);
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),  // 설정 안 됨
                arguments: serde_json::json!({}),
            },
            ToolCall {
                id: "call_2".to_string(),
                name: "shell".to_string(),  // 설정 됨
                arguments: serde_json::json!({}),
            },
        ];

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("", tool_calls)
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();

        match result {
            ModelControl::Interrupt(req) => {
                // shell만 인터럽트
                assert_eq!(req.action_requests.len(), 1);
                assert_eq!(req.action_requests[0].name, "shell");
            }
            _ => panic!("Expected Interrupt"),
        }
    }

    #[tokio::test]
    async fn test_for_tool_helper() {
        let middleware = HumanInTheLoopMiddleware::for_tool("dangerous_action");
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "dangerous_action".to_string(),
            arguments: serde_json::json!({}),
        };

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("", vec![tool_call])
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();
        assert!(matches!(result, ModelControl::Interrupt(_)));
    }

    #[tokio::test]
    async fn test_allow_all_decisions() {
        let mut interrupt_on = HashMap::new();
        interrupt_on.insert("shell".to_string(), InterruptOnConfig::allow_all());

        let middleware = HumanInTheLoopMiddleware::new(interrupt_on);
        let runtime = create_runtime();
        let state = AgentState::new();

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "shell".to_string(),
            arguments: serde_json::json!({}),
        };

        let response = ModelResponse::new(
            Message::assistant_with_tool_calls("", vec![tool_call])
        );

        let result = middleware.after_model(&response, &state, &runtime).await.unwrap();

        match result {
            ModelControl::Interrupt(req) => {
                let decisions = &req.review_configs[0].allowed_decisions;
                assert!(decisions.contains(&Decision::Approve));
                assert!(decisions.contains(&Decision::Reject));
                assert!(decisions.contains(&Decision::Edit));
            }
            _ => panic!("Expected Interrupt"),
        }
    }
}
