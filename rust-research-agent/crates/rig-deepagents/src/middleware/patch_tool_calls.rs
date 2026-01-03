//! PatchToolCallsMiddleware - 댕글링 도구 호출 패치
//!
//! AIMessage에 tool_calls가 있지만 대응하는 ToolMessage가 없는 경우를
//! 자동으로 패치합니다. 이는 다음 상황에서 발생할 수 있습니다:
//!
//! - 에이전트 실행이 도구 호출 중간에 인터럽트됨
//! - 메시지가 수동으로 편집/잘림
//! - 충돌 후 대화가 재개됨
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_deepagents::middleware::PatchToolCallsMiddleware;
//!
//! let middleware = PatchToolCallsMiddleware::new();
//! // 또는 커스텀 취소 메시지와 함께
//! let middleware = PatchToolCallsMiddleware::new()
//!     .with_message("도구 호출이 취소되었습니다");
//! ```

use async_trait::async_trait;
use std::collections::HashSet;

use crate::error::MiddlewareError;
use crate::middleware::{AgentMiddleware, StateUpdate};
use crate::runtime::ToolRuntime;
use crate::state::{AgentState, Message, Role};

/// 댕글링 도구 호출을 패치하는 미들웨어
///
/// 에이전트 실행 전에 메시지 히스토리를 검사하여
/// AIMessage의 tool_calls에 대응하는 ToolMessage가 없는 경우
/// 합성 ToolMessage를 삽입합니다.
pub struct PatchToolCallsMiddleware {
    /// 패치된 도구 응답에 사용할 메시지
    cancellation_message: String,
}

impl Default for PatchToolCallsMiddleware {
    fn default() -> Self {
        Self {
            cancellation_message: "Tool call was cancelled - another message arrived before completion.".to_string(),
        }
    }
}

impl PatchToolCallsMiddleware {
    /// 새 PatchToolCallsMiddleware 생성
    pub fn new() -> Self {
        Self::default()
    }

    /// 커스텀 취소 메시지 설정
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.cancellation_message = msg.into();
        self
    }

    /// 대응하는 ToolMessage가 없는 tool_calls 찾기
    ///
    /// Returns: Vec<(ai_msg_index, tool_call_id, tool_name)>
    fn find_dangling_tool_calls(messages: &[Message]) -> Vec<(usize, String, String)> {
        let mut dangling = Vec::new();

        // 먼저 모든 ToolMessage의 tool_call_id 수집
        let responded_ids: HashSet<&str> = messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .filter_map(|m| m.tool_call_id.as_deref())
            .collect();

        // AIMessage의 tool_calls 중 응답이 없는 것 찾기
        for (i, msg) in messages.iter().enumerate() {
            if msg.role == Role::Assistant {
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        if !responded_ids.contains(tc.id.as_str()) {
                            dangling.push((i, tc.id.clone(), tc.name.clone()));
                        }
                    }
                }
            }
        }

        dangling
    }
}

#[async_trait]
impl AgentMiddleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str {
        "patch_tool_calls"
    }

    async fn before_agent(
        &self,
        state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<Option<StateUpdate>, MiddlewareError> {
        if state.messages.is_empty() {
            return Ok(None);
        }

        let dangling = Self::find_dangling_tool_calls(&state.messages);

        if dangling.is_empty() {
            return Ok(None);
        }

        tracing::debug!(
            dangling_count = dangling.len(),
            "Found dangling tool calls to patch"
        );

        // 패치된 메시지 목록 구성
        // 각 dangling tool call에 대해 해당 AIMessage 바로 다음에 ToolMessage 삽입
        let mut patched = Vec::with_capacity(state.messages.len() + dangling.len());

        // dangling을 AIMessage 인덱스별로 그룹화
        let mut dangling_by_index: std::collections::HashMap<usize, Vec<(String, String)>> =
            std::collections::HashMap::new();
        for (idx, id, name) in dangling {
            dangling_by_index.entry(idx).or_default().push((id, name));
        }

        for (i, msg) in state.messages.iter().enumerate() {
            patched.push(msg.clone());

            // 이 AIMessage 다음에 합성 ToolMessage 삽입
            if let Some(calls) = dangling_by_index.get(&i) {
                for (tool_call_id, tool_name) in calls {
                    let content = format!(
                        "Tool call '{}' (ID: {}) was cancelled. {}",
                        tool_name, tool_call_id, self.cancellation_message
                    );
                    patched.push(Message::tool(&content, tool_call_id));

                    tracing::debug!(
                        tool_name = %tool_name,
                        tool_call_id = %tool_call_id,
                        "Patched dangling tool call"
                    );
                }
            }
        }

        Ok(Some(StateUpdate::SetMessages(patched)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::ToolCall;
    use std::sync::Arc;

    fn create_runtime(state: &AgentState) -> ToolRuntime {
        let backend = Arc::new(MemoryBackend::new());
        ToolRuntime::new(state.clone(), backend)
    }

    #[tokio::test]
    async fn test_no_messages() {
        let middleware = PatchToolCallsMiddleware::new();
        let mut state = AgentState::new();
        let runtime = create_runtime(&state);

        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_no_dangling_calls() {
        let middleware = PatchToolCallsMiddleware::new();

        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };

        let mut state = AgentState::with_messages(vec![
            Message::user("Search for something"),
            Message::assistant_with_tool_calls("Let me search", vec![tool_call]),
            Message::tool("Found results", "call_123"),  // 대응하는 ToolMessage 있음
            Message::assistant("Here are the results"),
        ]);

        let runtime = create_runtime(&state);
        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();

        // 댕글링이 없으므로 None
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_patches_single_dangling_call() {
        let middleware = PatchToolCallsMiddleware::new();

        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };

        let mut state = AgentState::with_messages(vec![
            Message::user("Search for something"),
            Message::assistant_with_tool_calls("Let me search", vec![tool_call]),
            // ToolMessage 없음!
            Message::user("Never mind, do something else"),
        ]);

        let runtime = create_runtime(&state);
        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();

        assert!(result.is_some());

        if let Some(StateUpdate::SetMessages(msgs)) = result {
            assert_eq!(msgs.len(), 4); // 원래 3개 + 합성 ToolMessage 1개
            assert_eq!(msgs[2].role, Role::Tool);
            assert_eq!(msgs[2].tool_call_id, Some("call_123".to_string()));
            assert!(msgs[2].content.contains("cancelled"));
        } else {
            panic!("Expected SetMessages");
        }
    }

    #[tokio::test]
    async fn test_patches_multiple_dangling_calls() {
        let middleware = PatchToolCallsMiddleware::new();

        let tool_call1 = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({}),
        };
        let tool_call2 = ToolCall {
            id: "call_2".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({}),
        };

        let mut state = AgentState::with_messages(vec![
            Message::user("Do multiple things"),
            Message::assistant_with_tool_calls("", vec![tool_call1, tool_call2]),
            // 둘 다 ToolMessage 없음!
            Message::user("Cancel all"),
        ]);

        let runtime = create_runtime(&state);
        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();

        assert!(result.is_some());

        if let Some(StateUpdate::SetMessages(msgs)) = result {
            assert_eq!(msgs.len(), 5); // 원래 3개 + 합성 ToolMessage 2개
            assert_eq!(msgs[2].role, Role::Tool);
            assert_eq!(msgs[3].role, Role::Tool);
        } else {
            panic!("Expected SetMessages");
        }
    }

    #[tokio::test]
    async fn test_partial_dangling() {
        let middleware = PatchToolCallsMiddleware::new();

        let tool_call1 = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({}),
        };
        let tool_call2 = ToolCall {
            id: "call_2".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({}),
        };

        let mut state = AgentState::with_messages(vec![
            Message::user("Do multiple things"),
            Message::assistant_with_tool_calls("", vec![tool_call1, tool_call2]),
            Message::tool("Search result", "call_1"),  // call_1만 응답 있음
            // call_2는 응답 없음!
            Message::user("Cancel"),
        ]);

        let runtime = create_runtime(&state);
        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();

        assert!(result.is_some());

        if let Some(StateUpdate::SetMessages(msgs)) = result {
            // 원래 4개 + call_2에 대한 합성 ToolMessage 1개 = 5개
            assert_eq!(msgs.len(), 5);

            // call_2에 대한 합성 메시지 확인
            let synthetic = msgs.iter().find(|m|
                m.role == Role::Tool &&
                m.tool_call_id.as_deref() == Some("call_2") &&
                m.content.contains("cancelled")
            );
            assert!(synthetic.is_some());
        } else {
            panic!("Expected SetMessages");
        }
    }

    #[tokio::test]
    async fn test_custom_message() {
        let middleware = PatchToolCallsMiddleware::new()
            .with_message("사용자가 취소함");

        let tool_call = ToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({}),
        };

        let mut state = AgentState::with_messages(vec![
            Message::user("Search"),
            Message::assistant_with_tool_calls("", vec![tool_call]),
        ]);

        let runtime = create_runtime(&state);
        let result = middleware.before_agent(&mut state, &runtime).await.unwrap();

        if let Some(StateUpdate::SetMessages(msgs)) = result {
            assert!(msgs[2].content.contains("사용자가 취소함"));
        } else {
            panic!("Expected SetMessages");
        }
    }
}
