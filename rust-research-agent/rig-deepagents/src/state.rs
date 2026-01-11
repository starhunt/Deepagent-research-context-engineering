// src/state.rs
//! 에이전트 상태 정의
//!
//! Python Reference: langchain/agents/middleware/types.py의 AgentState

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::any::Any;
use chrono::Utc;
use tracing::warn;

/// Todo 상태
/// Python: Literal["pending", "in_progress", "completed"]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Todo 아이템
/// Python: Todo(TypedDict)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    pub status: TodoStatus,
}

impl Todo {
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            status: TodoStatus::Pending,
        }
    }

    pub fn with_status(content: &str, status: TodoStatus) -> Self {
        Self {
            content: content.to_string(),
            status,
        }
    }
}

/// 파일 데이터
/// Python: FileData(TypedDict) in filesystem.py
///
/// **Note:** 이 타입은 error.rs의 WriteResult/EditResult에서도 사용됨
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    pub content: Vec<String>,
    pub created_at: String,
    pub modified_at: String,
}

impl FileData {
    pub fn new(content: &str) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            content: content.lines().map(String::from).collect(),
            created_at: now.clone(),
            modified_at: now,
        }
    }

    pub fn as_string(&self) -> String {
        self.content.join("\n")
    }

    pub fn update(&mut self, new_content: &str) {
        self.content = new_content.lines().map(String::from).collect();
        self.modified_at = Utc::now().to_rfc3339();
    }

    pub fn line_count(&self) -> usize {
        self.content.len()
    }
}

/// 메시지 역할
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

/// 도구 호출 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// 메시지
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            status: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            status: None,
        }
    }

    pub fn assistant_with_tool_calls(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
            status: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            tool_call_id: None,
            tool_calls: None,
            status: None,
        }
    }

    pub fn tool(content: &str, tool_call_id: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(tool_call_id.to_string()),
            tool_calls: None,
            status: None,
        }
    }

    pub fn tool_with_status(content: &str, tool_call_id: &str, status: &str) -> Self {
        Self {
            role: Role::Tool,
            content: content.to_string(),
            tool_call_id: Some(tool_call_id.to_string()),
            tool_calls: None,
            status: Some(status.to_string()),
        }
    }

    /// 이 메시지에 dangling tool call이 있는지 확인
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

/// 에이전트 상태
/// Python: AgentState(TypedDict) + FilesystemState + PlanningState
///
/// Note: Clone은 extensions 필드 없이 수동 구현됨 (dyn Any는 Clone 불가)
#[derive(Debug, Default)]
pub struct AgentState {
    /// 메시지 히스토리
    pub messages: Vec<Message>,

    /// Todo 리스트 (TodoListMiddleware)
    pub todos: Vec<Todo>,

    /// 가상 파일 시스템 (FilesystemMiddleware)
    pub files: HashMap<String, FileData>,

    /// 구조화된 응답
    pub structured_response: Option<serde_json::Value>,

    /// 확장 데이터 (미들웨어별 커스텀 상태)
    /// Note: 이 필드는 Clone되지 않음 - 새 HashMap으로 초기화됨
    extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}

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

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 초기 메시지로 상태 생성
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    /// 확장 데이터 설정
    pub fn set_extension<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.extensions.insert(key.to_string(), Box::new(value));
    }

    /// 확장 데이터 조회
    pub fn get_extension<T: Any>(&self, key: &str) -> Option<&T> {
        self.extensions.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// 마지막 사용자 메시지 가져오기
    pub fn last_user_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::User)
    }

    /// 마지막 어시스턴트 메시지 가져오기
    pub fn last_assistant_message(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::Assistant)
    }

    /// 메시지 추가
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// 메시지 수 반환
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_status_serialization() {
        let status = TodoStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::new();
        assert!(state.messages.is_empty());
        assert!(state.todos.is_empty());
        assert!(state.files.is_empty());
    }

    #[test]
    fn test_file_data_creation() {
        let file = FileData::new("hello\nworld");
        assert_eq!(file.content, vec!["hello", "world"]);
        assert!(!file.created_at.is_empty());
        assert_eq!(file.line_count(), 2);
    }

    #[test]
    fn test_message_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/test.txt"}),
        };
        let msg = Message::assistant_with_tool_calls("", vec![tool_call]);
        assert!(msg.has_tool_calls());
    }

    #[test]
    fn test_agent_state_with_messages() {
        let state = AgentState::with_messages(vec![Message::user("Hello")]);
        assert_eq!(state.message_count(), 1);
        assert!(state.last_user_message().is_some());
    }
}
