//! Message conversion between DeepAgents and Rig formats
//!
//! This module provides bidirectional conversion between DeepAgents' message types
//! and Rig's completion message format.
//!
//! # Architecture
//!
//! DeepAgents uses a simple struct-based Message with a Role enum:
//! ```text
//! Message { role: Role, content: String, tool_call_id, tool_calls }
//! ```
//!
//! Rig uses an enum-based Message with rich content types:
//! ```text
//! Message::User { content: OneOrMany<UserContent> }
//! Message::Assistant { id, content: OneOrMany<AssistantContent> }
//! ```
//!
//! This module bridges these two representations.

use crate::state::{Message, Role, ToolCall};
use crate::middleware::ToolDefinition;
use crate::error::DeepAgentError;
use rig::completion::message::{
    AssistantContent, Message as RigMessage, Text, ToolResultContent,
    UserContent,
};
use rig::completion::ToolDefinition as RigToolDefinition;
use rig::OneOrMany;

/// Trait for converting DeepAgents messages to Rig format
pub trait MessageConverter {
    /// Convert to Rig message format
    fn to_rig_message(&self) -> Result<RigMessage, DeepAgentError>;
}

/// Trait for converting Rig messages to DeepAgents format
#[allow(dead_code)] // Reserved for future bidirectional conversion
pub trait FromRigMessage {
    /// Convert from Rig message format
    fn from_rig_message(msg: &RigMessage) -> Result<Self, DeepAgentError>
    where
        Self: Sized;
}

/// Trait for converting DeepAgents tool definitions to Rig format
pub trait ToolConverter {
    /// Convert to Rig tool definition format
    fn to_rig_tool(&self) -> RigToolDefinition;
}

impl MessageConverter for Message {
    fn to_rig_message(&self) -> Result<RigMessage, DeepAgentError> {
        match self.role {
            Role::User => {
                Ok(RigMessage::user(&self.content))
            }
            Role::Assistant => {
                if let Some(tool_calls) = &self.tool_calls {
                    // Assistant message with tool calls
                    let mut contents: Vec<AssistantContent> = Vec::new();

                    // Add text content if present
                    if !self.content.is_empty() {
                        contents.push(AssistantContent::text(&self.content));
                    }

                    // Add tool calls
                    for tc in tool_calls {
                        contents.push(AssistantContent::tool_call(
                            &tc.id,
                            &tc.name,
                            tc.arguments.clone(),
                        ));
                    }

                    Ok(RigMessage::Assistant {
                        id: None,
                        content: OneOrMany::many(contents)
                            .map_err(|e| DeepAgentError::Conversion(format!(
                                "Failed to create assistant content: {}", e
                            )))?,
                    })
                } else {
                    // Simple assistant message
                    Ok(RigMessage::assistant(&self.content))
                }
            }
            Role::System => {
                // Rig handles system messages as preamble, not as conversation messages.
                // For compatibility, we convert to a user message with [System] prefix.
                // The actual system prompt should be set via completion request preamble.
                Ok(RigMessage::user(format!("[System]: {}", self.content)))
            }
            Role::Tool => {
                // Tool result message
                let tool_id = self.tool_call_id.clone().unwrap_or_default();
                Ok(RigMessage::tool_result(&tool_id, &self.content))
            }
        }
    }
}

impl FromRigMessage for Message {
    fn from_rig_message(msg: &RigMessage) -> Result<Self, DeepAgentError> {
        match msg {
            RigMessage::User { content } => {
                // Extract text content, handling tool results specially
                let mut text_parts = Vec::new();
                let mut tool_id = None;
                let mut is_tool_result = false;

                for item in content.iter() {
                    match item {
                        UserContent::Text(Text { text }) => {
                            text_parts.push(text.clone());
                        }
                        UserContent::ToolResult(result) => {
                            is_tool_result = true;
                            tool_id = Some(result.id.clone());
                            for content in result.content.iter() {
                                if let ToolResultContent::Text(Text { text }) = content {
                                    text_parts.push(text.clone());
                                }
                            }
                        }
                        _ => {
                            // Skip other content types (images, etc.)
                        }
                    }
                }

                let content = text_parts.join("\n");

                if is_tool_result {
                    Ok(Message::tool(&content, &tool_id.unwrap_or_default()))
                } else {
                    Ok(Message::user(&content))
                }
            }
            RigMessage::Assistant { id: _, content } => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Text(Text { text }) => {
                            text_parts.push(text.clone());
                        }
                        AssistantContent::ToolCall(tc) => {
                            tool_calls.push(ToolCall {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            });
                        }
                        _ => {
                            // Skip reasoning, images, etc.
                        }
                    }
                }

                let content = text_parts.join("\n");

                if tool_calls.is_empty() {
                    Ok(Message::assistant(&content))
                } else {
                    Ok(Message::assistant_with_tool_calls(&content, tool_calls))
                }
            }
        }
    }
}

impl ToolConverter for ToolDefinition {
    fn to_rig_tool(&self) -> RigToolDefinition {
        RigToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

/// Convert a slice of DeepAgents messages to Rig format
///
/// Filters out messages that cannot be converted (e.g., system messages
/// that should be handled via preamble).
pub fn convert_messages(messages: &[Message]) -> Result<Vec<RigMessage>, DeepAgentError> {
    messages
        .iter()
        .filter(|m| m.role != Role::System) // System messages handled via preamble
        .map(|m| m.to_rig_message())
        .collect()
}

/// Convert a slice of tool definitions to Rig format
pub fn convert_tools(tools: &[ToolDefinition]) -> Vec<RigToolDefinition> {
    tools.iter().map(|t| t.to_rig_tool()).collect()
}

/// Extract system message content for use as preamble
///
/// Returns the combined content of all system messages.
pub fn extract_system_preamble(messages: &[Message]) -> Option<String> {
    let system_messages: Vec<&str> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.as_str())
        .collect();

    if system_messages.is_empty() {
        None
    } else {
        Some(system_messages.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message_conversion() {
        let msg = Message::user("Hello, world!");
        let rig_msg = msg.to_rig_message().unwrap();

        match rig_msg {
            RigMessage::User { content } => {
                assert!(!content.is_empty());
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_assistant_message_conversion() {
        let msg = Message::assistant("I'm here to help!");
        let rig_msg = msg.to_rig_message().unwrap();

        match rig_msg {
            RigMessage::Assistant { id: _, content } => {
                assert!(!content.is_empty());
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_assistant_with_tool_calls_conversion() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/test.txt"}),
        };

        let msg = Message::assistant_with_tool_calls("Reading file...", vec![tool_call]);
        let rig_msg = msg.to_rig_message().unwrap();

        match rig_msg {
            RigMessage::Assistant { id: _, content } => {
                // Should have text + tool call
                assert!(content.len() >= 1);
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_tool_result_conversion() {
        let msg = Message::tool("File contents here", "call_123");
        let rig_msg = msg.to_rig_message().unwrap();

        match rig_msg {
            RigMessage::User { content } => {
                // Tool results are sent as User messages with ToolResult content
                assert!(!content.is_empty());
            }
            _ => panic!("Expected User message (tool result)"),
        }
    }

    #[test]
    fn test_system_message_conversion() {
        let msg = Message::system("You are a helpful assistant.");
        let rig_msg = msg.to_rig_message().unwrap();

        // System messages are converted to User messages with prefix
        match rig_msg {
            RigMessage::User { content } => {
                assert!(!content.is_empty());
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_roundtrip_user_message() {
        let original = Message::user("Test message");
        let rig_msg = original.to_rig_message().unwrap();
        let converted = Message::from_rig_message(&rig_msg).unwrap();

        assert_eq!(converted.role, Role::User);
        assert_eq!(converted.content, "Test message");
    }

    #[test]
    fn test_roundtrip_assistant_message() {
        let original = Message::assistant("Response text");
        let rig_msg = original.to_rig_message().unwrap();
        let converted = Message::from_rig_message(&rig_msg).unwrap();

        assert_eq!(converted.role, Role::Assistant);
        assert_eq!(converted.content, "Response text");
    }

    #[test]
    fn test_tool_definition_conversion() {
        let tool = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }),
        };

        let rig_tool = tool.to_rig_tool();

        assert_eq!(rig_tool.name, "read_file");
        assert_eq!(rig_tool.description, "Read a file from disk");
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            Message::system("System prompt"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let rig_messages = convert_messages(&messages).unwrap();

        // System message should be filtered out
        assert_eq!(rig_messages.len(), 2);
    }

    #[test]
    fn test_extract_system_preamble() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::system("Be concise."),
            Message::user("Hello"),
        ];

        let preamble = extract_system_preamble(&messages);

        assert!(preamble.is_some());
        let preamble = preamble.unwrap();
        assert!(preamble.contains("You are helpful"));
        assert!(preamble.contains("Be concise"));
    }

    #[test]
    fn test_extract_system_preamble_none() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
        ];

        let preamble = extract_system_preamble(&messages);
        assert!(preamble.is_none());
    }
}
