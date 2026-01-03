//! Token Counting Utilities
//!
//! Provides approximate token counting for messages to determine when
//! summarization should be triggered.
//!
//! The counting algorithm is based on LangChain's `count_tokens_approximately`:
//! - Characters are divided by chars_per_token (model-dependent)
//! - Additional overhead is added per message for role/structure
//! - Tool calls are serialized and included in the count

use crate::state::Message;

/// Default characters per token for most models (OpenAI GPT-4, etc.)
pub const DEFAULT_CHARS_PER_TOKEN: f32 = 4.0;

/// Characters per token for Anthropic Claude models (more efficient encoding)
pub const CLAUDE_CHARS_PER_TOKEN: f32 = 3.3;

/// Default overhead tokens added per message for structure
pub const DEFAULT_OVERHEAD_PER_MESSAGE: f32 = 3.0;

/// Count tokens approximately for a slice of messages.
///
/// This is a fast, lightweight approximation suitable for hot-path token counting.
/// It does not require any external API calls.
///
/// # Algorithm
///
/// For each message:
/// 1. Count characters in content
/// 2. Add serialized tool_calls length (if present)
/// 3. Add role name length
/// 4. Divide by chars_per_token and round up
/// 5. Add overhead per message
///
/// # Arguments
///
/// * `messages` - Slice of messages to count
/// * `chars_per_token` - Model-specific character ratio (4.0 for GPT, 3.3 for Claude)
/// * `overhead_per_message` - Additional tokens per message for structure
///
/// # Returns
///
/// Approximate total token count
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::middleware::summarization::token_counter::count_tokens_approximately;
/// use rig_deepagents::state::Message;
///
/// let messages = vec![
///     Message::user("Hello, world!"),
///     Message::assistant("Hi there! How can I help you?"),
/// ];
///
/// let tokens = count_tokens_approximately(&messages, 4.0, 3.0);
/// println!("Approximate tokens: {}", tokens);
/// ```
pub fn count_tokens_approximately(
    messages: &[Message],
    chars_per_token: f32,
    overhead_per_message: f32,
) -> usize {
    messages
        .iter()
        .map(|msg| count_message_tokens(msg, chars_per_token, overhead_per_message))
        .sum()
}

/// Count tokens for a single message
fn count_message_tokens(msg: &Message, chars_per_token: f32, overhead_per_message: f32) -> usize {
    let mut char_count = msg.content.len();

    // Add role name length
    char_count += role_name_length(&msg.role);

    // Add tool call ID if present
    if let Some(ref tool_call_id) = msg.tool_call_id {
        char_count += tool_call_id.len();
    }

    // Add serialized tool calls if present
    if let Some(ref tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            // Approximate: id + name + serialized arguments
            char_count += tc.id.len();
            char_count += tc.name.len();
            // Serialize arguments to get approximate length
            if let Ok(args_str) = serde_json::to_string(&tc.arguments) {
                char_count += args_str.len();
            }
        }
    }

    // Calculate tokens
    let tokens = (char_count as f32 / chars_per_token).ceil() as usize;
    tokens + overhead_per_message as usize
}

/// Get the length of the role name for token counting
fn role_name_length(role: &crate::state::Role) -> usize {
    match role {
        crate::state::Role::User => 4,      // "user"
        crate::state::Role::Assistant => 9, // "assistant"
        crate::state::Role::System => 6,    // "system"
        crate::state::Role::Tool => 4,      // "tool"
    }
}

/// Get the recommended chars_per_token for a model name.
///
/// Returns a model-specific ratio for more accurate token counting.
///
/// # Arguments
///
/// * `model` - Model name or identifier
///
/// # Returns
///
/// Characters per token ratio (lower = more tokens per character)
pub fn get_chars_per_token(model: &str) -> f32 {
    let model_lower = model.to_lowercase();

    if model_lower.contains("claude") {
        CLAUDE_CHARS_PER_TOKEN
    } else {
        DEFAULT_CHARS_PER_TOKEN
    }
}

/// Configuration for token counting
#[derive(Debug, Clone)]
pub struct TokenCounterConfig {
    /// Characters per token ratio
    pub chars_per_token: f32,
    /// Overhead tokens per message
    pub overhead_per_message: f32,
}

impl Default for TokenCounterConfig {
    fn default() -> Self {
        Self {
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            overhead_per_message: DEFAULT_OVERHEAD_PER_MESSAGE,
        }
    }
}

impl TokenCounterConfig {
    /// Create a config tuned for a specific model
    pub fn for_model(model: &str) -> Self {
        Self {
            chars_per_token: get_chars_per_token(model),
            overhead_per_message: DEFAULT_OVERHEAD_PER_MESSAGE,
        }
    }

    /// Count tokens using this config
    pub fn count(&self, messages: &[Message]) -> usize {
        count_tokens_approximately(messages, self.chars_per_token, self.overhead_per_message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Message, ToolCall};

    #[test]
    fn test_count_simple_messages() {
        let messages = vec![
            Message::user("Hello"),           // 5 chars + 4 (role) = 9
            Message::assistant("Hi there!"), // 9 chars + 9 (role) = 18
        ];

        let tokens = count_tokens_approximately(&messages, 4.0, 3.0);
        // (9/4).ceil() + 3 + (18/4).ceil() + 3 = 3 + 3 + 5 + 3 = 14
        assert!(tokens > 0);
        assert!(tokens < 50); // Reasonable upper bound
    }

    #[test]
    fn test_count_empty_messages() {
        let messages: Vec<Message> = vec![];
        let tokens = count_tokens_approximately(&messages, 4.0, 3.0);
        assert_eq!(tokens, 0);
    }

    #[test]
    fn test_count_with_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/test.txt"}),
        };

        let messages = vec![Message::assistant_with_tool_calls("Reading file...", vec![tool_call])];

        let tokens = count_tokens_approximately(&messages, 4.0, 3.0);
        // Should be higher due to tool call serialization
        assert!(tokens > 5);
    }

    #[test]
    fn test_count_tool_result() {
        let messages = vec![Message::tool("File contents here", "call_123")];

        let tokens = count_tokens_approximately(&messages, 4.0, 3.0);
        assert!(tokens > 0);
    }

    #[test]
    fn test_claude_vs_openai_ratio() {
        let messages = vec![
            Message::user("This is a longer message with more content to analyze."),
            Message::assistant("And this is a response with even more detailed content."),
        ];

        let openai_tokens = count_tokens_approximately(&messages, 4.0, 3.0);
        let claude_tokens = count_tokens_approximately(&messages, 3.3, 3.0);

        // Claude should count more tokens (smaller chars_per_token = more tokens)
        assert!(claude_tokens > openai_tokens);
    }

    #[test]
    fn test_get_chars_per_token() {
        assert_eq!(get_chars_per_token("gpt-4"), DEFAULT_CHARS_PER_TOKEN);
        assert_eq!(get_chars_per_token("gpt-4-turbo"), DEFAULT_CHARS_PER_TOKEN);
        assert_eq!(get_chars_per_token("claude-3-opus"), CLAUDE_CHARS_PER_TOKEN);
        assert_eq!(get_chars_per_token("claude-3-sonnet"), CLAUDE_CHARS_PER_TOKEN);
        assert_eq!(get_chars_per_token("Claude-3-Haiku"), CLAUDE_CHARS_PER_TOKEN);
    }

    #[test]
    fn test_token_counter_config() {
        let config = TokenCounterConfig::for_model("claude-3-opus");
        assert_eq!(config.chars_per_token, CLAUDE_CHARS_PER_TOKEN);

        let messages = vec![Message::user("Test message")];
        let tokens = config.count(&messages);
        assert!(tokens > 0);
    }

    #[test]
    fn test_realistic_conversation() {
        // Simulate a realistic conversation
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Can you help me with some code?"),
            Message::assistant("Of course! What would you like help with?"),
            Message::user("I need to write a function that calculates fibonacci numbers."),
            Message::assistant("Here's a simple recursive implementation:\n\n```rust\nfn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}\n```"),
        ];

        let tokens = count_tokens_approximately(&messages, 4.0, 3.0);

        // Should be a reasonable count for this conversation
        assert!(tokens > 50);
        assert!(tokens < 500);
    }
}
