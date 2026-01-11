use crate::middleware::summarization::token_counter::{
    count_tokens_approximately, DEFAULT_CHARS_PER_TOKEN, DEFAULT_OVERHEAD_PER_MESSAGE,
};
use crate::state::Message;
#[cfg(feature = "tokenizer-tiktoken")]
use crate::state::Role;

pub trait TokenCounter: Send + Sync {
    fn count_text(&self, text: &str) -> usize;
    fn count_message(&self, message: &Message) -> usize;
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages.iter().map(|msg| self.count_message(msg)).sum()
    }
}

#[derive(Debug, Clone)]
pub struct ApproxTokenCounter {
    pub chars_per_token: f32,
    pub overhead_per_message: usize,
}

impl ApproxTokenCounter {
    pub fn new(chars_per_token: f32, overhead_per_message: usize) -> Self {
        Self {
            chars_per_token,
            overhead_per_message,
        }
    }
}

impl Default for ApproxTokenCounter {
    fn default() -> Self {
        Self {
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            overhead_per_message: DEFAULT_OVERHEAD_PER_MESSAGE as usize,
        }
    }
}

impl TokenCounter for ApproxTokenCounter {
    fn count_text(&self, text: &str) -> usize {
        (text.len() as f32 / self.chars_per_token).ceil() as usize
    }

    fn count_message(&self, message: &Message) -> usize {
        count_tokens_approximately(
            std::slice::from_ref(message),
            self.chars_per_token,
            self.overhead_per_message as f32,
        )
    }
}

#[cfg(feature = "tokenizer-tiktoken")]
#[derive(Debug, Clone)]
pub struct TiktokenTokenCounter {
    encoder: tiktoken_rs::CoreBPE,
}

#[cfg(feature = "tokenizer-tiktoken")]
impl TiktokenTokenCounter {
    pub fn new(encoder: tiktoken_rs::CoreBPE) -> Self {
        Self { encoder }
    }

    pub fn cl100k_base() -> Result<Self, tiktoken_rs::Error> {
        Ok(Self {
            encoder: tiktoken_rs::cl100k_base()?,
        })
    }
}

#[cfg(feature = "tokenizer-tiktoken")]
impl TokenCounter for TiktokenTokenCounter {
    fn count_text(&self, text: &str) -> usize {
        self.encoder.encode_with_special_tokens(text).len()
    }

    fn count_message(&self, message: &Message) -> usize {
        let message_text = build_message_text(message);
        self.count_text(&message_text)
    }
}

#[cfg(feature = "tokenizer-tiktoken")]
fn role_name(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

#[cfg(feature = "tokenizer-tiktoken")]
fn build_message_text(message: &Message) -> String {
    let mut text = String::new();
    text.push_str(&message.content);
    text.push_str(role_name(&message.role));

    if let Some(ref tool_call_id) = message.tool_call_id {
        text.push_str(tool_call_id);
    }

    if let Some(ref tool_calls) = message.tool_calls {
        for tc in tool_calls {
            text.push_str(&tc.id);
            text.push_str(&tc.name);
            if let Ok(args_str) = serde_json::to_string(&tc.arguments) {
                text.push_str(&args_str);
            }
        }
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Message;

    #[test]
    fn test_approx_counter_counts_non_zero() {
        let counter = ApproxTokenCounter::new(4.0, 3);
        let messages = vec![Message::user("Hello there")];
        assert!(counter.count_messages(&messages) > 0);
        assert!(counter.count_text("Hello there") > 0);
    }

    #[cfg(feature = "tokenizer-tiktoken")]
    #[test]
    fn test_tiktoken_counter_counts_non_zero() {
        let counter = TiktokenTokenCounter::cl100k_base().unwrap();
        let messages = vec![Message::assistant("Hello there")];
        assert!(counter.count_messages(&messages) > 0);
        assert!(counter.count_text("Hello there") > 0);
    }
}
