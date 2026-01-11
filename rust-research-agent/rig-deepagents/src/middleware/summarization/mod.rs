//! Summarization Middleware
//!
//! Provides automatic context summarization for long-running agent conversations.
//! When token count exceeds configurable thresholds, older messages are summarized
//! to stay within the model's context window.
//!
//! # Overview
//!
//! The middleware:
//! 1. Counts tokens in the current conversation (approximate)
//! 2. Checks trigger conditions (token count, message count, or fraction of max)
//! 3. If triggered, partitions messages into "to summarize" and "preserved"
//! 4. Calls an LLM to generate a summary of the older messages
//! 5. Replaces the conversation with: summary + preserved messages
//!
//! # Example
//!
//! ```rust,ignore
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig_deepagents::middleware::summarization::{
//!     SummarizationMiddleware, SummarizationConfig, TriggerCondition, KeepSize
//! };
//! use rig_deepagents::RigAgentAdapter;
//! use std::sync::Arc;
//!
//! let client = rig::providers::openai::Client::from_env();
//! let agent = client.agent("gpt-4").build();
//! let provider = Arc::new(RigAgentAdapter::new(agent));
//!
//! let config = SummarizationConfig::builder()
//!     .trigger(TriggerCondition::Fraction(0.85))
//!     .keep(KeepSize::Fraction(0.10))
//!     .max_input_tokens(128_000)
//!     .build();
//!
//! let middleware = SummarizationMiddleware::new(provider, config);
//! ```

pub mod token_counter;
pub mod trigger;
pub mod config;

pub use token_counter::{
    count_tokens_approximately, get_chars_per_token, TokenCounterConfig,
    DEFAULT_CHARS_PER_TOKEN, CLAUDE_CHARS_PER_TOKEN, DEFAULT_OVERHEAD_PER_MESSAGE,
};
pub use trigger::{TriggerCondition, KeepSize};
pub use config::{SummarizationConfig, SummarizationConfigBuilder, DEFAULT_SUMMARY_PROMPT};

use std::sync::Arc;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::error::MiddlewareError;
use crate::llm::LLMProvider;
use crate::middleware::traits::{AgentMiddleware, DynTool, ModelControl, ModelRequest};
use crate::runtime::ToolRuntime;
use crate::state::{AgentState, Message, Role};
use crate::tokenization::{ApproxTokenCounter, TokenCounter};

/// Summarization Middleware for token budget management.
///
/// Automatically summarizes older messages when the conversation exceeds
/// configurable token thresholds, preserving recent context while staying
/// within the model's context window.
pub struct SummarizationMiddleware {
    /// LLM provider for generating summaries
    llm_provider: Arc<dyn LLMProvider>,
    /// Configuration
    config: SummarizationConfig,
    token_counter: Arc<dyn TokenCounter>,
}

impl SummarizationMiddleware {
    /// Create a new SummarizationMiddleware.
    ///
    /// # Arguments
    ///
    /// * `llm_provider` - LLM provider for generating summaries
    /// * `config` - Configuration for triggers, keep size, and prompts
    pub fn new(llm_provider: Arc<dyn LLMProvider>, config: SummarizationConfig) -> Self {
        let token_counter = Arc::new(ApproxTokenCounter::new(
            config.chars_per_token,
            config.overhead_per_message as usize,
        ));
        Self {
            llm_provider,
            config,
            token_counter,
        }
    }

    pub fn with_token_counter(
        llm_provider: Arc<dyn LLMProvider>,
        config: SummarizationConfig,
        token_counter: Arc<dyn TokenCounter>,
    ) -> Self {
        Self {
            llm_provider,
            config,
            token_counter,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(llm_provider: Arc<dyn LLMProvider>) -> Self {
        Self::new(llm_provider, SummarizationConfig::default())
    }

    /// Create with model-tuned configuration.
    pub fn for_model(llm_provider: Arc<dyn LLMProvider>, model: &str) -> Self {
        Self::new(llm_provider, SummarizationConfig::for_model(model))
    }

    /// Count tokens in the current messages.
    fn count_tokens(&self, messages: &[Message]) -> usize {
        self.token_counter.count_messages(messages)
    }

    /// Check if summarization should be triggered.
    fn should_summarize(&self, token_count: usize, message_count: usize) -> bool {
        self.config.should_summarize(token_count, message_count)
    }

    /// Partition messages into (to_summarize, preserved).
    ///
    /// Respects AI/Tool message pair boundaries.
    fn partition_messages(&self, messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
        if messages.is_empty() {
            return (vec![], vec![]);
        }

        let cutoff = self.find_cutoff_index(messages);

        if cutoff == 0 {
            // Nothing to summarize
            return (vec![], messages.to_vec());
        }

        if cutoff >= messages.len() {
            // Everything to summarize (edge case)
            return (messages.to_vec(), vec![]);
        }

        (messages[..cutoff].to_vec(), messages[cutoff..].to_vec())
    }

    /// Find the cutoff index for message partitioning.
    ///
    /// Ensures we don't split in the middle of an AI/Tool message pair.
    fn find_cutoff_index(&self, messages: &[Message]) -> usize {
        let keep_count = if self.config.keep.is_message_based() {
            self.config.keep.message_count().unwrap_or(6)
        } else {
            // Token-based: estimate messages to keep
            let keep_tokens = self.config.keep.calculate_keep_tokens(self.config.max_input_tokens);
            self.estimate_messages_for_tokens(messages, keep_tokens)
        };

        // Calculate initial cutoff
        let initial_cutoff = messages.len().saturating_sub(keep_count);

        // Adjust to respect AI/Tool pair boundaries
        self.find_safe_cutoff(messages, initial_cutoff)
    }

    /// Estimate how many messages (from the end) fit within a token budget.
    fn estimate_messages_for_tokens(&self, messages: &[Message], token_budget: usize) -> usize {
        let mut total_tokens = 0;
        let mut count = 0;

        for msg in messages.iter().rev() {
            let msg_tokens = self.token_counter.count_message(msg);

            if total_tokens + msg_tokens > token_budget {
                break;
            }

            total_tokens += msg_tokens;
            count += 1;
        }

        count.max(1) // Keep at least 1 message
    }

    /// Find a safe cutoff point that doesn't split AI/Tool pairs.
    ///
    /// If the initial cutoff lands on a Tool message, advance past all consecutive
    /// Tool messages to keep the AI message with its responses.
    fn find_safe_cutoff(&self, messages: &[Message], initial_cutoff: usize) -> usize {
        if initial_cutoff >= messages.len() {
            return messages.len();
        }

        let mut cutoff = initial_cutoff;

        while cutoff > 0 && messages[cutoff].role == Role::Tool {
            cutoff -= 1;
        }

        cutoff
    }

    /// Generate a summary of the messages.
    async fn generate_summary(&self, messages: &[Message]) -> Result<String, MiddlewareError> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        // Trim messages to fit summarizer's context
        let trimmed = self.trim_for_summary(messages);

        // Format messages for the prompt
        let conversation_text = self.format_messages(&trimmed);

        // Build the summarization prompt
        let prompt = format!(
            "{}\n{}\n</conversation_to_summarize>",
            self.config.summary_prompt,
            conversation_text
        );

        // Create a simple request
        let request_messages = vec![Message::user(&prompt)];

        debug!(
            message_count = trimmed.len(),
            prompt_length = prompt.len(),
            "Generating summary"
        );

        // Call LLM
        let response = self.llm_provider
            .complete(&request_messages, &[], None)
            .await
            .map_err(|e| MiddlewareError::ToolExecution(format!("Summary generation failed: {}", e)))?;

        Ok(response.message.content)
    }

    /// Trim messages to fit within the summarizer's token budget.
    fn trim_for_summary(&self, messages: &[Message]) -> Vec<Message> {
        let max_tokens = self.config.trim_tokens_to_summarize;
        let mut total_tokens = 0;
        let mut result = Vec::new();

        // Take messages from the end (most recent first), respecting token budget
        for msg in messages.iter().rev() {
            let msg_tokens = self.token_counter.count_message(msg);

            if total_tokens + msg_tokens > max_tokens {
                break;
            }

            result.push(msg.clone());
            total_tokens += msg_tokens;
        }

        // Reverse to maintain chronological order
        result.reverse();
        result
    }

    /// Format messages for inclusion in the summary prompt.
    fn format_messages(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::System => "System",
                    Role::Tool => "Tool",
                };

                if let Some(ref tool_calls) = msg.tool_calls {
                    let calls: Vec<String> = tool_calls
                        .iter()
                        .map(|tc| format!("  {}({})", tc.name, tc.arguments))
                        .collect();
                    format!("{}: {}\nTool Calls:\n{}", role, msg.content, calls.join("\n"))
                } else if let Some(ref tool_id) = msg.tool_call_id {
                    format!("{} ({}): {}", role, tool_id, msg.content)
                } else {
                    format!("{}: {}", role, msg.content)
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl AgentMiddleware for SummarizationMiddleware {
    fn name(&self) -> &str {
        "summarization"
    }

    fn tools(&self) -> Vec<DynTool> {
        // No tools provided; summarization is automatic
        vec![]
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        // No prompt modification needed
        prompt
    }

    async fn before_model(
        &self,
        request: &mut ModelRequest,
        state: &mut AgentState,
        _runtime: &ToolRuntime,
    ) -> Result<ModelControl, MiddlewareError> {
        let token_count = self.count_tokens(&state.messages);
        let message_count = state.messages.len();

        debug!(
            token_count = token_count,
            message_count = message_count,
            max_tokens = self.config.max_input_tokens,
            "Checking summarization trigger"
        );

        // Check if we should summarize
        if !self.should_summarize(token_count, message_count) {
            return Ok(ModelControl::Continue);
        }

        info!(
            token_count = token_count,
            message_count = message_count,
            "Triggering summarization"
        );

        // Partition messages
        let (to_summarize, preserved) = self.partition_messages(&state.messages);

        if to_summarize.is_empty() {
            debug!("No messages to summarize");
            return Ok(ModelControl::Continue);
        }

        debug!(
            to_summarize = to_summarize.len(),
            preserved = preserved.len(),
            "Partitioned messages"
        );

        // Generate summary
        let summary = match self.generate_summary(&to_summarize).await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "Failed to generate summary, keeping original messages");
                return Ok(ModelControl::Continue);
            }
        };

        // Build new message list
        let summary_message = format!(
            "Here is a summary of the conversation to date:\n\n{}",
            summary
        );
        let mut new_messages = vec![Message::user(&summary_message)];
        new_messages.extend(preserved);

        let new_token_count = self.count_tokens(&new_messages);
        info!(
            original_tokens = token_count,
            new_tokens = new_token_count,
            tokens_saved = token_count.saturating_sub(new_token_count),
            "Summarization complete"
        );

        state.messages = new_messages.clone();
        request.messages = new_messages.clone();

        Ok(ModelControl::ModifyRequest(request.clone()))
    }
}

impl std::fmt::Debug for SummarizationMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummarizationMiddleware")
            .field("config", &self.config)
            .field("provider", &self.llm_provider.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LLMConfig, LLMResponse};
    use crate::middleware::{ModelControl, ModelRequest};
    use crate::runtime::ToolRuntime;

    /// Mock LLM provider for testing
    struct MockProvider {
        summary_response: String,
    }

    impl MockProvider {
        fn new(response: &str) -> Self {
            Self {
                summary_response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[crate::middleware::ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, crate::error::DeepAgentError> {
            Ok(LLMResponse::new(Message::assistant(&self.summary_response)))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }
    }

    #[test]
    fn test_partition_empty_messages() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::default();
        let middleware = SummarizationMiddleware::new(provider, config);

        let (to_summarize, preserved) = middleware.partition_messages(&[]);
        assert!(to_summarize.is_empty());
        assert!(preserved.is_empty());
    }

    #[test]
    fn test_partition_respects_keep_size() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::builder()
            .keep(KeepSize::Messages(2))
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        let messages = vec![
            Message::user("First"),
            Message::assistant("Response 1"),
            Message::user("Second"),
            Message::assistant("Response 2"),
            Message::user("Third"),
            Message::assistant("Response 3"),
        ];

        let (to_summarize, preserved) = middleware.partition_messages(&messages);

        // Should keep last 2 messages
        assert_eq!(preserved.len(), 2);
        assert_eq!(to_summarize.len(), 4);
    }

    #[test]
    fn test_safe_cutoff_moves_backward_for_tool_messages() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::builder()
            .keep(KeepSize::Messages(3))
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        let messages = vec![
            Message::user("Request"),
            Message::assistant_with_tool_calls("Let me check", vec![
                crate::state::ToolCall {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!({"path": "/test"}),
                }
            ]),
            Message::tool("File contents", "call_1"),
            Message::assistant("Here's what I found"),
            Message::user("Thanks"),
        ];

        let (to_summarize, preserved) = middleware.partition_messages(&messages);

        assert_eq!(to_summarize.len(), 1);
        assert_eq!(preserved.len(), 4);
        assert!(preserved[0].tool_calls.is_some());
        assert_eq!(
            to_summarize.len() + preserved.len(),
            messages.len(),
            "Partition should not lose any messages"
        );
    }

    #[tokio::test]
    async fn test_before_model_summarizes_request_messages() {
        let provider = Arc::new(MockProvider::new("Summary text"));
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Messages(2))
            .keep(KeepSize::Messages(1))
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        let mut state = AgentState::with_messages(vec![
            Message::user("First"),
            Message::assistant("Second"),
            Message::user("Third"),
        ]);

        let mut request = ModelRequest::new(state.messages.clone(), vec![]);
        let backend = Arc::new(crate::backends::MemoryBackend::new());
        let runtime = ToolRuntime::new(state.clone(), backend);

        let control = middleware
            .before_model(&mut request, &mut state, &runtime)
            .await
            .unwrap();

        assert!(matches!(control, ModelControl::ModifyRequest(_)));
        assert_eq!(request.messages.len(), state.messages.len());
        assert_eq!(request.messages[0].role, state.messages[0].role);
        assert_eq!(request.messages[0].content, state.messages[0].content);
        assert_eq!(request.messages[1].content, state.messages[1].content);
        assert_eq!(state.messages.len(), 2);
        assert!(state.messages[0].content.contains("Summary text"));
    }

    #[test]
    fn test_format_messages() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::default();
        let middleware = SummarizationMiddleware::new(provider, config);

        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let formatted = middleware.format_messages(&messages);

        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant: Hi there!"));
    }

    #[test]
    fn test_should_summarize_token_trigger() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Tokens(100))
            .max_input_tokens(200)
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        assert!(!middleware.should_summarize(50, 5));
        assert!(middleware.should_summarize(100, 5));
        assert!(middleware.should_summarize(150, 5));
    }

    #[test]
    fn test_should_summarize_fraction_trigger() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Fraction(0.8))
            .max_input_tokens(100)
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        assert!(!middleware.should_summarize(70, 5)); // 70% < 80%
        assert!(middleware.should_summarize(80, 5));   // 80% = 80%
        assert!(middleware.should_summarize(90, 5));   // 90% > 80%
    }

    #[tokio::test]
    async fn test_generate_summary() {
        let provider = Arc::new(MockProvider::new("This is the summary."));
        let config = SummarizationConfig::default();
        let middleware = SummarizationMiddleware::new(provider, config);

        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
        ];

        let summary = middleware.generate_summary(&messages).await.unwrap();
        assert_eq!(summary, "This is the summary.");
    }

    #[test]
    fn test_trim_for_summary() {
        let provider = Arc::new(MockProvider::new("Summary"));
        let config = SummarizationConfig::builder()
            .trim_tokens_to_summarize(50)
            .chars_per_token(1.0) // 1 char = 1 token for easy testing
            .overhead_per_message(0.0)
            .build();
        let middleware = SummarizationMiddleware::new(provider, config);

        let messages = vec![
            Message::user(&"A".repeat(30)),  // 30 tokens
            Message::user(&"B".repeat(30)),  // 30 tokens
            Message::user(&"C".repeat(30)),  // 30 tokens
        ];

        let trimmed = middleware.trim_for_summary(&messages);

        // Should only include messages that fit in 50 tokens
        // Last message is 30, fits. Second-to-last would be 60, doesn't fit.
        assert!(trimmed.len() <= 2);
    }
}
