//! LLM Provider trait definition
//!
//! Defines the core abstraction for interacting with LLM providers.
//! Implementations bridge to specific providers (OpenAI, Anthropic, etc.)
//! via Rig's CompletionModel trait.

use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;

use crate::error::DeepAgentError;
use crate::state::Message;
use crate::middleware::ToolDefinition;
use super::config::{LLMConfig, TokenUsage};

/// LLM completion response
///
/// Contains the assistant's response message along with optional
/// token usage statistics for cost tracking.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    /// The assistant's response message
    pub message: Message,
    /// Token usage statistics (if available from provider)
    pub usage: Option<TokenUsage>,
}

impl LLMResponse {
    /// Create a new response with just a message
    pub fn new(message: Message) -> Self {
        Self { message, usage: None }
    }

    /// Add token usage statistics to the response
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Streaming response chunk
///
/// Represents a single chunk of a streaming LLM response.
/// Used for real-time output display.
#[derive(Debug, Clone)]
pub struct MessageChunk {
    /// Content fragment for this chunk
    pub content: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Token usage (typically only in final chunk)
    pub usage: Option<TokenUsage>,
}

/// Streaming response wrapper
///
/// Wraps an async stream of message chunks for streaming completions.
pub struct LLMResponseStream {
    inner: Pin<Box<dyn Stream<Item = Result<MessageChunk, DeepAgentError>> + Send>>,
}

impl LLMResponseStream {
    /// Create a new stream from any compatible async stream
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<MessageChunk, DeepAgentError>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Create a stream from a complete (non-streaming) response
    ///
    /// Useful for providers that don't support streaming or as a fallback.
    pub fn from_complete(response: LLMResponse) -> Self {
        let content = response.message.content.clone();
        let chunk = MessageChunk {
            content,
            is_final: true,
            usage: response.usage,
        };
        Self::new(futures::stream::once(async move { Ok(chunk) }))
    }

    /// Get a reference to the inner stream
    pub fn into_inner(self) -> Pin<Box<dyn Stream<Item = Result<MessageChunk, DeepAgentError>> + Send>> {
        self.inner
    }
}

/// Core LLM Provider trait
///
/// Provides a provider-agnostic interface for LLM completion.
/// Implementations should bridge to specific providers (OpenAI, Anthropic, etc.)
/// via Rig's CompletionModel trait or direct API calls.
///
/// # Design Principles
///
/// This trait is inspired by LangChain's `BaseChatModel`:
/// - Provider-agnostic interface
/// - Support for both streaming and non-streaming
/// - Tool/function calling support
/// - Configuration override at call time
///
/// # Example Implementation
///
/// ```rust,ignore
/// use async_trait::async_trait;
/// use rig_deepagents::llm::{LLMProvider, LLMResponse, LLMConfig};
///
/// struct MyProvider { /* ... */ }
///
/// #[async_trait]
/// impl LLMProvider for MyProvider {
///     async fn complete(
///         &self,
///         messages: &[Message],
///         tools: &[ToolDefinition],
///         config: Option<&LLMConfig>,
///     ) -> Result<LLMResponse, DeepAgentError> {
///         // Implementation here
///     }
///
///     fn name(&self) -> &str { "my-provider" }
///     fn default_model(&self) -> &str { "my-model" }
/// }
/// ```
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Generate a completion response (non-streaming)
    ///
    /// # Arguments
    /// * `messages` - Conversation history including the current prompt
    /// * `tools` - Available tools for the model to call
    /// * `config` - Optional runtime configuration overrides
    ///
    /// # Returns
    /// The assistant's response with optional token usage
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, DeepAgentError>;

    /// Generate a streaming completion response
    ///
    /// Default implementation falls back to non-streaming `complete()`.
    /// Override for providers that support native streaming.
    ///
    /// # Arguments
    /// * `messages` - Conversation history including the current prompt
    /// * `tools` - Available tools for the model to call
    /// * `config` - Optional runtime configuration overrides
    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponseStream, DeepAgentError> {
        let response = self.complete(messages, tools, config).await?;
        Ok(LLMResponseStream::from_complete(response))
    }

    /// Provider name for logging/debugging
    fn name(&self) -> &str;

    /// Default model identifier for this provider
    fn default_model(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Role;

    /// Mock provider for testing
    struct MockProvider {
        response_prefix: String,
    }

    impl MockProvider {
        fn new(prefix: impl Into<String>) -> Self {
            Self {
                response_prefix: prefix.into(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn complete(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
            _config: Option<&LLMConfig>,
        ) -> Result<LLMResponse, DeepAgentError> {
            let last_content = messages
                .last()
                .map(|m| m.content.as_str())
                .unwrap_or("Hello");

            let response_content = format!("{}: {}", self.response_prefix, last_content);
            Ok(LLMResponse::new(Message::assistant(&response_content)))
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model-v1"
        }
    }

    #[tokio::test]
    async fn test_mock_provider_complete() {
        let provider = MockProvider::new("Echo");
        let messages = vec![Message::user("Hello, world!")];

        let response = provider.complete(&messages, &[], None).await.unwrap();

        assert!(response.message.content.contains("Echo:"));
        assert!(response.message.content.contains("Hello, world!"));
        assert_eq!(response.message.role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_mock_provider_with_config() {
        let provider = MockProvider::new("Test");
        let messages = vec![Message::user("Config test")];
        let config = LLMConfig::new("custom-model").with_temperature(0.5);

        let response = provider.complete(&messages, &[], Some(&config)).await.unwrap();

        assert!(response.message.content.contains("Config test"));
    }

    #[tokio::test]
    async fn test_stream_fallback() {
        let provider = MockProvider::new("Stream");
        let messages = vec![Message::user("Test streaming")];

        let stream = provider.stream(&messages, &[], None).await.unwrap();

        // Stream should be created successfully via fallback
        // (actual consumption would require polling the stream)
        let _ = stream.into_inner();
    }

    #[test]
    fn test_llm_response_with_usage() {
        let message = Message::assistant("Hello");
        let usage = TokenUsage::new(10, 5);

        let response = LLMResponse::new(message).with_usage(usage.clone());

        assert_eq!(response.usage, Some(usage));
    }

    #[test]
    fn test_message_chunk() {
        let chunk = MessageChunk {
            content: "Hello".to_string(),
            is_final: true,
            usage: Some(TokenUsage::new(5, 3)),
        };

        assert_eq!(chunk.content, "Hello");
        assert!(chunk.is_final);
        assert!(chunk.usage.is_some());
    }
}
