//! Anthropic (Claude) LLM Provider implementation via Rig
//!
//! Provides Anthropic Claude API access through rig-core's Anthropic client.

use async_trait::async_trait;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::anthropic::Client;
use rig::providers::anthropic::completion::CLAUDE_3_5_SONNET;
use rig::completion::Prompt;

use super::config::LLMConfig;
use super::message::{extract_system_preamble};
use super::provider::{LLMProvider, LLMResponse};
use crate::error::DeepAgentError;
use crate::middleware::ToolDefinition;
use crate::state::Message;

/// Anthropic (Claude) LLM Provider
///
/// Wraps rig-core's Anthropic client to provide a unified LLMProvider interface.
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::llm::AnthropicProvider;
///
/// // Create from environment (ANTHROPIC_API_KEY)
/// let provider = AnthropicProvider::from_env()?;
///
/// // Or with explicit configuration
/// let provider = AnthropicProvider::new("sk-ant-...", "claude-3-5-sonnet-latest");
/// ```
///
/// # Note
///
/// Anthropic requires max_tokens to be set for all requests.
/// This provider defaults to 4096 tokens if not specified.
pub struct AnthropicProvider {
    client: Client,
    default_model: String,
    default_config: LLMConfig,
}

impl AnthropicProvider {
    /// Create from ANTHROPIC_API_KEY environment variable with default model
    pub fn from_env() -> Result<Self, DeepAgentError> {
        Self::from_env_with_model(CLAUDE_3_5_SONNET)
    }

    /// Create from environment with specific model
    pub fn from_env_with_model(model: impl Into<String>) -> Result<Self, DeepAgentError> {
        let client = Client::from_env();
        let model = model.into();

        Ok(Self {
            client,
            default_model: model.clone(),
            // Anthropic requires max_tokens
            default_config: LLMConfig::new(model).with_max_tokens(4096),
        })
    }

    /// Create with explicit API key and model
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self, DeepAgentError> {
        let api_key: String = api_key.into();
        let client = Client::from_val(api_key);
        let model = model.into();

        Ok(Self {
            client,
            default_model: model.clone(),
            default_config: LLMConfig::new(model).with_max_tokens(4096),
        })
    }

    /// Get effective configuration, preferring runtime config over defaults
    fn effective_config<'a>(&'a self, runtime: Option<&'a LLMConfig>) -> &'a LLMConfig {
        runtime.unwrap_or(&self.default_config)
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, DeepAgentError> {
        let config = self.effective_config(config);
        let model_name = &config.model;

        // Build agent with model
        let mut agent_builder = self.client.agent(model_name);

        // Set preamble from system messages
        if let Some(preamble) = extract_system_preamble(messages) {
            agent_builder = agent_builder.preamble(&preamble);
        }

        // Set temperature if specified
        if let Some(temp) = config.temperature {
            agent_builder = agent_builder.temperature(temp);
        }

        // Set max tokens (required for Anthropic)
        let max_tokens = config.max_tokens.unwrap_or(4096);
        agent_builder = agent_builder.max_tokens(max_tokens);

        let agent = agent_builder.build();

        // Get the last user message as the prompt
        let prompt = messages
            .iter()
            .rfind(|m| m.role == crate::state::Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Execute the prompt
        let response = agent
            .prompt(&prompt)
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Anthropic completion failed: {}", e)))?;

        Ok(LLMResponse::new(Message::assistant(&response)))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_provider_name() {
        // Just verify the trait implementation compiles
        fn assert_provider<T: LLMProvider>() {}
        assert_provider::<AnthropicProvider>();
    }

    #[tokio::test]
    #[ignore] // Requires ANTHROPIC_API_KEY environment variable
    async fn test_anthropic_provider_complete() {
        let provider = AnthropicProvider::from_env().unwrap();
        let messages = vec![Message::user("Say 'hello' and nothing else.")];

        let response = provider.complete(&messages, &[], None).await.unwrap();

        assert!(!response.message.content.is_empty());
    }

    #[tokio::test]
    #[ignore] // Requires ANTHROPIC_API_KEY environment variable
    async fn test_anthropic_provider_with_custom_config() {
        let provider = AnthropicProvider::from_env().unwrap();
        let messages = vec![Message::user("Say 'hello' briefly.")];

        let config = LLMConfig::new("claude-3-5-sonnet-latest")
            .with_temperature(0.0)
            .with_max_tokens(100);

        let response = provider
            .complete(&messages, &[], Some(&config))
            .await
            .unwrap();

        assert!(!response.message.content.is_empty());
    }
}
