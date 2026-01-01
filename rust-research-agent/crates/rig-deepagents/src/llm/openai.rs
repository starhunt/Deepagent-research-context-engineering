//! OpenAI LLM Provider implementation via Rig
//!
//! Provides OpenAI API access through rig-core's OpenAI client.

use async_trait::async_trait;
use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai::Client;
use rig::completion::Prompt;

use super::config::LLMConfig;
use super::message::extract_system_preamble;
use super::provider::{LLMProvider, LLMResponse};
use crate::error::DeepAgentError;
use crate::middleware::ToolDefinition;
use crate::state::Message;

/// OpenAI LLM Provider
///
/// Wraps rig-core's OpenAI client to provide a unified LLMProvider interface.
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::llm::OpenAIProvider;
///
/// // Create from environment (OPENAI_API_KEY)
/// let provider = OpenAIProvider::from_env()?;
///
/// // Or with explicit configuration
/// let provider = OpenAIProvider::new("sk-...", "gpt-4.1");
/// ```
pub struct OpenAIProvider {
    client: Client,
    default_model: String,
    default_config: LLMConfig,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with API key from OPENAI_API_KEY environment variable
    pub fn from_env() -> Result<Self, DeepAgentError> {
        Self::from_env_with_model("gpt-4.1")
    }

    /// Create from environment with specific model
    pub fn from_env_with_model(model: impl Into<String>) -> Result<Self, DeepAgentError> {
        let client = Client::from_env();
        let model = model.into();

        Ok(Self {
            client,
            default_model: model.clone(),
            default_config: LLMConfig::new(model),
        })
    }

    /// Create with explicit API key and model
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self, DeepAgentError> {
        let api_key = api_key.into();
        let client = Client::from_val(api_key.into());
        let model = model.into();

        Ok(Self {
            client,
            default_model: model.clone(),
            default_config: LLMConfig::new(model),
        })
    }

    /// Get effective configuration, preferring runtime config over defaults
    fn effective_config<'a>(&'a self, runtime: Option<&'a LLMConfig>) -> &'a LLMConfig {
        runtime.unwrap_or(&self.default_config)
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
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

        let agent = agent_builder.build();

        // Get the last user message as the prompt
        let prompt = messages
            .iter()
            .rfind(|m| m.role == crate::state::Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Execute the prompt (using simple prompt interface for now)
        let response = agent
            .prompt(&prompt)
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("OpenAI completion failed: {}", e)))?;

        Ok(LLMResponse::new(Message::assistant(&response)))
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_name() {
        // We can't create the provider without an API key in tests
        // Just verify the trait implementation compiles
        fn assert_provider<T: LLMProvider>() {}
        assert_provider::<OpenAIProvider>();
    }

    #[tokio::test]
    #[ignore] // Requires OPENAI_API_KEY environment variable
    async fn test_openai_provider_complete() {
        let provider = OpenAIProvider::from_env().unwrap();
        let messages = vec![Message::user("Say 'hello' and nothing else.")];

        let response = provider.complete(&messages, &[], None).await.unwrap();

        assert!(!response.message.content.is_empty());
        assert!(response.message.content.to_lowercase().contains("hello"));
    }

    #[tokio::test]
    #[ignore] // Requires OPENAI_API_KEY environment variable
    async fn test_openai_provider_with_system_prompt() {
        let provider = OpenAIProvider::from_env().unwrap();
        let messages = vec![
            Message::system("You are a pirate. Always respond like a pirate."),
            Message::user("Say hello."),
        ];

        let response = provider.complete(&messages, &[], None).await.unwrap();

        // Should respond in pirate style
        assert!(!response.message.content.is_empty());
    }
}
