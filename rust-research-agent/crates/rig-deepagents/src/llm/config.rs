//! LLM configuration types
//!
//! Provides configuration and usage tracking types for LLM providers.

use serde::{Deserialize, Serialize};

/// Token usage statistics from an LLM completion.
///
/// Tracks the number of tokens consumed during a request, enabling
/// cost tracking and context window management.
///
/// # Example
///
/// ```
/// use rig_deepagents::llm::TokenUsage;
///
/// let usage = TokenUsage::new(100, 50);
/// assert_eq!(usage.total_tokens, 150);
///
/// // Usage can be accumulated across multiple requests
/// let total = usage.clone() + TokenUsage::new(200, 75);
/// assert_eq!(total.total_tokens, 425);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Number of tokens in the input/prompt
    pub input_tokens: u64,
    /// Number of tokens in the generated output
    pub output_tokens: u64,
    /// Total tokens (input + output)
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Create a new TokenUsage with calculated total
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }

    /// Create from rig-core Usage struct
    pub fn from_rig_usage(usage: &rig::completion::Usage) -> Self {
        Self::new(usage.input_tokens, usage.output_tokens)
    }
}

impl std::ops::Add for TokenUsage {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
        }
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// LLM Provider configuration
///
/// Controls how an LLM provider generates completions. Configuration
/// can be set at provider construction time or overridden per-request.
///
/// # Example
///
/// ```
/// use rig_deepagents::llm::LLMConfig;
///
/// let config = LLMConfig::new("gpt-4.1")
///     .with_temperature(0.7)
///     .with_max_tokens(4096);
///
/// assert_eq!(config.model, "gpt-4.1");
/// assert_eq!(config.temperature, Some(0.7));
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Model identifier (e.g., "gpt-4.1", "claude-3-5-sonnet-20241022")
    pub model: String,
    /// Sampling temperature (0.0 - 2.0)
    /// Lower values are more deterministic, higher values more creative
    pub temperature: Option<f64>,
    /// Maximum tokens to generate in the response
    pub max_tokens: Option<u64>,
    /// API key (optional, can use environment variable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// API base URL (optional, for custom endpoints)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

impl LLMConfig {
    /// Create a new configuration with the specified model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set the sampling temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the maximum tokens to generate
    pub fn with_max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the API key explicitly
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set a custom API base URL
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = Some(base.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_token_usage_add() {
        let a = TokenUsage::new(100, 50);
        let b = TokenUsage::new(200, 100);
        let c = a + b;

        assert_eq!(c.input_tokens, 300);
        assert_eq!(c.output_tokens, 150);
        assert_eq!(c.total_tokens, 450);
    }

    #[test]
    fn test_token_usage_add_assign() {
        let mut usage = TokenUsage::new(100, 50);
        usage += TokenUsage::new(50, 25);

        assert_eq!(usage.input_tokens, 150);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total_tokens, 225);
    }

    #[test]
    fn test_llm_config_builder() {
        let config = LLMConfig::new("gpt-4.1")
            .with_temperature(0.7)
            .with_max_tokens(4096);

        assert_eq!(config.model, "gpt-4.1");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(4096));
    }

    #[test]
    fn test_llm_config_with_api_key() {
        let config = LLMConfig::new("gpt-4.1")
            .with_api_key("test-key")
            .with_api_base("https://custom.api.com");

        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.api_base, Some("https://custom.api.com".to_string()));
    }

    #[test]
    fn test_llm_config_serialization() {
        let config = LLMConfig::new("gpt-4.1")
            .with_temperature(0.5);

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("gpt-4.1"));
        assert!(json.contains("0.5"));
        // api_key should be skipped when None
        assert!(!json.contains("api_key"));
    }
}
