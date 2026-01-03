//! Summarization Configuration
//!
//! Configuration types for the SummarizationMiddleware.

use super::trigger::{KeepSize, TriggerCondition};
use super::token_counter::DEFAULT_CHARS_PER_TOKEN;

/// Default summarization prompt (ported from LangChain DeepAgents)
pub const DEFAULT_SUMMARY_PROMPT: &str = r#"<role>Context Extraction Assistant</role>

<primary_objective>
Extract the highest quality and most relevant context from the conversation history.
</primary_objective>

<context>
You are approaching your token limit and must extract the most important information
from the conversation history. This extracted context will replace the older messages.
</context>

<instructions>
1. Focus on key decisions, findings, and important context
2. Preserve critical technical details and file paths mentioned
3. Don't repeat actions that have already been completed
4. Summarize the overall goal and current progress
5. Keep information that will be needed for future steps
6. Be concise but preserve essential details

Respond ONLY with the extracted context. Do not include any additional commentary.
</instructions>

<conversation_to_summarize>"#;

/// Configuration for the SummarizationMiddleware.
///
/// Controls when summarization triggers and how much context to keep.
///
/// # Example
///
/// ```rust,ignore
/// use rig_deepagents::middleware::summarization::{SummarizationConfig, TriggerCondition, KeepSize};
///
/// // Default configuration (85% trigger, 10% keep)
/// let config = SummarizationConfig::default();
///
/// // Custom configuration
/// let config = SummarizationConfig::builder()
///     .trigger(TriggerCondition::Tokens(170_000))
///     .keep(KeepSize::Messages(10))
///     .max_input_tokens(200_000)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct SummarizationConfig {
    /// Trigger conditions (OR logic - any match triggers summarization)
    pub triggers: Vec<TriggerCondition>,

    /// How much context to keep after summarization
    pub keep: KeepSize,

    /// Maximum tokens to include when calling the summarizer
    pub trim_tokens_to_summarize: usize,

    /// Characters per token ratio for counting
    pub chars_per_token: f32,

    /// Overhead tokens per message
    pub overhead_per_message: f32,

    /// Custom summarization prompt (uses default if None)
    pub summary_prompt: String,

    /// Model's maximum input token limit
    pub max_input_tokens: usize,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            triggers: vec![TriggerCondition::Fraction(0.85)],
            keep: KeepSize::Fraction(0.10),
            trim_tokens_to_summarize: 4000,
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            overhead_per_message: 3.0,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            max_input_tokens: 128_000, // Default for GPT-4 Turbo
        }
    }
}

impl SummarizationConfig {
    /// Create a new builder for SummarizationConfig
    pub fn builder() -> SummarizationConfigBuilder {
        SummarizationConfigBuilder::default()
    }

    /// Create a config with common presets for a model
    pub fn for_model(model: &str) -> Self {
        let mut config = Self::default();

        let model_lower = model.to_lowercase();

        // Set model-specific parameters
        if model_lower.contains("claude") {
            config.chars_per_token = 3.3;
            // All Claude 3+ models (Opus, Sonnet, Haiku) have 200K context window
            config.max_input_tokens = 200_000;
        } else if model_lower.contains("gpt-4") {
            config.chars_per_token = 4.0;
            if model_lower.contains("turbo") || model_lower.contains("128k") {
                config.max_input_tokens = 128_000;
            } else if model_lower.contains("32k") {
                config.max_input_tokens = 32_768;
            } else {
                config.max_input_tokens = 8_192;
            }
        } else if model_lower.contains("gpt-3.5") {
            config.chars_per_token = 4.0;
            config.max_input_tokens = 16_385;
        }

        config
    }

    /// Check if summarization should be triggered based on current state
    pub fn should_summarize(&self, token_count: usize, message_count: usize) -> bool {
        self.triggers
            .iter()
            .any(|t| t.should_trigger(token_count, message_count, self.max_input_tokens))
    }
}

/// Builder for SummarizationConfig
#[derive(Debug, Default)]
pub struct SummarizationConfigBuilder {
    triggers: Option<Vec<TriggerCondition>>,
    keep: Option<KeepSize>,
    trim_tokens_to_summarize: Option<usize>,
    chars_per_token: Option<f32>,
    overhead_per_message: Option<f32>,
    summary_prompt: Option<String>,
    max_input_tokens: Option<usize>,
}

impl SummarizationConfigBuilder {
    /// Add a trigger condition
    pub fn trigger(mut self, trigger: TriggerCondition) -> Self {
        self.triggers
            .get_or_insert_with(Vec::new)
            .push(trigger);
        self
    }

    /// Set multiple trigger conditions
    pub fn triggers(mut self, triggers: Vec<TriggerCondition>) -> Self {
        self.triggers = Some(triggers);
        self
    }

    /// Set the keep size
    pub fn keep(mut self, keep: KeepSize) -> Self {
        self.keep = Some(keep);
        self
    }

    /// Set the maximum tokens to send to summarizer
    pub fn trim_tokens_to_summarize(mut self, tokens: usize) -> Self {
        self.trim_tokens_to_summarize = Some(tokens);
        self
    }

    /// Set the characters per token ratio
    pub fn chars_per_token(mut self, ratio: f32) -> Self {
        self.chars_per_token = Some(ratio);
        self
    }

    /// Set the overhead tokens per message
    pub fn overhead_per_message(mut self, overhead: f32) -> Self {
        self.overhead_per_message = Some(overhead);
        self
    }

    /// Set a custom summary prompt
    pub fn summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = Some(prompt.into());
        self
    }

    /// Set the model's maximum input tokens
    pub fn max_input_tokens(mut self, tokens: usize) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Build the configuration
    pub fn build(self) -> SummarizationConfig {
        let default = SummarizationConfig::default();

        SummarizationConfig {
            triggers: self.triggers.unwrap_or(default.triggers),
            keep: self.keep.unwrap_or(default.keep),
            trim_tokens_to_summarize: self
                .trim_tokens_to_summarize
                .unwrap_or(default.trim_tokens_to_summarize),
            chars_per_token: self.chars_per_token.unwrap_or(default.chars_per_token),
            overhead_per_message: self
                .overhead_per_message
                .unwrap_or(default.overhead_per_message),
            summary_prompt: self.summary_prompt.unwrap_or(default.summary_prompt),
            max_input_tokens: self.max_input_tokens.unwrap_or(default.max_input_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SummarizationConfig::default();

        assert_eq!(config.triggers.len(), 1);
        assert!(matches!(config.triggers[0], TriggerCondition::Fraction(f) if (f - 0.85).abs() < 0.001));
        assert!(matches!(config.keep, KeepSize::Fraction(f) if (f - 0.10).abs() < 0.001));
        assert_eq!(config.trim_tokens_to_summarize, 4000);
        assert_eq!(config.max_input_tokens, 128_000);
    }

    #[test]
    fn test_for_model_claude() {
        let config = SummarizationConfig::for_model("claude-3-opus");

        assert_eq!(config.chars_per_token, 3.3);
        assert_eq!(config.max_input_tokens, 200_000);
    }

    #[test]
    fn test_for_model_gpt4() {
        let config = SummarizationConfig::for_model("gpt-4-turbo");

        assert_eq!(config.chars_per_token, 4.0);
        assert_eq!(config.max_input_tokens, 128_000);
    }

    #[test]
    fn test_should_summarize() {
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Tokens(100))
            .max_input_tokens(200)
            .build();

        assert!(!config.should_summarize(50, 5));
        assert!(config.should_summarize(100, 5));
        assert!(config.should_summarize(150, 5));
    }

    #[test]
    fn test_builder() {
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Tokens(170_000))
            .trigger(TriggerCondition::Messages(100))
            .keep(KeepSize::Messages(6))
            .max_input_tokens(200_000)
            .chars_per_token(3.3)
            .build();

        assert_eq!(config.triggers.len(), 2);
        assert!(matches!(config.keep, KeepSize::Messages(6)));
        assert_eq!(config.max_input_tokens, 200_000);
        assert_eq!(config.chars_per_token, 3.3);
    }

    #[test]
    fn test_or_logic_triggers() {
        let config = SummarizationConfig::builder()
            .trigger(TriggerCondition::Tokens(1000))
            .trigger(TriggerCondition::Messages(10))
            .max_input_tokens(2000)
            .build();

        // Neither condition met
        assert!(!config.should_summarize(500, 5));

        // Only token condition met
        assert!(config.should_summarize(1000, 5));

        // Only message condition met
        assert!(config.should_summarize(500, 10));

        // Both conditions met
        assert!(config.should_summarize(1000, 10));
    }
}
