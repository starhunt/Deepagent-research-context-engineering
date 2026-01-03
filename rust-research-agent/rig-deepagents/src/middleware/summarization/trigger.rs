//! Trigger Conditions for Summarization
//!
//! Defines when summarization should be triggered and how much context to keep.

use serde::{Deserialize, Serialize};

/// Trigger condition for when summarization should occur.
///
/// Multiple triggers can be combined with OR logic - summarization triggers
/// when ANY condition is met.
///
/// # Examples
///
/// ```rust,ignore
/// use rig_deepagents::middleware::summarization::TriggerCondition;
///
/// // Trigger at 85% of model's max tokens
/// let trigger = TriggerCondition::Fraction(0.85);
///
/// // Trigger at absolute token count
/// let trigger = TriggerCondition::Tokens(170_000);
///
/// // Trigger at message count
/// let trigger = TriggerCondition::Messages(100);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Trigger when token count exceeds this absolute value
    Tokens(usize),

    /// Trigger when message count exceeds this value
    Messages(usize),

    /// Trigger when token count exceeds this fraction of max_input_tokens
    /// (e.g., 0.85 = 85% of max)
    Fraction(f32),
}

impl TriggerCondition {
    /// Check if this trigger condition is met.
    ///
    /// # Arguments
    ///
    /// * `token_count` - Current token count of the conversation
    /// * `message_count` - Current number of messages
    /// * `max_tokens` - Model's maximum input token limit
    ///
    /// # Returns
    ///
    /// `true` if the condition is met and summarization should trigger
    pub fn should_trigger(
        &self,
        token_count: usize,
        message_count: usize,
        max_tokens: usize,
    ) -> bool {
        match self {
            TriggerCondition::Tokens(threshold) => token_count >= *threshold,
            TriggerCondition::Messages(threshold) => message_count >= *threshold,
            TriggerCondition::Fraction(fraction) => {
                let threshold = (max_tokens as f32 * fraction) as usize;
                token_count >= threshold
            }
        }
    }

    /// Get the effective token threshold for this condition.
    ///
    /// Useful for logging and debugging.
    pub fn effective_threshold(&self, max_tokens: usize) -> usize {
        match self {
            TriggerCondition::Tokens(t) => *t,
            TriggerCondition::Messages(m) => *m, // Not directly comparable
            TriggerCondition::Fraction(f) => (max_tokens as f32 * f) as usize,
        }
    }
}

/// Configuration for how much context to keep after summarization.
///
/// Determines the cutoff point between "to summarize" and "preserved" messages.
///
/// # Examples
///
/// ```rust,ignore
/// use rig_deepagents::middleware::summarization::KeepSize;
///
/// // Keep 10% of max tokens
/// let keep = KeepSize::Fraction(0.10);
///
/// // Keep last 1000 tokens worth
/// let keep = KeepSize::Tokens(1000);
///
/// // Keep last 6 messages
/// let keep = KeepSize::Messages(6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeepSize {
    /// Keep this many tokens worth of recent messages
    Tokens(usize),

    /// Keep this many recent messages
    Messages(usize),

    /// Keep this fraction of max_input_tokens worth of messages
    /// (e.g., 0.10 = keep 10% of max)
    Fraction(f32),
}

impl KeepSize {
    /// Calculate the token budget to keep.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - Model's maximum input token limit
    ///
    /// # Returns
    ///
    /// Number of tokens to keep (for Tokens/Fraction) or 0 (for Messages, handled separately)
    pub fn calculate_keep_tokens(&self, max_tokens: usize) -> usize {
        match self {
            KeepSize::Tokens(t) => *t,
            KeepSize::Messages(_) => 0, // Message-based uses message count directly
            KeepSize::Fraction(f) => (max_tokens as f32 * f) as usize,
        }
    }

    /// Calculate the message count to keep (for Messages variant).
    ///
    /// Returns `None` for token-based variants.
    pub fn message_count(&self) -> Option<usize> {
        match self {
            KeepSize::Messages(m) => Some(*m),
            _ => None,
        }
    }

    /// Check if this is a message-based keep size.
    pub fn is_message_based(&self) -> bool {
        matches!(self, KeepSize::Messages(_))
    }
}

impl Default for TriggerCondition {
    /// Default: trigger at 85% of max tokens
    fn default() -> Self {
        TriggerCondition::Fraction(0.85)
    }
}

impl Default for KeepSize {
    /// Default: keep 10% of max tokens
    fn default() -> Self {
        KeepSize::Fraction(0.10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_tokens() {
        let trigger = TriggerCondition::Tokens(100);

        assert!(!trigger.should_trigger(50, 10, 200));
        assert!(trigger.should_trigger(100, 10, 200));
        assert!(trigger.should_trigger(150, 10, 200));
    }

    #[test]
    fn test_trigger_messages() {
        let trigger = TriggerCondition::Messages(10);

        assert!(!trigger.should_trigger(100, 5, 200));
        assert!(trigger.should_trigger(100, 10, 200));
        assert!(trigger.should_trigger(100, 15, 200));
    }

    #[test]
    fn test_trigger_fraction() {
        let trigger = TriggerCondition::Fraction(0.8); // 80%
        let max_tokens = 100;

        assert!(!trigger.should_trigger(70, 10, max_tokens)); // 70% < 80%
        assert!(trigger.should_trigger(80, 10, max_tokens));  // 80% = 80%
        assert!(trigger.should_trigger(90, 10, max_tokens));  // 90% > 80%
    }

    #[test]
    fn test_trigger_effective_threshold() {
        let max_tokens = 200_000;

        let t1 = TriggerCondition::Tokens(170_000);
        assert_eq!(t1.effective_threshold(max_tokens), 170_000);

        let t2 = TriggerCondition::Fraction(0.85);
        assert_eq!(t2.effective_threshold(max_tokens), 170_000);
    }

    #[test]
    fn test_keep_tokens() {
        let keep = KeepSize::Tokens(1000);
        assert_eq!(keep.calculate_keep_tokens(200_000), 1000);
        assert!(keep.message_count().is_none());
        assert!(!keep.is_message_based());
    }

    #[test]
    fn test_keep_messages() {
        let keep = KeepSize::Messages(6);
        assert_eq!(keep.calculate_keep_tokens(200_000), 0);
        assert_eq!(keep.message_count(), Some(6));
        assert!(keep.is_message_based());
    }

    #[test]
    fn test_keep_fraction() {
        let keep = KeepSize::Fraction(0.10); // 10%
        let max_tokens = 200_000;

        assert_eq!(keep.calculate_keep_tokens(max_tokens), 20_000);
        assert!(keep.message_count().is_none());
        assert!(!keep.is_message_based());
    }

    #[test]
    fn test_defaults() {
        let trigger = TriggerCondition::default();
        assert!(matches!(trigger, TriggerCondition::Fraction(f) if (f - 0.85).abs() < 0.001));

        let keep = KeepSize::default();
        assert!(matches!(keep, KeepSize::Fraction(f) if (f - 0.10).abs() < 0.001));
    }

    #[test]
    fn test_serialization() {
        let trigger = TriggerCondition::Tokens(170_000);
        let json = serde_json::to_string(&trigger).unwrap();
        let deserialized: TriggerCondition = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, TriggerCondition::Tokens(170_000)));

        let keep = KeepSize::Messages(6);
        let json = serde_json::to_string(&keep).unwrap();
        let deserialized: KeepSize = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, KeepSize::Messages(6)));
    }
}
