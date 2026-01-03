//! Pregel runtime configuration
//!
//! Configuration for the Pregel execution engine including
//! parallelism, timeouts, checkpointing, and retry policies.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Execution mode for the Pregel runtime
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// All vertices start Active. Vertices must explicitly send messages.
    /// Edges are stored but not used for automatic routing.
    /// This is the legacy behavior for backward compatibility.
    #[default]
    MessageBased,

    /// Only the entry vertex starts Active. Other vertices start Halted.
    /// When a vertex halts, Activate messages are automatically sent to edge targets.
    /// This matches LangGraph's execution model.
    EdgeDriven,
}

/// Pregel runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PregelConfig {
    /// Maximum supersteps before forced termination
    pub max_supersteps: usize,

    /// Maximum concurrent vertex computations
    pub parallelism: usize,

    /// Checkpoint frequency (every N supersteps, 0 = disabled)
    pub checkpoint_interval: usize,

    /// Timeout for individual vertex computation
    #[serde(with = "humantime_serde")]
    pub vertex_timeout: Duration,

    /// Timeout for entire workflow
    #[serde(with = "humantime_serde")]
    pub workflow_timeout: Duration,

    /// Enable detailed tracing
    pub tracing_enabled: bool,

    /// Retry policy for failed vertices
    pub retry_policy: RetryPolicy,

    /// Execution mode controlling vertex activation and edge routing
    pub execution_mode: ExecutionMode,
}

impl Default for PregelConfig {
    fn default() -> Self {
        Self {
            max_supersteps: 100,
            parallelism: num_cpus::get(),
            checkpoint_interval: 10,
            vertex_timeout: Duration::from_secs(300),    // 5 min per vertex
            workflow_timeout: Duration::from_secs(3600), // 1 hour total
            tracing_enabled: true,
            retry_policy: RetryPolicy::default(),
            execution_mode: ExecutionMode::default(),
        }
    }
}

impl PregelConfig {
    /// Create a new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum supersteps
    pub fn with_max_supersteps(mut self, max: usize) -> Self {
        self.max_supersteps = max;
        self
    }

    /// Set parallelism level
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism.max(1);
        self
    }

    /// Set checkpoint interval (0 to disable)
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Set vertex timeout
    pub fn with_vertex_timeout(mut self, timeout: Duration) -> Self {
        self.vertex_timeout = timeout;
        self
    }

    /// Set workflow timeout
    pub fn with_workflow_timeout(mut self, timeout: Duration) -> Self {
        self.workflow_timeout = timeout;
        self
    }

    /// Enable or disable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }

    /// Set retry policy
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Set the execution mode
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Check if checkpointing is enabled
    pub fn checkpointing_enabled(&self) -> bool {
        self.checkpoint_interval > 0
    }

    /// Check if a checkpoint should be taken at this superstep
    #[allow(clippy::manual_is_multiple_of)] // Using % for compatibility with older Rust versions
    pub fn should_checkpoint(&self, superstep: usize) -> bool {
        self.checkpointing_enabled() && superstep > 0 && superstep % self.checkpoint_interval == 0
    }
}

/// Retry policy for failed vertex computations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_retries: usize,

    /// Base delay for exponential backoff
    #[serde(with = "humantime_serde")]
    pub backoff_base: Duration,

    /// Maximum delay between retries
    #[serde(with = "humantime_serde")]
    pub backoff_max: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_base: Duration::from_millis(100),
            backoff_max: Duration::from_secs(10),
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set backoff base duration
    pub fn with_backoff_base(mut self, base: Duration) -> Self {
        self.backoff_base = base;
        self
    }

    /// Set maximum backoff duration
    pub fn with_backoff_max(mut self, max: Duration) -> Self {
        self.backoff_max = max;
        self
    }

    /// Calculate delay for a given retry attempt (exponential backoff)
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        let multiplier = 2u32.saturating_pow(attempt as u32);
        let delay = self.backoff_base.saturating_mul(multiplier);
        delay.min(self.backoff_max)
    }

    /// Check if more retries are allowed
    pub fn should_retry(&self, attempts: usize) -> bool {
        attempts < self.max_retries
    }

    /// Create a no-retry policy
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PregelConfig::default();
        assert_eq!(config.max_supersteps, 100);
        assert!(config.parallelism > 0);
        assert_eq!(config.checkpoint_interval, 10);
        assert!(config.tracing_enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = PregelConfig::default()
            .with_max_supersteps(50)
            .with_parallelism(4)
            .with_checkpoint_interval(5);

        assert_eq!(config.max_supersteps, 50);
        assert_eq!(config.parallelism, 4);
        assert_eq!(config.checkpoint_interval, 5);
    }

    #[test]
    fn test_parallelism_minimum() {
        let config = PregelConfig::default().with_parallelism(0);
        assert_eq!(config.parallelism, 1);
    }

    #[test]
    fn test_checkpointing_enabled() {
        let config = PregelConfig::default();
        assert!(config.checkpointing_enabled());

        let disabled = config.with_checkpoint_interval(0);
        assert!(!disabled.checkpointing_enabled());
    }

    #[test]
    fn test_should_checkpoint() {
        let config = PregelConfig::default().with_checkpoint_interval(5);

        assert!(!config.should_checkpoint(0));
        assert!(!config.should_checkpoint(1));
        assert!(config.should_checkpoint(5));
        assert!(config.should_checkpoint(10));
        assert!(!config.should_checkpoint(7));
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(policy.should_retry(0));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_retry_backoff() {
        let policy = RetryPolicy::default();

        let delay0 = policy.delay_for_attempt(0);
        let delay1 = policy.delay_for_attempt(1);
        let delay2 = policy.delay_for_attempt(2);

        assert_eq!(delay0, Duration::from_millis(100));
        assert_eq!(delay1, Duration::from_millis(200));
        assert_eq!(delay2, Duration::from_millis(400));
    }

    #[test]
    fn test_retry_backoff_max() {
        let policy = RetryPolicy::default().with_backoff_max(Duration::from_millis(300));

        let delay_high = policy.delay_for_attempt(10);
        assert_eq!(delay_high, Duration::from_millis(300));
    }

    #[test]
    fn test_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        assert!(!policy.should_retry(0));
    }

    #[test]
    fn test_config_with_timeouts() {
        let config = PregelConfig::default()
            .with_vertex_timeout(Duration::from_secs(60))
            .with_workflow_timeout(Duration::from_secs(120));

        assert_eq!(config.vertex_timeout, Duration::from_secs(60));
        assert_eq!(config.workflow_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_execution_mode_default_is_message_based() {
        let config = PregelConfig::default();
        assert_eq!(config.execution_mode, ExecutionMode::MessageBased);
    }

    #[test]
    fn test_execution_mode_builder() {
        let config = PregelConfig::default()
            .with_execution_mode(ExecutionMode::EdgeDriven);
        assert_eq!(config.execution_mode, ExecutionMode::EdgeDriven);
    }
}
