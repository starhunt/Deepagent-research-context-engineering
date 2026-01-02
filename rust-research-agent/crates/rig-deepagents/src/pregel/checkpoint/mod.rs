//! Checkpointing System for Pregel Runtime
//!
//! Provides durable state persistence for fault-tolerant workflow execution.
//! Checkpoints capture the complete workflow state at superstep boundaries,
//! enabling recovery from failures without losing progress.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Checkpointer                              │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
//! │  │   File   │  │  SQLite  │  │  Redis   │  │ Postgres │    │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
//! │        │            │            │            │              │
//! │        └────────────┴────────────┴────────────┘              │
//! │                          │                                    │
//! │                          ▼                                    │
//! │              Checkpoint<S: WorkflowState>                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::pregel::checkpoint::{Checkpointer, CheckpointerConfig, create_checkpointer};
//!
//! // Create a file-based checkpointer
//! let config = CheckpointerConfig::File {
//!     path: PathBuf::from("./checkpoints"),
//!     compression: true,
//! };
//! let checkpointer = create_checkpointer::<MyState>(config)?;
//!
//! // Save a checkpoint
//! checkpointer.save(&checkpoint).await?;
//!
//! // Load the latest checkpoint
//! if let Some(checkpoint) = checkpointer.latest().await? {
//!     // Resume from checkpoint
//! }
//! ```

mod file;
#[cfg(feature = "checkpointer-sqlite")]
mod sqlite;

pub use file::FileCheckpointer;
#[cfg(feature = "checkpointer-sqlite")]
pub use sqlite::SqliteCheckpointer;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;

use super::error::PregelError;
use super::message::WorkflowMessage;
use super::state::WorkflowState;
use super::vertex::{VertexId, VertexState};

/// A checkpoint captures the complete workflow state at a superstep boundary.
///
/// Checkpoints are the foundation of fault tolerance in the Pregel runtime.
/// They capture:
/// - The workflow state (user-defined data)
/// - The state of each vertex (Active, Halted, Completed)
/// - Pending messages that haven't been delivered yet
/// - Metadata for debugging and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint<S>
where
    S: WorkflowState,
{
    /// Unique identifier for the workflow instance
    pub workflow_id: String,

    /// The superstep number when this checkpoint was created
    pub superstep: usize,

    /// The workflow state at this superstep
    pub state: S,

    /// The state of each vertex (Active, Halted, Completed)
    pub vertex_states: HashMap<VertexId, VertexState>,

    /// Pending messages waiting to be delivered in the next superstep
    pub pending_messages: HashMap<VertexId, Vec<WorkflowMessage>>,

    /// When this checkpoint was created
    pub timestamp: DateTime<Utc>,

    /// Optional metadata for debugging or external tools
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl<S> Checkpoint<S>
where
    S: WorkflowState,
{
    /// Create a new checkpoint
    pub fn new(
        workflow_id: impl Into<String>,
        superstep: usize,
        state: S,
        vertex_states: HashMap<VertexId, VertexState>,
        pending_messages: HashMap<VertexId, Vec<WorkflowMessage>>,
    ) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            superstep,
            state,
            vertex_states,
            pending_messages,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this checkpoint
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if this checkpoint is empty (no vertex states or messages)
    pub fn is_empty(&self) -> bool {
        self.vertex_states.is_empty() && self.pending_messages.is_empty()
    }

    /// Get the total number of pending messages across all vertices
    pub fn pending_message_count(&self) -> usize {
        self.pending_messages.values().map(|v| v.len()).sum()
    }
}

/// Trait for checkpointing workflow state.
///
/// Implementations provide durable storage for checkpoints, enabling
/// recovery from failures and inspection of workflow history.
#[async_trait]
pub trait Checkpointer<S>: Send + Sync
where
    S: WorkflowState + Send + Sync,
{
    /// Save a checkpoint.
    ///
    /// Implementations should ensure atomic writes to prevent corruption.
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError>;

    /// Load a checkpoint by superstep number.
    ///
    /// Returns `None` if no checkpoint exists for that superstep.
    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError>;

    /// Load the latest checkpoint.
    ///
    /// Returns `None` if no checkpoints exist.
    async fn latest(&self) -> Result<Option<Checkpoint<S>>, PregelError>;

    /// List all available checkpoint superstep numbers, sorted ascending.
    async fn list(&self) -> Result<Vec<usize>, PregelError>;

    /// Delete a specific checkpoint.
    async fn delete(&self, superstep: usize) -> Result<(), PregelError>;

    /// Prune checkpoints, keeping only the most recent `keep` checkpoints.
    ///
    /// This is useful for managing storage space in long-running workflows.
    async fn prune(&self, keep: usize) -> Result<usize, PregelError> {
        let checkpoints = self.list().await?;
        let to_delete = checkpoints.len().saturating_sub(keep);
        let mut deleted = 0;

        for superstep in checkpoints.into_iter().take(to_delete) {
            self.delete(superstep).await?;
            deleted += 1;
        }

        Ok(deleted)
    }

    /// Clear all checkpoints for this workflow.
    async fn clear(&self) -> Result<(), PregelError> {
        for superstep in self.list().await? {
            self.delete(superstep).await?;
        }
        Ok(())
    }
}

/// Configuration for creating checkpointers.
///
/// Use with `create_checkpointer()` to instantiate the appropriate backend.
#[derive(Debug, Clone, Default)]
pub enum CheckpointerConfig {
    /// In-memory checkpointing (for testing only, not durable)
    #[default]
    Memory,

    /// File-based checkpointing
    File {
        /// Directory to store checkpoint files
        path: PathBuf,
        /// Whether to compress checkpoint data (uses zstd)
        compression: bool,
    },

    /// SQLite-based checkpointing (requires `checkpointer-sqlite` feature)
    #[cfg(feature = "checkpointer-sqlite")]
    Sqlite {
        /// Path to the SQLite database file, or `:memory:` for in-memory
        path: String,
    },

    /// Redis-based checkpointing (requires `checkpointer-redis` feature)
    #[cfg(feature = "checkpointer-redis")]
    Redis {
        /// Redis connection URL
        url: String,
        /// TTL for checkpoint keys (optional)
        ttl_seconds: Option<u64>,
    },

    /// PostgreSQL-based checkpointing (requires `checkpointer-postgres` feature)
    #[cfg(feature = "checkpointer-postgres")]
    Postgres {
        /// PostgreSQL connection URL
        url: String,
    },
}


/// In-memory checkpointer for testing.
///
/// This implementation stores checkpoints in memory and is not durable.
/// Use only for testing or development.
#[derive(Debug, Default)]
pub struct MemoryCheckpointer<S>
where
    S: WorkflowState,
{
    checkpoints: tokio::sync::RwLock<HashMap<usize, Checkpoint<S>>>,
}

impl<S> MemoryCheckpointer<S>
where
    S: WorkflowState,
{
    /// Create a new in-memory checkpointer
    pub fn new() -> Self {
        Self {
            checkpoints: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<S> Checkpointer<S> for MemoryCheckpointer<S>
where
    S: WorkflowState + Clone + Send + Sync,
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError> {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(checkpoint.superstep, checkpoint.clone());
        Ok(())
    }

    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError> {
        let checkpoints = self.checkpoints.read().await;
        Ok(checkpoints.get(&superstep).cloned())
    }

    async fn latest(&self) -> Result<Option<Checkpoint<S>>, PregelError> {
        let checkpoints = self.checkpoints.read().await;
        let max_superstep = checkpoints.keys().max().copied();
        match max_superstep {
            Some(superstep) => Ok(checkpoints.get(&superstep).cloned()),
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<usize>, PregelError> {
        let checkpoints = self.checkpoints.read().await;
        let mut supersteps: Vec<usize> = checkpoints.keys().copied().collect();
        supersteps.sort();
        Ok(supersteps)
    }

    async fn delete(&self, superstep: usize) -> Result<(), PregelError> {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.remove(&superstep);
        Ok(())
    }
}

/// Create a checkpointer from configuration.
///
/// This factory function creates the appropriate checkpointer backend
/// based on the provided configuration.
///
/// # Example
///
/// ```ignore
/// let config = CheckpointerConfig::File {
///     path: PathBuf::from("./checkpoints"),
///     compression: true,
/// };
/// let checkpointer = create_checkpointer::<MyState>(config, "workflow-123")?;
/// ```
pub fn create_checkpointer<S>(
    config: CheckpointerConfig,
    workflow_id: impl Into<String>,
) -> Result<Box<dyn Checkpointer<S>>, PregelError>
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    let workflow_id = workflow_id.into();

    match config {
        CheckpointerConfig::Memory => Ok(Box::new(MemoryCheckpointer::<S>::new())),

        CheckpointerConfig::File { path, compression } => {
            let checkpointer = FileCheckpointer::new(path, workflow_id, compression);
            Ok(Box::new(checkpointer))
        }

        #[cfg(feature = "checkpointer-sqlite")]
        CheckpointerConfig::Sqlite { path } => {
            // Note: We need to wrap this in a blocking call since create_checkpointer is sync
            // For async initialization, use SqliteCheckpointer::new() directly
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| PregelError::checkpoint_error("No tokio runtime available"))?;

            let checkpointer = rt.block_on(async {
                SqliteCheckpointer::new(&path, workflow_id).await
            })?;

            Ok(Box::new(checkpointer))
        }

        #[cfg(feature = "checkpointer-redis")]
        CheckpointerConfig::Redis { url, ttl_seconds } => {
            // Redis checkpointer will be implemented in Task 8.2.4
            Err(PregelError::not_implemented("Redis checkpointer"))
        }

        #[cfg(feature = "checkpointer-postgres")]
        CheckpointerConfig::Postgres { url } => {
            // PostgreSQL checkpointer will be implemented in Task 8.2.5
            Err(PregelError::not_implemented("PostgreSQL checkpointer"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::state::UnitState;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );

        assert_eq!(checkpoint.workflow_id, "test-workflow");
        assert_eq!(checkpoint.superstep, 5);
        assert!(checkpoint.is_empty());
        assert_eq!(checkpoint.pending_message_count(), 0);
    }

    #[test]
    fn test_checkpoint_with_metadata() {
        let checkpoint = Checkpoint::new(
            "test-workflow",
            10,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        )
        .with_metadata("version", "1.0")
        .with_metadata("creator", "test");

        assert_eq!(checkpoint.metadata.get("version"), Some(&"1.0".to_string()));
        assert_eq!(checkpoint.metadata.get("creator"), Some(&"test".to_string()));
    }

    #[test]
    fn test_checkpoint_with_vertex_states() {
        let mut vertex_states = HashMap::new();
        vertex_states.insert(VertexId::new("a"), VertexState::Active);
        vertex_states.insert(VertexId::new("b"), VertexState::Halted);

        let checkpoint = Checkpoint::new(
            "test-workflow",
            3,
            UnitState,
            vertex_states,
            HashMap::new(),
        );

        assert!(!checkpoint.is_empty());
        assert_eq!(checkpoint.vertex_states.len(), 2);
    }

    #[test]
    fn test_checkpoint_pending_message_count() {
        let mut pending_messages = HashMap::new();
        pending_messages.insert(
            VertexId::new("a"),
            vec![WorkflowMessage::Activate, WorkflowMessage::Activate],
        );
        pending_messages.insert(
            VertexId::new("b"),
            vec![WorkflowMessage::Activate],
        );

        let checkpoint = Checkpoint::new(
            "test-workflow",
            7,
            UnitState,
            HashMap::new(),
            pending_messages,
        );

        assert_eq!(checkpoint.pending_message_count(), 3);
    }

    #[tokio::test]
    async fn test_memory_checkpointer_save_load() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        let checkpoint = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );

        checkpointer.save(&checkpoint).await.unwrap();
        let loaded = checkpointer.load(5).await.unwrap().unwrap();

        assert_eq!(loaded.superstep, 5);
        assert_eq!(loaded.workflow_id, "test-workflow");
    }

    #[tokio::test]
    async fn test_memory_checkpointer_latest() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        // Save checkpoints at supersteps 1, 3, 5
        for superstep in [1, 3, 5] {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        let latest = checkpointer.latest().await.unwrap().unwrap();
        assert_eq!(latest.superstep, 5);
    }

    #[tokio::test]
    async fn test_memory_checkpointer_list() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        // Save checkpoints at supersteps 5, 1, 3 (out of order)
        for superstep in [5, 1, 3] {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list, vec![1, 3, 5]); // Should be sorted
    }

    #[tokio::test]
    async fn test_memory_checkpointer_delete() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        let checkpoint = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer.save(&checkpoint).await.unwrap();

        checkpointer.delete(5).await.unwrap();
        let loaded = checkpointer.load(5).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_memory_checkpointer_prune() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        // Save 5 checkpoints
        for superstep in 1..=5 {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        // Prune to keep only 2
        let deleted = checkpointer.prune(2).await.unwrap();
        assert_eq!(deleted, 3);

        let remaining = checkpointer.list().await.unwrap();
        assert_eq!(remaining, vec![4, 5]);
    }

    #[tokio::test]
    async fn test_memory_checkpointer_clear() {
        let checkpointer = MemoryCheckpointer::<UnitState>::new();

        for superstep in 1..=3 {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        checkpointer.clear().await.unwrap();
        let list = checkpointer.list().await.unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_checkpointer_config_default() {
        let config = CheckpointerConfig::default();
        assert!(matches!(config, CheckpointerConfig::Memory));
    }
}
