//! SQLite-based Checkpointer Implementation
//!
//! Stores checkpoints in a SQLite database for durable, queryable persistence.
//! Supports both file-based and in-memory databases.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS checkpoints (
//!     superstep INTEGER PRIMARY KEY,
//!     workflow_id TEXT NOT NULL,
//!     data BLOB NOT NULL,
//!     created_at TEXT NOT NULL
//! );
//! CREATE INDEX IF NOT EXISTS idx_workflow_superstep ON checkpoints(workflow_id, superstep);
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::pregel::checkpoint::SqliteCheckpointer;
//!
//! // File-based database
//! let checkpointer = SqliteCheckpointer::new("./checkpoints.db", "my-workflow").await?;
//!
//! // In-memory database (for testing)
//! let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow").await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio_rusqlite::Connection;

use super::{Checkpoint, Checkpointer};
use crate::pregel::error::PregelError;
use crate::pregel::state::WorkflowState;

/// SQLite-based checkpointer for durable workflow state persistence.
///
/// Uses SQLite's ACID guarantees for reliable checkpoint storage.
/// Supports both file-based and in-memory databases.
#[derive(Debug)]
pub struct SqliteCheckpointer {
    /// Async SQLite connection
    conn: Arc<Connection>,
    /// Workflow identifier for isolation
    workflow_id: String,
    /// Whether to use compression
    compression: bool,
}

impl SqliteCheckpointer {
    /// Create a new SQLite checkpointer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to SQLite database file, or `:memory:` for in-memory
    /// * `workflow_id` - Unique identifier for this workflow
    ///
    /// # Example
    ///
    /// ```ignore
    /// // File-based
    /// let cp = SqliteCheckpointer::new("./checkpoints.db", "workflow-1").await?;
    ///
    /// // In-memory (for testing)
    /// let cp = SqliteCheckpointer::new(":memory:", "test").await?;
    /// ```
    pub async fn new(
        path: impl AsRef<str>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, PregelError> {
        Self::with_compression(path, workflow_id, false).await
    }

    /// Create a new SQLite checkpointer with compression option.
    pub async fn with_compression(
        path: impl AsRef<str>,
        workflow_id: impl Into<String>,
        compression: bool,
    ) -> Result<Self, PregelError> {
        let path = path.as_ref().to_string();
        let workflow_id = workflow_id.into();

        let conn = Connection::open(&path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to open SQLite: {}", e)))?;

        // Initialize schema
        conn.call(|conn| {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    superstep INTEGER NOT NULL,
                    data BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(workflow_id, superstep)
                );
                CREATE INDEX IF NOT EXISTS idx_workflow_superstep
                    ON checkpoints(workflow_id, superstep);
                "#,
            )?;
            Ok(())
        })
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to create schema: {}", e)))?;

        Ok(Self {
            conn: Arc::new(conn),
            workflow_id,
            compression,
        })
    }

    /// Compress data using zstd
    fn compress(data: &[u8]) -> Result<Vec<u8>, PregelError> {
        use std::io::Write;
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3)
            .map_err(|e| PregelError::checkpoint_error(format!("Compression init failed: {}", e)))?;
        encoder
            .write_all(data)
            .map_err(|e| PregelError::checkpoint_error(format!("Compression write failed: {}", e)))?;
        encoder
            .finish()
            .map_err(|e| PregelError::checkpoint_error(format!("Compression finish failed: {}", e)))
    }

    /// Decompress data using zstd
    fn decompress(data: &[u8]) -> Result<Vec<u8>, PregelError> {
        zstd::stream::decode_all(data)
            .map_err(|e| PregelError::checkpoint_error(format!("Decompression failed: {}", e)))
    }
}

#[async_trait]
impl<S> Checkpointer<S> for SqliteCheckpointer
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static,
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError> {
        // Serialize checkpoint
        let json = serde_json::to_vec(checkpoint)
            .map_err(|e| PregelError::checkpoint_error(format!("Serialization failed: {}", e)))?;

        // Optionally compress
        let data = if self.compression {
            Self::compress(&json)?
        } else {
            json
        };

        let workflow_id = self.workflow_id.clone();
        let superstep = checkpoint.superstep;
        let created_at = checkpoint.timestamp.to_rfc3339();

        self.conn
            .call(move |conn| {
                conn.execute(
                    r#"
                    INSERT OR REPLACE INTO checkpoints (workflow_id, superstep, data, created_at)
                    VALUES (?1, ?2, ?3, ?4)
                    "#,
                    rusqlite::params![workflow_id, superstep as i64, data, created_at],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to save checkpoint: {}", e)))?;

        Ok(())
    }

    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError> {
        let workflow_id = self.workflow_id.clone();
        let compression = self.compression;

        let result = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT data FROM checkpoints WHERE workflow_id = ?1 AND superstep = ?2",
                )?;
                let mut rows = stmt.query(rusqlite::params![workflow_id, superstep as i64])?;

                if let Some(row) = rows.next()? {
                    let data: Vec<u8> = row.get(0)?;
                    Ok(Some(data))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to load checkpoint: {}", e)))?;

        match result {
            Some(data) => {
                // Decompress if needed
                let json = if compression {
                    Self::decompress(&data)?
                } else {
                    data
                };

                let checkpoint: Checkpoint<S> = serde_json::from_slice(&json).map_err(|e| {
                    PregelError::checkpoint_error(format!("Deserialization failed: {}", e))
                })?;

                Ok(Some(checkpoint))
            }
            None => Ok(None),
        }
    }

    async fn latest(&self) -> Result<Option<Checkpoint<S>>, PregelError> {
        let workflow_id = self.workflow_id.clone();

        let max_superstep = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT MAX(superstep) FROM checkpoints WHERE workflow_id = ?1",
                )?;
                let mut rows = stmt.query(rusqlite::params![workflow_id])?;

                if let Some(row) = rows.next()? {
                    let superstep: Option<i64> = row.get(0)?;
                    Ok(superstep.map(|s| s as usize))
                } else {
                    Ok(None)
                }
            })
            .await
            .map_err(|e| {
                PregelError::checkpoint_error(format!("Failed to get latest checkpoint: {}", e))
            })?;

        match max_superstep {
            Some(superstep) => self.load(superstep).await,
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<usize>, PregelError> {
        let workflow_id = self.workflow_id.clone();

        self.conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT superstep FROM checkpoints WHERE workflow_id = ?1 ORDER BY superstep ASC",
                )?;
                let rows = stmt.query_map(rusqlite::params![workflow_id], |row| {
                    let superstep: i64 = row.get(0)?;
                    Ok(superstep as usize)
                })?;

                let mut supersteps = Vec::new();
                for row in rows {
                    supersteps.push(row?);
                }
                Ok(supersteps)
            })
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to list checkpoints: {}", e)))
    }

    async fn delete(&self, superstep: usize) -> Result<(), PregelError> {
        let workflow_id = self.workflow_id.clone();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "DELETE FROM checkpoints WHERE workflow_id = ?1 AND superstep = ?2",
                    rusqlite::params![workflow_id, superstep as i64],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to delete checkpoint: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::message::WorkflowMessage;
    use crate::pregel::state::UnitState;
    use crate::pregel::vertex::{VertexId, VertexState};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_sqlite_checkpointer_save_load() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        let checkpoint = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );

        checkpointer.save(&checkpoint).await.unwrap();
        let loaded: Checkpoint<UnitState> = checkpointer.load(5).await.unwrap().unwrap();

        assert_eq!(loaded.superstep, 5);
        assert_eq!(loaded.workflow_id, "test-workflow");
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_with_compression() {
        let checkpointer = SqliteCheckpointer::with_compression(":memory:", "compressed-workflow", true)
            .await
            .unwrap();

        let mut vertex_states = HashMap::new();
        vertex_states.insert(VertexId::new("vertex1"), VertexState::Active);
        vertex_states.insert(VertexId::new("vertex2"), VertexState::Halted);

        let checkpoint = Checkpoint::new(
            "compressed-workflow",
            10,
            UnitState,
            vertex_states,
            HashMap::new(),
        );

        checkpointer.save(&checkpoint).await.unwrap();
        let loaded: Checkpoint<UnitState> = checkpointer.load(10).await.unwrap().unwrap();

        assert_eq!(loaded.superstep, 10);
        assert_eq!(loaded.vertex_states.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_load_nonexistent() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        let result: Option<Checkpoint<UnitState>> = checkpointer.load(999).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_list() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        // Save checkpoints at supersteps 5, 1, 10 (out of order)
        for superstep in [5, 1, 10] {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        let list = <SqliteCheckpointer as Checkpointer<UnitState>>::list(&checkpointer)
            .await
            .unwrap();
        assert_eq!(list, vec![1, 5, 10]); // Should be sorted
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_latest() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        for superstep in [1, 5, 3] {
            let checkpoint = Checkpoint::new(
                "test-workflow",
                superstep,
                UnitState,
                HashMap::new(),
                HashMap::new(),
            );
            checkpointer.save(&checkpoint).await.unwrap();
        }

        let latest: Checkpoint<UnitState> = checkpointer.latest().await.unwrap().unwrap();
        assert_eq!(latest.superstep, 5);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_delete() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        let checkpoint = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer.save(&checkpoint).await.unwrap();

        // Verify exists
        let exists: Option<Checkpoint<UnitState>> = checkpointer.load(5).await.unwrap();
        assert!(exists.is_some());

        // Delete
        <SqliteCheckpointer as Checkpointer<UnitState>>::delete(&checkpointer, 5)
            .await
            .unwrap();

        // Verify gone
        let gone: Option<Checkpoint<UnitState>> = checkpointer.load(5).await.unwrap();
        assert!(gone.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_prune() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        // Create 5 checkpoints
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
        let deleted = <SqliteCheckpointer as Checkpointer<UnitState>>::prune(&checkpointer, 2)
            .await
            .unwrap();
        assert_eq!(deleted, 3);

        let remaining = <SqliteCheckpointer as Checkpointer<UnitState>>::list(&checkpointer)
            .await
            .unwrap();
        assert_eq!(remaining, vec![4, 5]);
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_update_existing() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        // Save initial checkpoint
        let checkpoint1 = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer.save(&checkpoint1).await.unwrap();

        // Update with new data
        let mut vertex_states = HashMap::new();
        vertex_states.insert(VertexId::new("new-vertex"), VertexState::Completed);

        let checkpoint2 = Checkpoint::new(
            "test-workflow",
            5,
            UnitState,
            vertex_states,
            HashMap::new(),
        );
        checkpointer.save(&checkpoint2).await.unwrap();

        // Verify updated
        let loaded: Checkpoint<UnitState> = checkpointer.load(5).await.unwrap().unwrap();
        assert_eq!(loaded.vertex_states.len(), 1);
        assert!(loaded.vertex_states.contains_key(&VertexId::new("new-vertex")));
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_workflow_isolation() {
        let checkpointer1 = SqliteCheckpointer::new(":memory:", "workflow-1")
            .await
            .unwrap();
        let checkpointer2 = SqliteCheckpointer::new(":memory:", "workflow-2")
            .await
            .unwrap();

        // Save to workflow 1
        let checkpoint = Checkpoint::new(
            "workflow-1",
            1,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer1.save(&checkpoint).await.unwrap();

        // Workflow 2 should not see it
        let list = <SqliteCheckpointer as Checkpointer<UnitState>>::list(&checkpointer2)
            .await
            .unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_sqlite_checkpointer_with_messages() {
        let checkpointer = SqliteCheckpointer::new(":memory:", "test-workflow")
            .await
            .unwrap();

        let mut pending_messages = HashMap::new();
        pending_messages.insert(
            VertexId::new("vertex-a"),
            vec![
                WorkflowMessage::Activate,
                WorkflowMessage::Data {
                    key: "test".to_string(),
                    value: serde_json::json!("data"),
                },
            ],
        );

        let checkpoint = Checkpoint::new(
            "test-workflow",
            7,
            UnitState,
            HashMap::new(),
            pending_messages,
        );

        checkpointer.save(&checkpoint).await.unwrap();
        let loaded: Checkpoint<UnitState> = checkpointer.load(7).await.unwrap().unwrap();

        assert_eq!(loaded.pending_message_count(), 2);
    }
}
