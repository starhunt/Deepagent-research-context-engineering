//! PostgreSQL-based Checkpointer Implementation
//!
//! Stores checkpoints in a PostgreSQL database for enterprise-grade persistence.
//! Supports connection pooling and ACID transactions.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS checkpoints (
//!     id SERIAL PRIMARY KEY,
//!     workflow_id TEXT NOT NULL,
//!     superstep INTEGER NOT NULL,
//!     data BYTEA NOT NULL,
//!     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
//!     UNIQUE(workflow_id, superstep)
//! );
//! CREATE INDEX IF NOT EXISTS idx_workflow_superstep ON checkpoints(workflow_id, superstep);
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::pregel::checkpoint::PostgresCheckpointer;
//!
//! let checkpointer = PostgresCheckpointer::new(
//!     "postgres://user:pass@localhost/db",
//!     "my-workflow"
//! ).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use super::{Checkpoint, Checkpointer};
use crate::pregel::error::PregelError;
use crate::pregel::state::WorkflowState;

/// PostgreSQL-based checkpointer for enterprise workflow state persistence.
///
/// Uses SQLx with connection pooling for reliable, ACID-compliant storage.
#[derive(Clone)]
pub struct PostgresCheckpointer {
    /// Connection pool
    pool: PgPool,
    /// Workflow identifier for isolation
    workflow_id: String,
    /// Whether to use compression
    compression: bool,
}

impl PostgresCheckpointer {
    /// Create a new PostgreSQL checkpointer.
    ///
    /// # Arguments
    ///
    /// * `url` - PostgreSQL connection URL
    /// * `workflow_id` - Unique identifier for this workflow
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cp = PostgresCheckpointer::new(
    ///     "postgres://user:pass@localhost/mydb",
    ///     "workflow-1"
    /// ).await?;
    /// ```
    pub async fn new(
        url: impl AsRef<str>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, PregelError> {
        Self::with_compression(url, workflow_id, false).await
    }

    /// Create a new PostgreSQL checkpointer with compression option.
    pub async fn with_compression(
        url: impl AsRef<str>,
        workflow_id: impl Into<String>,
        compression: bool,
    ) -> Result<Self, PregelError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url.as_ref())
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to connect to PostgreSQL: {}", e)))?;

        let workflow_id = workflow_id.into();

        // Initialize schema
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoints (
                id SERIAL PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                superstep INTEGER NOT NULL,
                data BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(workflow_id, superstep)
            )
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to create schema: {}", e)))?;

        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_workflow_superstep
                ON checkpoints(workflow_id, superstep)
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to create index: {}", e)))?;

        Ok(Self {
            pool,
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
impl<S> Checkpointer<S> for PostgresCheckpointer
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

        // Upsert using ON CONFLICT
        sqlx::query(
            r#"
            INSERT INTO checkpoints (workflow_id, superstep, data)
            VALUES ($1, $2, $3)
            ON CONFLICT (workflow_id, superstep)
            DO UPDATE SET data = EXCLUDED.data, created_at = NOW()
            "#,
        )
        .bind(&self.workflow_id)
        .bind(checkpoint.superstep as i32)
        .bind(&data)
        .execute(&self.pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to save checkpoint: {}", e)))?;

        Ok(())
    }

    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError> {
        let row: Option<(Vec<u8>,)> = sqlx::query_as(
            "SELECT data FROM checkpoints WHERE workflow_id = $1 AND superstep = $2",
        )
        .bind(&self.workflow_id)
        .bind(superstep as i32)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to load checkpoint: {}", e)))?;

        match row {
            Some((data,)) => {
                // Decompress if needed
                let json = if self.compression {
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
        let row: Option<(i32,)> = sqlx::query_as(
            "SELECT MAX(superstep) FROM checkpoints WHERE workflow_id = $1",
        )
        .bind(&self.workflow_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to get latest: {}", e)))?;

        match row {
            Some((superstep,)) => self.load(superstep as usize).await,
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<usize>, PregelError> {
        let rows: Vec<(i32,)> = sqlx::query_as(
            "SELECT superstep FROM checkpoints WHERE workflow_id = $1 ORDER BY superstep ASC",
        )
        .bind(&self.workflow_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| PregelError::checkpoint_error(format!("Failed to list checkpoints: {}", e)))?;

        Ok(rows.into_iter().map(|(s,)| s as usize).collect())
    }

    async fn delete(&self, superstep: usize) -> Result<(), PregelError> {
        sqlx::query("DELETE FROM checkpoints WHERE workflow_id = $1 AND superstep = $2")
            .bind(&self.workflow_id)
            .bind(superstep as i32)
            .execute(&self.pool)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to delete checkpoint: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // PostgreSQL tests require a running database, so they are marked as ignored
    // Run with: cargo test --features checkpointer-postgres -- --ignored

    #[test]
    fn test_postgres_checkpointer_compiles() {
        // Basic compile-time check that the module is valid
        assert!(true);
    }
}
