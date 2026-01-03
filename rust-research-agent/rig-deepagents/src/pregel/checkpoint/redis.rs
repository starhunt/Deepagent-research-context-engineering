//! Redis-based Checkpointer Implementation
//!
//! Stores checkpoints in Redis for distributed, high-performance persistence.
//! Supports optional TTL for automatic checkpoint expiration.
//!
//! # Key Format
//!
//! ```text
//! workflow:{workflow_id}:checkpoint:{superstep:05}
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::pregel::checkpoint::RedisCheckpointer;
//!
//! // Connect to Redis
//! let checkpointer = RedisCheckpointer::new("redis://localhost:6379", "my-workflow").await?;
//!
//! // With TTL (auto-expire after 1 hour)
//! let checkpointer = RedisCheckpointer::with_ttl("redis://localhost", "workflow", Some(3600)).await?;
//! ```

use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};

use super::{Checkpoint, Checkpointer};
use crate::pregel::error::PregelError;
use crate::pregel::state::WorkflowState;

/// Redis-based checkpointer for distributed workflow state persistence.
///
/// Uses Redis for fast, distributed checkpoint storage with optional TTL.
#[derive(Clone)]
pub struct RedisCheckpointer {
    /// Async Redis connection manager
    conn: ConnectionManager,
    /// Workflow identifier for key isolation
    workflow_id: String,
    /// Whether to use compression
    compression: bool,
    /// Optional TTL in seconds for checkpoint expiration
    ttl_seconds: Option<u64>,
}

impl RedisCheckpointer {
    /// Create a new Redis checkpointer.
    ///
    /// # Arguments
    ///
    /// * `url` - Redis connection URL (e.g., "redis://localhost:6379")
    /// * `workflow_id` - Unique identifier for this workflow
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cp = RedisCheckpointer::new("redis://localhost:6379", "workflow-1").await?;
    /// ```
    pub async fn new(
        url: impl AsRef<str>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, PregelError> {
        Self::with_options(url, workflow_id, false, None).await
    }

    /// Create a new Redis checkpointer with TTL.
    ///
    /// Checkpoints will automatically expire after the specified duration.
    pub async fn with_ttl(
        url: impl AsRef<str>,
        workflow_id: impl Into<String>,
        ttl_seconds: Option<u64>,
    ) -> Result<Self, PregelError> {
        Self::with_options(url, workflow_id, false, ttl_seconds).await
    }

    /// Create a new Redis checkpointer with all options.
    pub async fn with_options(
        url: impl AsRef<str>,
        workflow_id: impl Into<String>,
        compression: bool,
        ttl_seconds: Option<u64>,
    ) -> Result<Self, PregelError> {
        let client = redis::Client::open(url.as_ref())
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to create Redis client: {}", e)))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to connect to Redis: {}", e)))?;

        Ok(Self {
            conn,
            workflow_id: workflow_id.into(),
            compression,
            ttl_seconds,
        })
    }

    /// Generate the Redis key for a checkpoint
    fn checkpoint_key(&self, superstep: usize) -> String {
        format!("workflow:{}:checkpoint:{:05}", self.workflow_id, superstep)
    }

    /// Generate the pattern for listing checkpoints
    fn checkpoint_pattern(&self) -> String {
        format!("workflow:{}:checkpoint:*", self.workflow_id)
    }

    /// Parse superstep from a checkpoint key
    fn parse_superstep(key: &str) -> Option<usize> {
        key.split(':').last()?.parse().ok()
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
impl<S> Checkpointer<S> for RedisCheckpointer
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

        let key = self.checkpoint_key(checkpoint.superstep);
        let mut conn = self.conn.clone();

        // Set with optional TTL
        if let Some(ttl) = self.ttl_seconds {
            conn.set_ex::<_, _, ()>(&key, data.as_slice(), ttl)
                .await
                .map_err(|e| PregelError::checkpoint_error(format!("Failed to save checkpoint: {}", e)))?;
        } else {
            conn.set::<_, _, ()>(&key, data.as_slice())
                .await
                .map_err(|e| PregelError::checkpoint_error(format!("Failed to save checkpoint: {}", e)))?;
        }

        Ok(())
    }

    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError> {
        let key = self.checkpoint_key(superstep);
        let mut conn = self.conn.clone();

        let data: Option<Vec<u8>> = conn.get(&key)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to load checkpoint: {}", e)))?;

        match data {
            Some(data) => {
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
        let supersteps = <Self as Checkpointer<S>>::list(self).await?;

        match supersteps.last() {
            Some(&superstep) => self.load(superstep).await,
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<usize>, PregelError> {
        let pattern = self.checkpoint_pattern();
        let mut conn = self.conn.clone();

        let keys: Vec<String> = conn.keys(&pattern)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to list checkpoints: {}", e)))?;

        let mut supersteps: Vec<usize> = keys
            .iter()
            .filter_map(|key| Self::parse_superstep(key))
            .collect();

        supersteps.sort();
        Ok(supersteps)
    }

    async fn delete(&self, superstep: usize) -> Result<(), PregelError> {
        let key = self.checkpoint_key(superstep);
        let mut conn = self.conn.clone();

        conn.del::<_, ()>(&key)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to delete checkpoint: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_key_format() {
        // We can't easily test the full checkpointer without Redis,
        // but we can test the key generation
        let workflow_id = "test-workflow";
        let key = format!("workflow:{}:checkpoint:{:05}", workflow_id, 42);
        assert_eq!(key, "workflow:test-workflow:checkpoint:00042");
    }

    #[test]
    fn test_parse_superstep() {
        assert_eq!(
            RedisCheckpointer::parse_superstep("workflow:test:checkpoint:00042"),
            Some(42)
        );
        assert_eq!(
            RedisCheckpointer::parse_superstep("workflow:test:checkpoint:00001"),
            Some(1)
        );
        assert_eq!(
            RedisCheckpointer::parse_superstep("invalid"),
            None
        );
    }
}
