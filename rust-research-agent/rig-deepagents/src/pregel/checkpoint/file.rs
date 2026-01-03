//! File-based Checkpointer Implementation
//!
//! Stores checkpoints as JSON files in a directory structure.
//! Supports optional compression via zstd for reduced storage.
//!
//! # Directory Structure
//!
//! ```text
//! checkpoints/
//! └── {workflow_id}/
//!     ├── checkpoint_00001.json[.zst]
//!     ├── checkpoint_00005.json[.zst]
//!     └── checkpoint_00010.json[.zst]
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{Checkpoint, Checkpointer};
use crate::pregel::error::PregelError;
use crate::pregel::state::WorkflowState;

/// File-based checkpointer that stores checkpoints as JSON files.
///
/// Each checkpoint is stored in a separate file, named by superstep number.
/// Atomic writes are ensured via temporary file + rename pattern.
#[derive(Debug)]
pub struct FileCheckpointer {
    /// Workflow-specific subdirectory
    workflow_path: PathBuf,
    /// Whether to compress checkpoints with zstd
    compression: bool,
}

impl FileCheckpointer {
    /// Create a new file-based checkpointer.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base directory for storing checkpoints
    /// * `workflow_id` - Unique identifier for this workflow
    /// * `compression` - Whether to compress checkpoint data
    pub fn new(base_path: impl Into<PathBuf>, workflow_id: impl AsRef<str>, compression: bool) -> Self {
        let base_path = base_path.into();
        let workflow_path = base_path.join(workflow_id.as_ref());

        Self {
            workflow_path,
            compression,
        }
    }

    /// Get the file path for a checkpoint at a given superstep
    fn checkpoint_path(&self, superstep: usize) -> PathBuf {
        let filename = if self.compression {
            format!("checkpoint_{:05}.json.zst", superstep)
        } else {
            format!("checkpoint_{:05}.json", superstep)
        };
        self.workflow_path.join(filename)
    }

    /// Get the temporary file path for atomic writes
    fn temp_path(&self, superstep: usize) -> PathBuf {
        let filename = format!("checkpoint_{:05}.tmp", superstep);
        self.workflow_path.join(filename)
    }

    /// Ensure the checkpoint directory exists
    async fn ensure_dir(&self) -> Result<(), PregelError> {
        fs::create_dir_all(&self.workflow_path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to create directory: {}", e)))
    }

    /// Compress data using zstd
    fn compress(data: &[u8]) -> Result<Vec<u8>, PregelError> {
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

    /// Parse superstep number from filename
    fn parse_superstep(path: &Path) -> Option<usize> {
        let filename = path.file_name()?.to_str()?;
        if !filename.starts_with("checkpoint_") {
            return None;
        }

        // Extract the number between "checkpoint_" and the extension
        let num_part = filename
            .strip_prefix("checkpoint_")?
            .split('.')
            .next()?;

        num_part.parse().ok()
    }

    /// List all superstep numbers (non-generic helper method)
    async fn list_supersteps(&self) -> Result<Vec<usize>, PregelError> {
        if !self.workflow_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = fs::read_dir(&self.workflow_path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to read directory: {}", e)))?;

        let mut supersteps = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to read entry: {}", e)))?
        {
            if let Some(superstep) = Self::parse_superstep(&entry.path()) {
                supersteps.push(superstep);
            }
        }

        supersteps.sort();
        Ok(supersteps)
    }
}

#[async_trait]
impl<S> Checkpointer<S> for FileCheckpointer
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError> {
        self.ensure_dir().await?;

        // Serialize checkpoint
        let json = serde_json::to_vec_pretty(checkpoint)
            .map_err(|e| PregelError::checkpoint_error(format!("Serialization failed: {}", e)))?;

        // Optionally compress
        let data = if self.compression {
            Self::compress(&json)?
        } else {
            json
        };

        // Write to temp file first (atomic write pattern)
        let temp_path = self.temp_path(checkpoint.superstep);
        let final_path = self.checkpoint_path(checkpoint.superstep);

        let mut file = fs::File::create(&temp_path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to create temp file: {}", e)))?;

        file.write_all(&data)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to write data: {}", e)))?;

        file.sync_all()
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to sync file: {}", e)))?;

        // Atomic rename
        fs::rename(&temp_path, &final_path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to rename file: {}", e)))?;

        Ok(())
    }

    async fn load(&self, superstep: usize) -> Result<Option<Checkpoint<S>>, PregelError> {
        let path = self.checkpoint_path(superstep);

        if !path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&path)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to open file: {}", e)))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .await
            .map_err(|e| PregelError::checkpoint_error(format!("Failed to read file: {}", e)))?;

        // Decompress if needed
        let json = if self.compression {
            Self::decompress(&data)?
        } else {
            data
        };

        let checkpoint: Checkpoint<S> = serde_json::from_slice(&json)
            .map_err(|e| PregelError::checkpoint_error(format!("Deserialization failed: {}", e)))?;

        Ok(Some(checkpoint))
    }

    async fn latest(&self) -> Result<Option<Checkpoint<S>>, PregelError> {
        // Use our own list_supersteps to avoid type inference issues
        let supersteps = self.list_supersteps().await?;
        match supersteps.last() {
            Some(&superstep) => self.load(superstep).await,
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<usize>, PregelError> {
        self.list_supersteps().await
    }

    async fn delete(&self, superstep: usize) -> Result<(), PregelError> {
        let path = self.checkpoint_path(superstep);

        if path.exists() {
            fs::remove_file(&path)
                .await
                .map_err(|e| PregelError::checkpoint_error(format!("Failed to delete file: {}", e)))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::state::UnitState;
    use crate::pregel::vertex::{VertexId, VertexState};
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_checkpointer_save_load() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

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
    async fn test_file_checkpointer_with_compression() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "compressed-workflow", true);

        // Create a checkpoint with some data
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

        // Verify the file is compressed (has .zst extension)
        let path = temp_dir.path().join("compressed-workflow/checkpoint_00010.json.zst");
        assert!(path.exists());

        // Load and verify
        let loaded: Checkpoint<UnitState> = checkpointer.load(10).await.unwrap().unwrap();
        assert_eq!(loaded.superstep, 10);
        assert_eq!(loaded.vertex_states.len(), 2);
    }

    #[tokio::test]
    async fn test_file_checkpointer_load_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

        let result: Option<Checkpoint<UnitState>> = checkpointer.load(999).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_file_checkpointer_list() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

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

        let list = <FileCheckpointer as Checkpointer<UnitState>>::list(&checkpointer).await.unwrap();
        assert_eq!(list, vec![1, 5, 10]); // Should be sorted
    }

    #[tokio::test]
    async fn test_file_checkpointer_latest() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

        // Save checkpoints
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
    async fn test_file_checkpointer_delete() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

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
        <FileCheckpointer as Checkpointer<UnitState>>::delete(&checkpointer, 5).await.unwrap();

        // Verify gone
        let gone: Option<Checkpoint<UnitState>> = checkpointer.load(5).await.unwrap();
        assert!(gone.is_none());
    }

    #[tokio::test]
    async fn test_file_checkpointer_prune() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

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
        let deleted = <FileCheckpointer as Checkpointer<UnitState>>::prune(&checkpointer, 2).await.unwrap();
        assert_eq!(deleted, 3);

        let remaining = <FileCheckpointer as Checkpointer<UnitState>>::list(&checkpointer).await.unwrap();
        assert_eq!(remaining, vec![4, 5]);
    }

    #[tokio::test]
    async fn test_file_checkpointer_atomic_write() {
        let temp_dir = tempdir().unwrap();
        let checkpointer = FileCheckpointer::new(temp_dir.path(), "test-workflow", false);

        let checkpoint = Checkpoint::new(
            "test-workflow",
            7,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );

        checkpointer.save(&checkpoint).await.unwrap();

        // Verify no temp file remains
        let temp_path = temp_dir.path().join("test-workflow/checkpoint_00007.tmp");
        assert!(!temp_path.exists());

        // Verify final file exists
        let final_path = temp_dir.path().join("test-workflow/checkpoint_00007.json");
        assert!(final_path.exists());
    }

    #[test]
    fn test_parse_superstep() {
        assert_eq!(
            FileCheckpointer::parse_superstep(Path::new("checkpoint_00005.json")),
            Some(5)
        );
        assert_eq!(
            FileCheckpointer::parse_superstep(Path::new("checkpoint_00123.json.zst")),
            Some(123)
        );
        assert_eq!(
            FileCheckpointer::parse_superstep(Path::new("other_file.json")),
            None
        );
    }
}
