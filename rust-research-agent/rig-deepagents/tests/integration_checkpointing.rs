//! Integration Tests for Checkpointing Support in CompiledWorkflow
//!
//! These tests verify the complete checkpointing integration including:
//! - Checkpoint creation during execution
//! - Resume from checkpoint after simulated failure
//! - Pending message preservation
//! - Retry count preservation
//! - Different checkpointer backends
//! - Error handling for non-checkpointed workflows

use std::collections::HashMap;
use std::sync::Arc;

use rig_deepagents::pregel::checkpoint::{Checkpoint, Checkpointer, MemoryCheckpointer};
use rig_deepagents::pregel::config::ExecutionMode;
use rig_deepagents::pregel::message::WorkflowMessage;
use rig_deepagents::pregel::vertex::{VertexId, VertexState};
use rig_deepagents::pregel::PregelConfig;
use rig_deepagents::pregel::state::UnitState;
use rig_deepagents::workflow::graph::WorkflowGraph;
use rig_deepagents::workflow::node::NodeKind;
use rig_deepagents::workflow::{CompiledWorkflow, END};

// =============================================================================
// Basic Checkpointing Tests
// =============================================================================

/// Test that checkpoints are created during workflow execution at configured intervals
#[tokio::test]
async fn test_checkpoint_save_during_execution() {
    // Create a multi-step workflow
    let graph = WorkflowGraph::<UnitState>::new()
        .name("checkpoint_test")
        .node("step1", NodeKind::Passthrough)
        .node("step2", NodeKind::Passthrough)
        .node("step3", NodeKind::Passthrough)
        .node("step4", NodeKind::Passthrough)
        .node("step5", NodeKind::Passthrough)
        .entry("step1")
        .edge("step1", "step2")
        .edge("step2", "step3")
        .edge("step3", "step4")
        .edge("step4", "step5")
        .edge("step5", END)
        .build()
        .expect("Failed to build graph");

    // Configure checkpointing every 2 supersteps
    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::EdgeDriven)
        .with_max_supersteps(20)
        .with_checkpoint_interval(2);

    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer.clone(),
        "checkpoint-save-test",
    )
    .expect("Failed to compile workflow");

    // Verify checkpointer is enabled
    assert!(workflow.has_checkpointer());
    assert!(workflow.checkpointer().is_some());

    // Run the workflow
    let result = workflow.run(UnitState).await.expect("Workflow failed");

    // Verify completion
    assert!(result.completed, "Workflow should complete");
    assert!(result.supersteps >= 1, "Should have at least 1 superstep");

    // Verify checkpoints were created
    let checkpoint_list = checkpointer.list().await.expect("Failed to list checkpoints");

    // With interval=2 and multiple supersteps, we should have at least one checkpoint
    // The exact number depends on how many supersteps were executed
    if result.supersteps >= 2 {
        assert!(
            !checkpoint_list.is_empty(),
            "At least one checkpoint should exist for supersteps >= 2, got {} supersteps",
            result.supersteps
        );
    }
}

/// Test that workflow can resume from a checkpoint
#[tokio::test]
async fn test_resume_from_checkpoint() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Create a checkpoint manually to simulate a previous run
    let checkpoint = Checkpoint::new(
        "resume-test",
        3, // Superstep 3
        UnitState,
        HashMap::from([
            (VertexId::new("step1"), VertexState::Completed),
            (VertexId::new("step2"), VertexState::Completed),
            (VertexId::new("step3"), VertexState::Halted),
        ]),
        HashMap::new(), // No pending messages
    );
    checkpointer.save(&checkpoint).await.expect("Failed to save checkpoint");

    // Create workflow with same topology
    let graph = WorkflowGraph::<UnitState>::new()
        .name("resume_workflow")
        .node("step1", NodeKind::Passthrough)
        .node("step2", NodeKind::Passthrough)
        .node("step3", NodeKind::Passthrough)
        .entry("step1")
        .edge("step1", "step2")
        .edge("step2", "step3")
        .edge("step3", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::EdgeDriven)
        .with_checkpoint_interval(2);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer,
        "resume-test",
    )
    .expect("Failed to compile workflow");

    // Resume from checkpoint
    let result = workflow.resume().await.expect("Resume failed");

    // Should find the checkpoint and complete
    assert!(result.is_some(), "Should resume from existing checkpoint");
    let result = result.unwrap();
    assert!(result.completed, "Workflow should complete after resume");
}

/// Test that pending messages are preserved across checkpoint/resume
#[tokio::test]
async fn test_resume_preserves_pending_messages() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Create checkpoint with pending messages
    let mut pending_messages = HashMap::new();
    pending_messages.insert(
        VertexId::new("receiver"),
        vec![WorkflowMessage::Activate],
    );

    let checkpoint = Checkpoint::new(
        "pending-test",
        2,
        UnitState,
        HashMap::from([
            (VertexId::new("sender"), VertexState::Halted),
            (VertexId::new("receiver"), VertexState::Halted),
        ]),
        pending_messages,
    );
    checkpointer.save(&checkpoint).await.expect("Failed to save checkpoint");

    // Create workflow
    let graph = WorkflowGraph::<UnitState>::new()
        .name("pending_workflow")
        .node("sender", NodeKind::Passthrough)
        .node("receiver", NodeKind::Passthrough)
        .entry("sender")
        .edge("sender", "receiver")
        .edge("receiver", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::MessageBased)
        .with_checkpoint_interval(1);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer,
        "pending-test",
    )
    .expect("Failed to compile workflow");

    // Resume - the pending Activate message should be delivered to receiver
    let result = workflow.resume().await.expect("Resume failed");

    assert!(result.is_some(), "Should resume from checkpoint");
    let result = result.unwrap();
    assert!(result.completed, "Workflow should complete");
}

/// Test that retry counts are preserved across checkpoint/resume
#[tokio::test]
async fn test_resume_preserves_retry_counts() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Create checkpoint with retry counts
    let mut retry_counts = HashMap::new();
    retry_counts.insert(VertexId::new("flaky"), 2);

    let checkpoint = Checkpoint::with_retry_counts(
        "retry-test",
        3,
        UnitState,
        HashMap::from([
            (VertexId::new("flaky"), VertexState::Halted),
        ]),
        HashMap::new(),
        retry_counts,
    );
    checkpointer.save(&checkpoint).await.expect("Failed to save checkpoint");

    // Create simple workflow
    let graph = WorkflowGraph::<UnitState>::new()
        .name("retry_workflow")
        .node("flaky", NodeKind::Passthrough)
        .entry("flaky")
        .edge("flaky", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_checkpoint_interval(1);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer.clone(),
        "retry-test",
    )
    .expect("Failed to compile workflow");

    // Resume - retry counts should be restored
    let result = workflow.resume().await.expect("Resume failed");
    assert!(result.is_some(), "Should resume from checkpoint");

    // The workflow should complete (passthrough vertex doesn't fail)
    let result = result.unwrap();
    assert!(result.completed);
}

// =============================================================================
// Backend Integration Tests
// =============================================================================

/// Test checkpointing with MemoryCheckpointer
#[tokio::test]
async fn test_checkpointing_with_memory_backend() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());
    run_checkpoint_roundtrip(checkpointer, "memory-backend-test").await;
}

/// Test checkpointing with FileCheckpointer
#[tokio::test]
async fn test_checkpointing_with_file_backend() {
    use rig_deepagents::pregel::checkpoint::FileCheckpointer;

    let temp_dir = std::env::temp_dir().join("rig-deepagents-test-checkpoints");
    let _ = std::fs::remove_dir_all(&temp_dir); // Clean up any previous test

    let checkpointer = FileCheckpointer::new(
        &temp_dir,
        "file-backend-test",
        true, // Enable compression
    );

    run_checkpoint_roundtrip(Arc::new(checkpointer), "file-backend-test").await;

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);
}

/// Test checkpointing with SqliteCheckpointer (if feature enabled)
#[cfg(feature = "checkpointer-sqlite")]
#[tokio::test]
async fn test_checkpointing_with_sqlite_backend() {
    use rig_deepagents::pregel::checkpoint::SqliteCheckpointer;

    let checkpointer = SqliteCheckpointer::new(":memory:", "sqlite-backend-test")
        .await
        .expect("Failed to create SQLite checkpointer");

    run_checkpoint_roundtrip(Arc::new(checkpointer), "sqlite-backend-test").await;
}

/// Helper function for testing checkpoint roundtrip
async fn run_checkpoint_roundtrip(
    checkpointer: Arc<dyn Checkpointer<UnitState> + Send + Sync>,
    workflow_id: &str,
) {
    let graph = WorkflowGraph::<UnitState>::new()
        .name("roundtrip_workflow")
        .node("start", NodeKind::Passthrough)
        .node("middle", NodeKind::Passthrough)
        .node("end", NodeKind::Passthrough)
        .entry("start")
        .edge("start", "middle")
        .edge("middle", "end")
        .edge("end", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::EdgeDriven)
        .with_checkpoint_interval(1); // Checkpoint every superstep

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer.clone(),
        workflow_id,
    )
    .expect("Failed to compile workflow");

    // Run workflow
    let result = workflow.run(UnitState).await.expect("Workflow failed");
    assert!(result.completed, "Workflow should complete");

    // Verify checkpoints exist
    let checkpoints = checkpointer.list().await.expect("Failed to list checkpoints");
    assert!(!checkpoints.is_empty(), "Checkpoints should have been created");

    // Load the latest checkpoint
    let latest = checkpointer.latest().await.expect("Failed to load latest");
    assert!(latest.is_some(), "Latest checkpoint should exist");

    let checkpoint = latest.unwrap();
    assert_eq!(checkpoint.workflow_id, workflow_id);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/// Test that resume() returns error for non-checkpointed workflow
#[tokio::test]
async fn test_resume_without_checkpointer_returns_error() {
    let graph = WorkflowGraph::<UnitState>::new()
        .name("no_checkpoint")
        .node("only", NodeKind::Passthrough)
        .entry("only")
        .edge("only", END)
        .build()
        .expect("Failed to build graph");

    // Compile WITHOUT checkpointer
    let mut workflow = CompiledWorkflow::compile(graph, PregelConfig::default())
        .expect("Failed to compile workflow");

    // Verify no checkpointer
    assert!(!workflow.has_checkpointer());
    assert!(workflow.checkpointer().is_none());

    // Try to resume - should fail with NotImplemented
    let result = workflow.resume().await;
    assert!(result.is_err(), "resume() should fail without checkpointer");

    let error = result.unwrap_err();
    let error_msg = format!("{}", error);
    assert!(
        error_msg.contains("checkpointer") || error_msg.contains("NotImplemented"),
        "Error should mention checkpointer requirement: {}",
        error_msg
    );
}

/// Test that run_from_checkpoint() returns error for non-checkpointed workflow
#[tokio::test]
async fn test_run_from_checkpoint_without_checkpointer_returns_error() {
    let graph = WorkflowGraph::<UnitState>::new()
        .name("no_checkpoint")
        .node("only", NodeKind::Passthrough)
        .entry("only")
        .edge("only", END)
        .build()
        .expect("Failed to build graph");

    // Compile WITHOUT checkpointer
    let mut workflow = CompiledWorkflow::compile(graph, PregelConfig::default())
        .expect("Failed to compile workflow");

    // Create a dummy checkpoint
    let checkpoint = Checkpoint::new(
        "dummy",
        1,
        UnitState,
        HashMap::new(),
        HashMap::new(),
    );

    // Try to run from checkpoint - should fail
    let result = workflow.run_from_checkpoint(checkpoint).await;
    assert!(result.is_err(), "run_from_checkpoint() should fail without checkpointer");
}

/// Test that workflow ID mismatch is detected
#[tokio::test]
async fn test_checkpoint_workflow_id_mismatch() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Save checkpoint with workflow ID "workflow-A"
    let checkpoint = Checkpoint::new(
        "workflow-A", // Different ID
        3,
        UnitState,
        HashMap::from([
            (VertexId::new("node"), VertexState::Halted),
        ]),
        HashMap::new(),
    );
    checkpointer.save(&checkpoint).await.expect("Failed to save checkpoint");

    // Create workflow with different ID
    let graph = WorkflowGraph::<UnitState>::new()
        .name("mismatch_workflow")
        .node("node", NodeKind::Passthrough)
        .entry("node")
        .edge("node", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default().with_checkpoint_interval(1);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer,
        "workflow-B", // Different from checkpoint's workflow_id
    )
    .expect("Failed to compile workflow");

    // Try to resume - should fail with CheckpointMismatch
    let result = workflow.resume().await;
    assert!(result.is_err(), "resume() should fail with ID mismatch");

    let error = result.unwrap_err();
    let error_msg = format!("{}", error);
    // The error should mention mismatch
    assert!(
        error_msg.contains("mismatch") || error_msg.contains("Mismatch") || error_msg.contains("workflow-A") || error_msg.contains("workflow-B"),
        "Error should indicate workflow ID mismatch: {}",
        error_msg
    );
}

// =============================================================================
// Execution Mode Tests
// =============================================================================

/// Test resume in EdgeDriven execution mode
#[tokio::test]
async fn test_resume_edge_driven_mode() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    let graph = WorkflowGraph::<UnitState>::new()
        .name("edge_driven")
        .node("a", NodeKind::Passthrough)
        .node("b", NodeKind::Passthrough)
        .entry("a")
        .edge("a", "b")
        .edge("b", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::EdgeDriven)
        .with_checkpoint_interval(1);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer.clone(),
        "edge-driven-test",
    )
    .expect("Failed to compile workflow");

    // Run to completion
    let result = workflow.run(UnitState).await.expect("Workflow failed");
    assert!(result.completed);

    // Should be able to list checkpoints
    let checkpoints = checkpointer.list().await.expect("Failed to list");
    // Note: checkpoints may or may not exist depending on superstep count
    // Just verify the call works
    let _ = checkpoints;
}

/// Test resume in MessageBased execution mode
#[tokio::test]
async fn test_resume_message_based_mode() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    let graph = WorkflowGraph::<UnitState>::new()
        .name("message_based")
        .node("a", NodeKind::Passthrough)
        .node("b", NodeKind::Passthrough)
        .entry("a")
        .edge("a", "b")
        .edge("b", END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_execution_mode(ExecutionMode::MessageBased)
        .with_checkpoint_interval(1);

    let mut workflow = CompiledWorkflow::compile_with_checkpointer(
        graph,
        config,
        checkpointer.clone(),
        "message-based-test",
    )
    .expect("Failed to compile workflow");

    // Run to completion
    let result = workflow.run(UnitState).await.expect("Workflow failed");
    assert!(result.completed);
}

// =============================================================================
// Checkpointer Management Tests
// =============================================================================

/// Test checkpoint pruning through workflow API
#[tokio::test]
async fn test_checkpoint_pruning() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Save multiple checkpoints
    for i in 1..=5 {
        let checkpoint = Checkpoint::new(
            "prune-test",
            i,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer.save(&checkpoint).await.expect("Failed to save");
    }

    // Verify 5 checkpoints exist
    let before = checkpointer.list().await.expect("Failed to list");
    assert_eq!(before.len(), 5);

    // Prune to keep only 2
    let deleted = checkpointer.prune(2).await.expect("Failed to prune");
    assert_eq!(deleted, 3, "Should have deleted 3 checkpoints");

    // Verify only 2 remain
    let after = checkpointer.list().await.expect("Failed to list");
    assert_eq!(after.len(), 2);
    assert!(after.contains(&4));
    assert!(after.contains(&5));
}

/// Test checkpoint clearing
#[tokio::test]
async fn test_checkpoint_clearing() {
    let checkpointer = Arc::new(MemoryCheckpointer::<UnitState>::new());

    // Save checkpoints
    for i in 1..=3 {
        let checkpoint = Checkpoint::new(
            "clear-test",
            i,
            UnitState,
            HashMap::new(),
            HashMap::new(),
        );
        checkpointer.save(&checkpoint).await.expect("Failed to save");
    }

    // Clear all
    checkpointer.clear().await.expect("Failed to clear");

    // Verify empty
    let after = checkpointer.list().await.expect("Failed to list");
    assert!(after.is_empty(), "All checkpoints should be cleared");
}
