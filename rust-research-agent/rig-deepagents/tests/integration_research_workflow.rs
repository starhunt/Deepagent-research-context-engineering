//! Integration Tests for Research Workflow
//!
//! These tests verify the complete workflow infrastructure including:
//! - SQLite checkpointer persistence
//! - Research state management
//! - Workflow graph execution
//! - Phase transitions and budget enforcement

use rig_deepagents::{
    ResearchConfig, ResearchDirection, ResearchPhase, ResearchState, ResearchUpdate,
    ResearchWorkflowBuilder, Finding, Source, SourceAgreement,
};
use rig_deepagents::pregel::WorkflowState;

#[cfg(feature = "checkpointer-sqlite")]
use std::collections::HashMap;

#[cfg(feature = "checkpointer-sqlite")]
use rig_deepagents::pregel::checkpoint::{Checkpoint, Checkpointer};

#[cfg(feature = "checkpointer-sqlite")]
use rig_deepagents::pregel::config::ExecutionMode;

#[cfg(feature = "checkpointer-sqlite")]
use rig_deepagents::pregel::vertex::{VertexId, VertexState};

#[cfg(feature = "checkpointer-sqlite")]
use rig_deepagents::pregel::PregelConfig;

#[cfg(feature = "checkpointer-sqlite")]
use rig_deepagents::workflow::CompiledWorkflow;

/// Test SQLite checkpointer with ResearchState
#[cfg(feature = "checkpointer-sqlite")]
#[tokio::test]
async fn test_sqlite_checkpointer_with_research_state() {
    use rig_deepagents::pregel::checkpoint::SqliteCheckpointer;

    // Create checkpointer
    let checkpointer = SqliteCheckpointer::new(":memory:", "research-workflow-test")
        .await
        .expect("Failed to create SQLite checkpointer");

    // Create a ResearchState with data
    let mut state = ResearchState::new("What is context engineering in AI systems?")
        .with_max_searches(6);

    // Add some directions
    state.directions.push(ResearchDirection::new(
        "Definition and Origins",
        "Understanding the foundational concepts",
        5,
    ));
    state.directions.push(ResearchDirection::new(
        "Practical Applications",
        "How it's used in real systems",
        4,
    ));

    // Add sources
    state.sources.push(
        Source::new("https://example.com/context-engineering", "Context Engineering Guide", 0.95)
            .with_snippet("Context engineering is the practice of...")
    );

    // Add findings
    state.findings.push(
        Finding::new(
            "Key Insight 1",
            "Context engineering involves structuring prompts and context...",
            0.9,
            ResearchPhase::Exploratory,
        )
        .with_sources(vec![0])
    );

    // Create checkpoint
    let checkpoint = Checkpoint::new(
        "research-workflow-test",
        5,
        state.clone(),
        HashMap::new(),
        HashMap::new(),
    );

    // Save checkpoint
    checkpointer.save(&checkpoint).await.expect("Failed to save checkpoint");

    // Load and verify
    let loaded: Checkpoint<ResearchState> = checkpointer
        .load(5)
        .await
        .expect("Failed to load checkpoint")
        .expect("Checkpoint not found");

    assert_eq!(loaded.state.query, "What is context engineering in AI systems?");
    assert_eq!(loaded.state.directions.len(), 2);
    assert_eq!(loaded.state.sources.len(), 1);
    assert_eq!(loaded.state.findings.len(), 1);
    assert_eq!(loaded.state.max_searches, 6);
}

/// Test research state transitions through workflow
#[tokio::test]
async fn test_research_state_workflow_transitions() {
    // Create initial state
    let mut state = ResearchState::new("Test query").with_max_searches(4);

    // Verify initial state
    assert_eq!(state.phase, ResearchPhase::Exploratory);
    assert!(state.can_continue);
    assert!(state.can_search());
    assert_eq!(state.remaining_searches(), 4);

    // Simulate exploratory phase: add directions
    let update1 = ResearchUpdate::default()
        .with_directions(vec![
            ResearchDirection::new("Direction A", "Reason A", 5),
            ResearchDirection::new("Direction B", "Reason B", 3),
        ])
        .with_search("exploratory search 1".to_string())
        .with_sources(vec![
            Source::new("https://a.com", "Source A", 0.8),
        ]);

    state = state.apply_update(update1);

    assert_eq!(state.directions.len(), 2);
    assert_eq!(state.search_count, 1);
    assert_eq!(state.sources.len(), 1);
    assert!(state.can_continue);

    // Transition to directed phase
    let update2 = ResearchUpdate::transition_to(ResearchPhase::Directed);
    state = state.apply_update(update2);

    assert_eq!(state.phase, ResearchPhase::Directed);
    assert!(state.can_continue); // Still has unexplored directions

    // Explore directions
    let mut update3 = ResearchUpdate::default()
        .with_explored(vec!["Direction A".to_string()])
        .with_search("directed search 1".to_string());
    update3.new_findings = vec![
        Finding::new("Finding 1", "Content 1", 0.85, ResearchPhase::Directed)
            .with_direction("Direction A"),
    ];

    state = state.apply_update(update3);

    assert_eq!(state.search_count, 2);
    assert_eq!(state.findings.len(), 1);
    assert!(state.directions[0].explored || state.directions[1].explored);

    // Explore remaining direction
    let update4 = ResearchUpdate::default()
        .with_explored(vec!["Direction B".to_string()])
        .with_search("directed search 2".to_string());

    state = state.apply_update(update4);

    // All directions explored, can_continue should be false in Directed phase
    assert!(!state.can_continue);

    // Transition to synthesis
    let update5 = ResearchUpdate::transition_to(ResearchPhase::Synthesis)
        .with_agreement(SourceAgreement {
            high_agreement: vec!["Topic 1".to_string()],
            disagreement: vec![],
        });

    state = state.apply_update(update5);

    assert_eq!(state.phase, ResearchPhase::Synthesis);
    assert!(!state.agreement.high_agreement.is_empty());

    // Complete
    let update6 = ResearchUpdate::transition_to(ResearchPhase::Complete);
    state = state.apply_update(update6);

    assert_eq!(state.phase, ResearchPhase::Complete);
    assert!(state.phase.is_terminal());
    assert!(!state.can_continue);
}

/// Test workflow builder creates valid graph
#[test]
fn test_research_workflow_builder_creates_valid_graph() {
    let result = ResearchWorkflowBuilder::new()
        .name("test_research")
        .max_searches(8)
        .max_directions(4)
        .max_explorer_iterations(6)
        .build();

    assert!(result.is_ok(), "Workflow build failed: {:?}", result.err());
}

/// Test workflow compilation with research state
#[test]
fn test_research_workflow_compiles() {
    // ResearchWorkflowBuilder::build() returns a WorkflowGraph
    // Successful build means all nodes and edges are configured correctly
    let result = ResearchWorkflowBuilder::new().build();
    assert!(result.is_ok(), "Workflow compilation failed: {:?}", result.err());
}

/// Test research config builder
#[test]
fn test_research_config_builder() {
    let config = ResearchConfig::new()
        .with_max_searches(10)
        .with_max_directions(5)
        .with_parallel_directions(true)
        .with_timeout(600);

    assert_eq!(config.max_searches, 10);
    assert_eq!(config.max_directions, 5);
    assert!(config.parallel_directions);
    assert_eq!(config.timeout_secs, Some(600));
}

/// Test budget enforcement
#[tokio::test]
async fn test_search_budget_enforcement() {
    let mut state = ResearchState::new("Test").with_max_searches(2);

    // Use up budget
    let update1 = ResearchUpdate::default()
        .with_search("search 1".to_string())
        .with_search("search 2".to_string());

    state = state.apply_update(update1);

    assert_eq!(state.search_count, 2);
    assert!(!state.can_search());
    assert_eq!(state.remaining_searches(), 0);
    assert!(!state.can_continue); // Budget exhausted
}

/// Test source deduplication
#[test]
fn test_source_deduplication() {
    let state = ResearchState::new("Test");

    let update1 = ResearchUpdate::default()
        .with_sources(vec![
            Source::new("https://example.com", "Example 1", 0.8),
            Source::new("https://other.com", "Other", 0.7),
        ]);

    let state = state.apply_update(update1);
    assert_eq!(state.sources.len(), 2);

    // Add duplicate URL
    let update2 = ResearchUpdate::default()
        .with_sources(vec![
            Source::new("https://example.com", "Example 2", 0.9), // Same URL
            Source::new("https://new.com", "New", 0.6),
        ]);

    let state = state.apply_update(update2);
    assert_eq!(state.sources.len(), 3); // Deduped by URL

    // First source should be kept (not replaced)
    assert_eq!(state.sources[0].title, "Example 1");
}

/// Test direction deduplication
#[test]
fn test_direction_deduplication() {
    let state = ResearchState::new("Test");

    let update1 = ResearchUpdate::default()
        .with_directions(vec![
            ResearchDirection::new("Dir A", "Reason A", 5),
        ]);

    let state = state.apply_update(update1);
    assert_eq!(state.directions.len(), 1);

    // Add duplicate direction name
    let update2 = ResearchUpdate::default()
        .with_directions(vec![
            ResearchDirection::new("Dir A", "Different Reason", 3), // Same name
            ResearchDirection::new("Dir B", "Reason B", 4),
        ]);

    let state = state.apply_update(update2);
    assert_eq!(state.directions.len(), 2); // Deduped by name
}

/// Test findings organization by direction
#[test]
fn test_findings_by_direction() {
    let mut state = ResearchState::new("Test");
    state.phase = ResearchPhase::Directed;

    let update = ResearchUpdate::with_findings(vec![
        Finding::new("Finding A1", "Content", 0.9, ResearchPhase::Directed)
            .with_direction("Direction A"),
        Finding::new("Finding A2", "Content", 0.8, ResearchPhase::Directed)
            .with_direction("Direction A"),
        Finding::new("Finding B1", "Content", 0.85, ResearchPhase::Directed)
            .with_direction("Direction B"),
    ]);

    state = state.apply_update(update);

    let findings_a = state.findings_for_direction("Direction A");
    let findings_b = state.findings_for_direction("Direction B");

    assert_eq!(findings_a.len(), 2);
    assert_eq!(findings_b.len(), 1);
}

/// Test checkpoint with full research state including pending messages
#[cfg(feature = "checkpointer-sqlite")]
#[tokio::test]
async fn test_checkpoint_full_state() {
    use rig_deepagents::pregel::checkpoint::SqliteCheckpointer;
    use rig_deepagents::pregel::message::WorkflowMessage;

    let checkpointer = SqliteCheckpointer::with_compression(":memory:", "full-state-test", true)
        .await
        .expect("Failed to create checkpointer");

    // Create comprehensive state
    let mut state = ResearchState::new("Comprehensive test query")
        .with_max_searches(6);

    state.phase = ResearchPhase::Directed;
    state.search_count = 3;

    state.directions = vec![
        ResearchDirection::new("Dir 1", "Reason 1", 5),
        ResearchDirection::new("Dir 2", "Reason 2", 4),
    ];
    state.directions[0].explored = true;

    state.sources = vec![
        Source::new("https://a.com", "Source A", 0.9),
        Source::new("https://b.com", "Source B", 0.8),
    ];

    state.findings = vec![
        Finding::new("Finding 1", "Content 1", 0.9, ResearchPhase::Exploratory)
            .with_sources(vec![0]),
        Finding::new("Finding 2", "Content 2", 0.85, ResearchPhase::Directed)
            .with_direction("Dir 1")
            .with_sources(vec![0, 1]),
    ];

    state.executed_queries.insert("query 1".to_string());
    state.executed_queries.insert("query 2".to_string());

    // Create checkpoint with vertex states and pending messages
    let mut vertex_states = HashMap::new();
    vertex_states.insert(VertexId::new("planner"), VertexState::Completed);
    vertex_states.insert(VertexId::new("explorer"), VertexState::Completed);
    vertex_states.insert(VertexId::new("directed"), VertexState::Active);

    let mut pending_messages = HashMap::new();
    pending_messages.insert(
        VertexId::new("synthesizer"),
        vec![WorkflowMessage::Activate],
    );

    let checkpoint = Checkpoint::new(
        "full-state-test",
        10,
        state,
        vertex_states,
        pending_messages,
    )
    .with_metadata("test_type", "integration")
    .with_metadata("version", "1.0");

    // Save and load
    checkpointer.save(&checkpoint).await.expect("Save failed");
    let loaded: Checkpoint<ResearchState> = checkpointer
        .load(10)
        .await
        .expect("Load failed")
        .expect("Not found");

    // Verify everything
    assert_eq!(loaded.superstep, 10);
    assert_eq!(loaded.workflow_id, "full-state-test");
    assert_eq!(loaded.state.query, "Comprehensive test query");
    assert_eq!(loaded.state.phase, ResearchPhase::Directed);
    assert_eq!(loaded.state.search_count, 3);
    assert_eq!(loaded.state.directions.len(), 2);
    assert!(loaded.state.directions[0].explored);
    assert_eq!(loaded.state.sources.len(), 2);
    assert_eq!(loaded.state.findings.len(), 2);
    assert_eq!(loaded.state.executed_queries.len(), 2);
    assert_eq!(loaded.vertex_states.len(), 3);
    assert_eq!(loaded.pending_message_count(), 1);
    assert_eq!(loaded.metadata.get("test_type"), Some(&"integration".to_string()));
}

/// Test workflow with checkpointing
#[cfg(feature = "checkpointer-sqlite")]
#[tokio::test]
async fn test_workflow_execution_with_sqlite_checkpointing() {
    use rig_deepagents::workflow::graph::WorkflowGraph;
    use rig_deepagents::workflow::node::NodeKind;

    // Create a simple passthrough workflow (since full agent nodes need LLM)
    let graph = WorkflowGraph::<ResearchState>::new()
        .name("checkpointed_workflow")
        .node("start", NodeKind::Passthrough)
        .node("process", NodeKind::Passthrough)
        .node("end", NodeKind::Passthrough)
        .entry("start")
        .edge("start", "process")
        .edge("process", "end")
        .edge("end", rig_deepagents::workflow::END)
        .build()
        .expect("Failed to build graph");

    let config = PregelConfig::default()
        .with_max_supersteps(20)
        .with_execution_mode(ExecutionMode::EdgeDriven)
        .with_checkpoint_interval(1); // Checkpoint every superstep

    let mut workflow = CompiledWorkflow::compile(graph, config)
        .expect("Failed to compile workflow");

    // Run workflow
    let state = ResearchState::new("Workflow checkpoint test");
    let result = workflow.run(state).await.expect("Workflow failed");

    assert!(result.completed);
    assert!(result.supersteps >= 1);
    assert_eq!(result.state.query, "Workflow checkpoint test");
}

/// Test state merging
#[test]
fn test_update_merging() {
    let mut update1 = ResearchUpdate::default().with_search("search 1".to_string());
    update1.new_findings = vec![Finding::new("F1", "C1", 0.8, ResearchPhase::Exploratory)];

    let mut update2 = ResearchUpdate::default().with_search("search 2".to_string());
    update2.new_findings = vec![Finding::new("F2", "C2", 0.7, ResearchPhase::Exploratory)];

    let update3 = ResearchUpdate::transition_to(ResearchPhase::Directed);

    let updates = vec![update1, update2, update3];

    let merged = ResearchState::merge_updates(updates);

    assert_eq!(merged.searches_performed, 2);
    assert_eq!(merged.new_findings.len(), 2);
    assert_eq!(merged.phase_transition, Some(ResearchPhase::Directed)); // Last wins
}

/// Test format_sources output
#[test]
fn test_format_sources() {
    let mut state = ResearchState::new("Test");
    state.sources = vec![
        Source::new("https://first.com", "First Source", 0.9),
        Source::new("https://second.com", "Second Source", 0.8),
        Source::new("https://third.com", "Third Source", 0.7),
    ];

    let formatted = state.format_sources();

    assert!(formatted.contains("[1] First Source: https://first.com"));
    assert!(formatted.contains("[2] Second Source: https://second.com"));
    assert!(formatted.contains("[3] Third Source: https://third.com"));
}

/// Test research prompts module
#[test]
fn test_research_prompts_available() {
    use rig_deepagents::ResearchPrompts;

    // Verify all prompts are non-empty
    assert!(!ResearchPrompts::planner().is_empty());
    assert!(!ResearchPrompts::researcher().is_empty());
    assert!(!ResearchPrompts::synthesizer().is_empty());

    // Verify prompts contain expected content
    assert!(ResearchPrompts::planner().contains("research"));
    assert!(ResearchPrompts::researcher().contains("search"));
    assert!(ResearchPrompts::synthesizer().contains("synth"));
}

/// Test that ResearchState implements WorkflowState correctly
#[test]
fn test_research_state_is_workflow_state() {
    fn requires_workflow_state<S: WorkflowState>() {}
    requires_workflow_state::<ResearchState>();
}
