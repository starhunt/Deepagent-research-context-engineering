//! Pregel Runtime - Core execution engine for workflow graphs
//!
//! The runtime executes workflows through synchronized supersteps.
//! Each superstep follows the sequence: Deliver → Compute → Collect → Route.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::timeout;

use super::checkpoint::{Checkpoint, Checkpointer};
use super::config::{ExecutionMode, PregelConfig};
use super::error::PregelError;
use super::message::{VertexMessage, WorkflowMessage};
use super::state::WorkflowState;
use super::vertex::{BoxedVertex, ComputeContext, ComputeResult, VertexId, VertexState};

/// Metadata for an edge between vertices
#[derive(Debug, Clone, Default)]
pub struct EdgeMetadata {
    /// Optional label for visualization in Mermaid diagrams
    pub label: Option<String>,
}

/// Result of a workflow execution
#[derive(Debug, Clone)]
pub struct WorkflowResult<S: WorkflowState> {
    /// Final workflow state
    pub state: S,
    /// Number of supersteps executed
    pub supersteps: usize,
    /// Whether the workflow completed successfully
    pub completed: bool,
    /// Final states of all vertices
    pub vertex_states: HashMap<VertexId, VertexState>,
}

/// Pregel Runtime for executing workflow graphs
///
/// Manages the execution of vertices through synchronized supersteps,
/// handling message passing, state updates, and termination detection.
pub struct PregelRuntime<S, M>
where
    S: WorkflowState,
    M: VertexMessage,
{
    /// Configuration for the runtime
    config: PregelConfig,
    /// Vertices in the workflow graph
    vertices: HashMap<VertexId, BoxedVertex<S, M>>,
    /// Current state of each vertex
    vertex_states: HashMap<VertexId, VertexState>,
    /// Pending messages for each vertex (delivered at start of next superstep)
    message_queues: HashMap<VertexId, Vec<M>>,
    /// Edges defining message routing (source -> targets with optional metadata)
    edges: HashMap<VertexId, Vec<(VertexId, Option<EdgeMetadata>)>>,
    /// Retry attempt counts per vertex (for retry policy enforcement)
    retry_counts: HashMap<VertexId, usize>,
    /// Entry vertex ID (for EdgeDriven mode reference)
    entry_vertex: Option<VertexId>,
    /// Unique identifier for this workflow instance (used for checkpointing)
    workflow_id: String,
    /// State type marker (used by specialized impl blocks)
    _state_marker: std::marker::PhantomData<S>,
}

impl<S, M> PregelRuntime<S, M>
where
    S: WorkflowState,
    M: VertexMessage,
{
    /// Create a new runtime with default configuration
    pub fn new() -> Self {
        Self::with_config(PregelConfig::default())
    }

    /// Create a new runtime with custom configuration
    pub fn with_config(config: PregelConfig) -> Self {
        Self {
            config,
            vertices: HashMap::new(),
            vertex_states: HashMap::new(),
            message_queues: HashMap::new(),
            edges: HashMap::new(),
            retry_counts: HashMap::new(),
            entry_vertex: None,
            workflow_id: uuid::Uuid::new_v4().to_string(),
            _state_marker: std::marker::PhantomData,
        }
    }

    /// Set the workflow ID for this runtime
    ///
    /// The workflow ID is used for checkpointing to ensure checkpoints
    /// are only restored for the same workflow instance.
    pub fn with_workflow_id(mut self, workflow_id: impl Into<String>) -> Self {
        self.workflow_id = workflow_id.into();
        self
    }

    /// Get the workflow ID
    pub fn workflow_id(&self) -> &str {
        &self.workflow_id
    }

    /// Add a vertex to the runtime
    pub fn add_vertex(&mut self, vertex: BoxedVertex<S, M>) -> &mut Self {
        let id = vertex.id().clone();
        // C1 Fix: Initial state depends on execution mode
        let initial_state = match self.config.execution_mode {
            ExecutionMode::MessageBased => VertexState::Active,
            ExecutionMode::EdgeDriven => VertexState::Halted,
        };
        self.vertex_states.insert(id.clone(), initial_state);
        self.message_queues.insert(id.clone(), Vec::new());
        self.vertices.insert(id, vertex);
        self
    }

    /// Add an edge between vertices (without label)
    pub fn add_edge(&mut self, from: impl Into<VertexId>, to: impl Into<VertexId>) -> &mut Self {
        self.add_edge_with_label(from, to, None)
    }

    /// Add an edge between vertices with an optional label
    ///
    /// Labels are displayed on Mermaid diagrams for visualization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// runtime.add_edge_with_label("router", "success_path", Some("success".into()));
    /// runtime.add_edge_with_label("router", "error_path", Some("error".into()));
    /// ```
    pub fn add_edge_with_label(
        &mut self,
        from: impl Into<VertexId>,
        to: impl Into<VertexId>,
        label: Option<String>,
    ) -> &mut Self {
        let from = from.into();
        let to = to.into();
        let metadata = label.map(|l| EdgeMetadata { label: Some(l) });
        self.edges.entry(from).or_default().push((to, metadata));
        self
    }

    /// Set the entry point (activate this vertex on start)
    pub fn set_entry(&mut self, entry: impl Into<VertexId>) -> &mut Self {
        let entry_id = entry.into();
        // C1 Fix: In EdgeDriven mode, ensure only entry is Active
        if self.config.execution_mode == ExecutionMode::EdgeDriven {
            // First, set all vertices to Halted
            for state in self.vertex_states.values_mut() {
                if state.is_active() {
                    *state = VertexState::Halted;
                }
            }
        }
        // Activate the entry vertex
        if let Some(state) = self.vertex_states.get_mut(&entry_id) {
            *state = VertexState::Active;
        }
        self.entry_vertex = Some(entry_id);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &PregelConfig {
        &self.config
    }

    /// Run the workflow to completion
    ///
    /// Enforces the configured `workflow_timeout` - if the workflow takes longer
    /// than this duration, it will return a `WorkflowTimeout` error.
    pub async fn run(&mut self, initial_state: S) -> Result<WorkflowResult<S>, PregelError> {
        let workflow_timeout = self.config.workflow_timeout;

        // C2 Fix: Wrap entire run loop with workflow timeout
        match timeout(workflow_timeout, self.run_inner(initial_state)).await {
            Ok(result) => result,
            Err(_) => Err(PregelError::WorkflowTimeout(workflow_timeout)),
        }
    }

    /// Internal run loop (extracted for timeout wrapping)
    async fn run_inner(&mut self, initial_state: S) -> Result<WorkflowResult<S>, PregelError> {
        let mut state = initial_state;
        let mut superstep = 0;

        loop {
            // Check max supersteps limit
            if superstep >= self.config.max_supersteps {
                return Err(PregelError::MaxSuperstepsExceeded(superstep));
            }

            // Check if workflow should terminate
            if self.should_terminate(&state) {
                return Ok(WorkflowResult {
                    state,
                    supersteps: superstep,
                    completed: true,
                    vertex_states: self.vertex_states.clone(),
                });
            }

            // Execute one superstep
            let updates = self.execute_superstep(superstep, &state).await?;

            // Apply state updates
            state = state.apply_updates(updates);

            superstep += 1;
        }
    }

    /// Check if the workflow should terminate
    pub(crate) fn should_terminate(&self, state: &S) -> bool {
        // Terminal state check
        if state.is_terminal() {
            return true;
        }

        // All vertices halted or completed AND no pending messages
        let all_inactive = self
            .vertex_states
            .values()
            .all(|s| !s.is_active());

        let no_pending_messages = self
            .message_queues
            .values()
            .all(|q| q.is_empty());

        all_inactive && no_pending_messages
    }

    /// Execute a single superstep
    pub(crate) async fn execute_superstep(
        &mut self,
        superstep: usize,
        state: &S,
    ) -> Result<Vec<S::Update>, PregelError> {
        // 1. Deliver messages - move pending messages to vertex inboxes
        let inboxes = self.deliver_messages();

        // 2. Reactivate halted vertices that received messages
        for (vertex_id, messages) in &inboxes {
            if !messages.is_empty() {
                if let Some(vertex_state) = self.vertex_states.get_mut(vertex_id) {
                    if vertex_state.is_halted() {
                        if let Some(vertex) = self.vertices.get(vertex_id) {
                            *vertex_state = vertex.on_reactivation(messages);
                        }
                    }
                }
            }
        }

        // 3. Compute active vertices in parallel
        let (updates, outboxes, newly_halted) = self.compute_vertices(superstep, state, &inboxes).await?;

        // 4. Route explicit messages from vertex outboxes
        self.route_messages(outboxes);

        // 5. C2 Fix: Route automatic edge messages for newly halted vertices
        self.route_edge_messages(&newly_halted);

        Ok(updates)
    }

    /// Deliver pending messages to vertex inboxes
    fn deliver_messages(&mut self) -> HashMap<VertexId, Vec<M>> {
        let mut inboxes = HashMap::new();
        for (vertex_id, queue) in &mut self.message_queues {
            if !queue.is_empty() {
                inboxes.insert(vertex_id.clone(), std::mem::take(queue));
            } else {
                inboxes.insert(vertex_id.clone(), Vec::new());
            }
        }
        inboxes
    }

    /// Compute all active vertices in parallel
    /// Returns (updates, outboxes, newly_halted_vertex_ids)
    async fn compute_vertices(
        &mut self,
        superstep: usize,
        state: &S,
        inboxes: &HashMap<VertexId, Vec<M>>,
    ) -> Result<(Vec<S::Update>, HashMap<VertexId, HashMap<VertexId, Vec<M>>>, Vec<VertexId>), PregelError> {
        let semaphore = Arc::new(Semaphore::new(self.config.parallelism));
        let updates = Arc::new(Mutex::new(Vec::new()));
        let outboxes = Arc::new(Mutex::new(HashMap::new()));
        let vertex_timeout = self.config.vertex_timeout;

        // Collect active vertices to compute
        let active_vertices: Vec<_> = self
            .vertex_states
            .iter()
            .filter(|(_, state)| state.is_active())
            .map(|(id, _)| id.clone())
            .collect();

        // Execute vertices in parallel
        let mut handles = Vec::new();

        for vertex_id in active_vertices {
            let vertex = match self.vertices.get(&vertex_id) {
                Some(v) => Arc::clone(v),
                None => continue,
            };
            let messages = inboxes.get(&vertex_id).cloned().unwrap_or_default();
            let state_clone = state.clone();
            let sem_clone = Arc::clone(&semaphore);
            let vid = vertex_id.clone();

            let handle = tokio::spawn(async move {
                // Acquire semaphore permit for parallelism control
                let _permit = sem_clone.acquire().await.unwrap();

                // Create compute context
                let mut ctx = ComputeContext::new(vid.clone(), &messages, superstep, &state_clone);

                // Execute with timeout
                let result: Result<ComputeResult<S::Update>, PregelError> = match timeout(
                    vertex_timeout,
                    vertex.compute(&mut ctx),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => Err(PregelError::VertexTimeout(vid.clone())),
                };

                let outbox = ctx.into_outbox();

                (vid, result, outbox)
            });

            handles.push(handle);
        }

        // Collect results
        let mut new_vertex_states = HashMap::new();
        let mut newly_halted = Vec::new();

        for handle in handles {
            let (vid, result, outbox) = handle.await.map_err(|e| {
                PregelError::vertex_error_with_source(
                    "unknown",
                    "task join error",
                    std::io::Error::other(e.to_string()),
                )
            })?;

            match result {
                Ok(compute_result) => {
                    // Success: reset retry count for this vertex
                    self.retry_counts.remove(&vid);
                    updates.lock().await.push(compute_result.update);
                    // C2 Fix: Track newly halted vertices for edge routing
                    if compute_result.state.is_halted() {
                        newly_halted.push(vid.clone());
                    }
                    new_vertex_states.insert(vid.clone(), compute_result.state);
                    outboxes.lock().await.insert(vid, outbox);
                }
                Err(e) => {
                    if e.is_recoverable() {
                        // C3 Fix: Track retry attempts and enforce max_retries
                        // retry_count tracks how many retries we've already attempted
                        let retry_count = self.retry_counts.entry(vid.clone()).or_insert(0);

                        // Check if we can retry BEFORE incrementing
                        if self.config.retry_policy.should_retry(*retry_count) {
                            // Apply backoff delay before next retry
                            let delay = self.config.retry_policy.delay_for_attempt(*retry_count);
                            tokio::time::sleep(delay).await;
                            // Track this retry attempt
                            *retry_count += 1;
                            // Keep vertex active for retry
                            new_vertex_states.insert(vid, VertexState::Active);
                        } else {
                            // Max retries exceeded (current attempt is retry_count + 1 total)
                            return Err(PregelError::MaxRetriesExceeded {
                                vertex_id: vid,
                                attempts: *retry_count + 1, // +1 for the current failed attempt
                            });
                        }
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // Update vertex states
        for (vid, new_state) in new_vertex_states {
            self.vertex_states.insert(vid, new_state);
        }

        // C1 Fix: Use async-safe lock instead of blocking_lock
        let final_updates = match Arc::try_unwrap(updates) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().await.clone(),
        };

        let final_outboxes = match Arc::try_unwrap(outboxes) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().await.clone(),
        };

        Ok((final_updates, final_outboxes, newly_halted))
    }

    /// Route outgoing messages to target vertex queues
    fn route_messages(&mut self, outboxes: HashMap<VertexId, HashMap<VertexId, Vec<M>>>) {
        for (_source, outbox) in outboxes {
            for (target, messages) in outbox {
                if let Some(queue) = self.message_queues.get_mut(&target) {
                    queue.extend(messages);
                }
            }
        }
    }

    /// Route automatic activation messages when vertices halt (EdgeDriven mode only)
    fn route_edge_messages(&mut self, newly_halted: &[VertexId]) {
        if self.config.execution_mode != ExecutionMode::EdgeDriven {
            return;
        }

        for source_id in newly_halted {
            // Get edge targets for this source
            if let Some(targets) = self.edges.get(source_id) {
                for (target_id, _metadata) in targets {
                    // Send Activate message to each edge target
                    if let Some(queue) = self.message_queues.get_mut(target_id) {
                        queue.push(M::activation_message());
                    }
                }
            }
        }
    }

    // =========================================================================
    // Visualization Methods
    // =========================================================================

    /// Generate a static Mermaid diagram of the workflow structure.
    ///
    /// This is useful for debugging - it shows nodes and edges without
    /// execution state. All nodes render as rectangles since NodeKind
    /// information is not stored in the runtime.
    ///
    /// # Example Output
    ///
    /// ```text
    /// graph TD
    ///     start([START])
    ///     agent[agent]
    ///     tool[tool]
    ///     end_node([END])
    ///
    ///     start --> agent
    ///     agent --> tool
    ///     tool --> end_node
    /// ```
    pub fn to_mermaid(&self) -> String {
        self.to_mermaid_internal(false, &std::collections::HashMap::new())
    }

    /// Generate a Mermaid diagram with node kinds for proper shape rendering.
    ///
    /// Use this when you have NodeKind information available (e.g., from
    /// a WorkflowGraph builder).
    pub fn to_mermaid_with_kinds(&self, node_kinds: &HashMap<VertexId, crate::workflow::NodeKind>) -> String {
        self.to_mermaid_internal(false, node_kinds)
    }

    /// Generate a Mermaid diagram with current execution state.
    ///
    /// Vertices are colored based on their state:
    /// - Active (green): Currently executing
    /// - Halted (orange): Waiting for messages
    /// - Completed (gray): Finished processing
    ///
    /// # Example Output
    ///
    /// ```text
    /// graph TD
    ///     start([START]):::completed
    ///     agent[agent]:::active
    ///     tool[tool]:::halted
    ///
    ///     start --> agent
    ///     agent --> tool
    ///
    ///     classDef active fill:#90EE90,stroke:#228B22,stroke-width:2px
    ///     classDef halted fill:#FFE4B5,stroke:#FF8C00,stroke-width:1px
    ///     classDef completed fill:#D3D3D3,stroke:#696969,stroke-width:1px
    /// ```
    pub fn to_mermaid_with_state(&self) -> String {
        self.to_mermaid_internal(true, &std::collections::HashMap::new())
    }

    /// Generate a Mermaid diagram with both state colors and node shapes.
    pub fn to_mermaid_with_state_and_kinds(
        &self,
        node_kinds: &HashMap<VertexId, crate::workflow::NodeKind>,
    ) -> String {
        self.to_mermaid_internal(true, node_kinds)
    }

    /// Internal implementation for Mermaid generation.
    fn to_mermaid_internal(
        &self,
        include_state: bool,
        node_kinds: &HashMap<VertexId, crate::workflow::NodeKind>,
    ) -> String {
        use std::fmt::Write;
        use super::visualization::{render_node, render_node_with_state, render_edge, STYLE_DEFS};
        use crate::workflow::NodeKind;

        let mut output = String::new();

        // Header
        writeln!(output, "graph TD").unwrap();

        // Collect all vertex IDs
        let vertex_ids: Vec<_> = self.vertices.keys().collect();

        // Find entry and terminal vertices for special shapes
        let entry_id = self.entry_vertex.as_ref();
        let terminal_ids: Vec<_> = self.find_terminal_vertices();

        // Render nodes
        for id in &vertex_ids {
            let kind = node_kinds.get(*id);
            let is_entry = entry_id == Some(*id);
            let is_terminal = terminal_ids.contains(id);

            // Use special shape for entry/terminal if no kind specified
            let effective_kind = if kind.is_none() && (is_entry || is_terminal) {
                None // Stadium shape for START/END
            } else if kind.is_none() {
                // Default to Agent shape (rectangle) for regular vertices
                Some(NodeKind::Agent(Default::default()))
            } else {
                kind.cloned()
            };

            let node_str = if include_state {
                let state = self.vertex_states.get(*id);
                render_node_with_state(id, effective_kind.as_ref(), state)
            } else {
                render_node(id, effective_kind.as_ref())
            };

            writeln!(output, "{}", node_str).unwrap();
        }

        // Empty line before edges
        writeln!(output).unwrap();

        // Render edges with labels
        for (from, targets) in &self.edges {
            for (to, metadata) in targets {
                let label = metadata.as_ref().and_then(|m| m.label.as_deref());
                writeln!(output, "{}", render_edge(from, to, label)).unwrap();
            }
        }

        // Style definitions for state visualization
        if include_state {
            output.push_str(STYLE_DEFS);
        }

        output
    }

    /// Find vertices with no outgoing edges (terminal vertices).
    fn find_terminal_vertices(&self) -> Vec<&VertexId> {
        self.vertices
            .keys()
            .filter(|id| {
                match self.edges.get(*id) {
                    None => true,
                    Some(targets) => targets.is_empty(),
                }
            })
            .collect()
    }

    /// Print execution state to terminal for monitoring.
    ///
    /// Shows the state of each vertex with Unicode symbols:
    /// - ▶ Active: Currently executing
    /// - ⏸ Halted: Waiting for messages
    /// - ✓ Completed: Finished processing
    ///
    /// # Example Output
    ///
    /// ```text
    /// [Superstep 0]
    ///   ▶ agent : Active
    ///   ⏸ tool : Halted
    ///   ⏸ router : Halted
    /// ```
    pub fn log_state(&self, superstep: usize) {
        println!("[Superstep {}]", superstep);

        // Sort vertex IDs for consistent output
        let mut vertex_ids: Vec<_> = self.vertex_states.keys().collect();
        vertex_ids.sort();

        for id in vertex_ids {
            if let Some(state) = self.vertex_states.get(id) {
                let symbol = match state {
                    VertexState::Active => "▶",
                    VertexState::Halted => "⏸",
                    VertexState::Completed => "✓",
                };
                println!("  {} {} : {:?}", symbol, id, state);
            }
        }
    }
}

impl<S, M> Default for PregelRuntime<S, M>
where
    S: WorkflowState,
    M: VertexMessage,
{
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Checkpointing Support (WorkflowMessage-specialized)
// =============================================================================

/// Checkpointing-enabled runtime holder
///
/// This type is used internally to store the checkpointer when attached
/// to a runtime. It's specialized for WorkflowMessage because that's what
/// the Checkpoint struct uses for pending_messages.
pub struct CheckpointingRuntime<S>
where
    S: WorkflowState + Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// The underlying runtime
    pub runtime: PregelRuntime<S, WorkflowMessage>,
    /// The checkpointer for state persistence
    checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
}

impl<S> CheckpointingRuntime<S>
where
    S: WorkflowState + Clone + Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Create a new checkpointing runtime
    ///
    /// # Arguments
    ///
    /// * `runtime` - The PregelRuntime to wrap
    /// * `checkpointer` - The checkpointer for state persistence
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use rig_deepagents::pregel::checkpoint::{MemoryCheckpointer, Checkpointer};
    /// use std::sync::Arc;
    ///
    /// let runtime = PregelRuntime::new();
    /// let checkpointer = Arc::new(MemoryCheckpointer::new());
    /// let checkpointing_runtime = CheckpointingRuntime::new(runtime, checkpointer);
    /// ```
    pub fn new(
        runtime: PregelRuntime<S, WorkflowMessage>,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
    ) -> Self {
        Self {
            runtime,
            checkpointer,
        }
    }

    /// Get the workflow ID
    pub fn workflow_id(&self) -> &str {
        &self.runtime.workflow_id
    }

    /// Run the workflow with automatic checkpointing
    ///
    /// Checkpoints are saved at intervals specified by `PregelConfig::checkpoint_interval`.
    pub async fn run(&mut self, initial_state: S) -> Result<WorkflowResult<S>, PregelError> {
        self.run_from_superstep(initial_state, 0).await
    }

    /// Run workflow from a specific superstep
    ///
    /// This is the core method for checkpoint-aware execution.
    /// It starts from the given superstep and saves checkpoints at configured intervals.
    async fn run_from_superstep(
        &mut self,
        initial_state: S,
        start_superstep: usize,
    ) -> Result<WorkflowResult<S>, PregelError> {
        let workflow_timeout = self.runtime.config.workflow_timeout;

        match timeout(workflow_timeout, self.run_inner_from(initial_state, start_superstep)).await {
            Ok(result) => result,
            Err(_) => Err(PregelError::WorkflowTimeout(workflow_timeout)),
        }
    }

    /// Internal run loop with checkpoint support (extracted for timeout wrapping)
    async fn run_inner_from(
        &mut self,
        initial_state: S,
        start_superstep: usize,
    ) -> Result<WorkflowResult<S>, PregelError> {
        let mut state = initial_state;
        let mut superstep = start_superstep;

        loop {
            // Check max supersteps limit (adjusted for resume)
            if superstep >= self.runtime.config.max_supersteps {
                return Err(PregelError::MaxSuperstepsExceeded(superstep));
            }

            // Check if workflow should terminate
            if self.runtime.should_terminate(&state) {
                return Ok(WorkflowResult {
                    state,
                    supersteps: superstep,
                    completed: true,
                    vertex_states: self.runtime.vertex_states.clone(),
                });
            }

            // Execute one superstep
            let updates = self.runtime.execute_superstep(superstep, &state).await?;

            // Apply state updates
            state = state.apply_updates(updates);

            superstep += 1;

            // Save checkpoint if interval reached
            if self.runtime.config.should_checkpoint(superstep) {
                self.save_checkpoint(superstep, &state).await?;
            }
        }
    }

    /// Resume workflow from the latest checkpoint
    ///
    /// Returns `None` if no checkpoint exists, otherwise returns the workflow result.
    ///
    /// # Critical Fix (Codex Review)
    ///
    /// This method properly resumes from a checkpoint by:
    /// 1. Loading the latest checkpoint
    /// 2. Restoring vertex states, message queues, and retry counts
    /// 3. Continuing execution from the checkpoint's superstep (not 0)
    pub async fn resume(&mut self) -> Result<Option<WorkflowResult<S>>, PregelError> {
        if let Some(checkpoint) = self.checkpointer.latest().await? {
            self.restore_from_checkpoint(&checkpoint)?;
            let result = self.run_from_superstep(checkpoint.state, checkpoint.superstep).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Run workflow from a specific checkpoint
    ///
    /// This is the primary method for resuming from a known checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to resume from
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load a specific checkpoint
    /// if let Some(checkpoint) = checkpointer.load(10).await? {
    ///     let result = runtime.run_from_checkpoint(checkpoint).await?;
    /// }
    /// ```
    pub async fn run_from_checkpoint(
        &mut self,
        checkpoint: Checkpoint<S>,
    ) -> Result<WorkflowResult<S>, PregelError> {
        // Restore state from checkpoint
        self.restore_from_checkpoint(&checkpoint)?;

        // Continue from checkpoint superstep
        self.run_from_superstep(checkpoint.state, checkpoint.superstep).await
    }

    /// Restore runtime state from a checkpoint
    ///
    /// # Critical Fixes (Gemini/Qwen Review)
    ///
    /// This method restores:
    /// - Vertex states (Active, Halted, Completed)
    /// - Pending message queues (with proper clear/overwrite to prevent state leak)
    /// - Retry counts (from metadata) - prevents retry count reset on resume
    ///
    /// The method also validates topology compatibility between checkpoint and runtime.
    fn restore_from_checkpoint(&mut self, checkpoint: &Checkpoint<S>) -> Result<(), PregelError> {
        // Validate workflow_id matches
        if checkpoint.workflow_id != self.runtime.workflow_id {
            return Err(PregelError::checkpoint_mismatch(
                &self.runtime.workflow_id,
                &checkpoint.workflow_id,
            ));
        }

        // Validate topology compatibility (Gemini/Qwen review feedback)
        // Check for vertices in checkpoint but not in runtime
        let missing_in_runtime: Vec<_> = checkpoint
            .vertex_states
            .keys()
            .filter(|vid| !self.runtime.vertices.contains_key(*vid))
            .collect();

        if !missing_in_runtime.is_empty() {
            return Err(PregelError::checkpoint_error(format!(
                "Checkpoint contains vertices not present in current runtime: {:?}",
                missing_in_runtime
            )));
        }

        // Check for vertices in runtime but not in checkpoint (warning only)
        let missing_in_checkpoint: Vec<_> = self
            .runtime
            .vertices
            .keys()
            .filter(|vid| !checkpoint.vertex_states.contains_key(*vid))
            .collect();

        if !missing_in_checkpoint.is_empty() {
            tracing::warn!(
                missing_vertices = ?missing_in_checkpoint,
                "Runtime contains vertices not present in checkpoint - they will start with default state"
            );
        }

        // Restore vertex states
        self.runtime.vertex_states = checkpoint.vertex_states.clone();

        // FIX: Properly restore pending messages (Gemini review feedback)
        // Clear/overwrite ALL message queues to prevent stale message leak
        for (vid, queue) in &mut self.runtime.message_queues {
            if let Some(msgs) = checkpoint.pending_messages.get(vid) {
                *queue = msgs.clone();
            } else {
                // Clear queues for vertices not in checkpoint to prevent state leak
                queue.clear();
            }
        }

        // Restore retry counts from struct field (Gemini/Qwen review feedback)
        // Now stored as first-class field instead of metadata JSON
        self.runtime.retry_counts = checkpoint.retry_counts.clone();

        tracing::info!(
            workflow_id = %checkpoint.workflow_id,
            superstep = checkpoint.superstep,
            "Restored from checkpoint"
        );

        Ok(())
    }

    /// Create a checkpoint at the current state
    ///
    /// # Design Decision (Gemini/Qwen Review)
    ///
    /// Uses `Checkpoint::with_retry_counts()` to store retry state as a first-class
    /// struct field rather than in metadata JSON. This provides:
    /// - Type safety (compile-time checking)
    /// - No double-serialization overhead
    /// - Clearer checkpoint schema
    fn create_checkpoint(&self, superstep: usize, state: &S) -> Checkpoint<S> {
        let pending_messages: HashMap<VertexId, Vec<WorkflowMessage>> = self
            .runtime
            .message_queues
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Checkpoint::with_retry_counts(
            &self.runtime.workflow_id,
            superstep,
            state.clone(),
            self.runtime.vertex_states.clone(),
            pending_messages,
            self.runtime.retry_counts.clone(),
        )
    }

    /// Save a checkpoint
    async fn save_checkpoint(&self, superstep: usize, state: &S) -> Result<(), PregelError> {
        let checkpoint = self.create_checkpoint(superstep, state);
        self.checkpointer.save(&checkpoint).await?;
        tracing::info!(
            workflow_id = %self.runtime.workflow_id,
            superstep,
            "Checkpoint saved"
        );
        Ok(())
    }

    /// Get access to the underlying checkpointer
    pub fn checkpointer(&self) -> &Arc<dyn Checkpointer<S> + Send + Sync> {
        &self.checkpointer
    }

    /// Access the underlying runtime (immutable)
    pub fn inner(&self) -> &PregelRuntime<S, WorkflowMessage> {
        &self.runtime
    }

    /// Access the underlying runtime (mutable)
    pub fn inner_mut(&mut self) -> &mut PregelRuntime<S, WorkflowMessage> {
        &mut self.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::message::WorkflowMessage;
    use super::super::vertex::{StateUpdate, Vertex};
    use async_trait::async_trait;
    use tokio::time::Duration;

    #[allow(unused_imports)]
    use super::super::state::WorkflowState as _;

    // Test state
    #[derive(Clone, Default, Debug)]
    struct TestState {
        counter: i32,
        messages_received: i32,
    }

    #[derive(Clone, Debug)]
    struct TestUpdate {
        counter_delta: i32,
        messages_delta: i32,
    }

    impl StateUpdate for TestUpdate {
        fn empty() -> Self {
            TestUpdate {
                counter_delta: 0,
                messages_delta: 0,
            }
        }

        fn is_empty(&self) -> bool {
            self.counter_delta == 0 && self.messages_delta == 0
        }
    }

    impl WorkflowState for TestState {
        type Update = TestUpdate;

        fn apply_update(&self, update: Self::Update) -> Self {
            TestState {
                counter: self.counter + update.counter_delta,
                messages_received: self.messages_received + update.messages_delta,
            }
        }

        fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
            TestUpdate {
                counter_delta: updates.iter().map(|u| u.counter_delta).sum(),
                messages_delta: updates.iter().map(|u| u.messages_delta).sum(),
            }
        }

        fn is_terminal(&self) -> bool {
            self.counter >= 10
        }
    }

    // Simple vertex that increments counter and halts
    struct IncrementVertex {
        id: VertexId,
        #[allow(dead_code)]
        increment: i32,
    }

    #[async_trait]
    impl Vertex<TestState, WorkflowMessage> for IncrementVertex {
        fn id(&self) -> &VertexId {
            &self.id
        }

        async fn compute(
            &self,
            _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
        ) -> Result<ComputeResult<TestUpdate>, PregelError> {
            // Just halt immediately
            Ok(ComputeResult::halt(TestUpdate::empty()))
        }
    }

    // Vertex that sends a message then halts
    struct MessageSenderVertex {
        id: VertexId,
        target: VertexId,
    }

    #[async_trait]
    impl Vertex<TestState, WorkflowMessage> for MessageSenderVertex {
        fn id(&self) -> &VertexId {
            &self.id
        }

        async fn compute(
            &self,
            ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
        ) -> Result<ComputeResult<TestUpdate>, PregelError> {
            if ctx.is_first_superstep() {
                ctx.send_message(self.target.clone(), WorkflowMessage::Activate);
            }
            Ok(ComputeResult::halt(TestUpdate::empty()))
        }
    }

    // Vertex that counts messages received
    struct MessageReceiverVertex {
        id: VertexId,
    }

    #[async_trait]
    impl Vertex<TestState, WorkflowMessage> for MessageReceiverVertex {
        fn id(&self) -> &VertexId {
            &self.id
        }

        async fn compute(
            &self,
            _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
        ) -> Result<ComputeResult<TestUpdate>, PregelError> {
            // Just halt after receiving messages
            Ok(ComputeResult::halt(TestUpdate::empty()))
        }
    }

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();
        assert_eq!(runtime.config().max_supersteps, 100);
    }

    #[tokio::test]
    async fn test_runtime_single_vertex_halts() {
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();

        runtime.add_vertex(Arc::new(IncrementVertex {
            id: VertexId::new("a"),
            increment: 1,
        }));

        let result = runtime.run(TestState::default()).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.completed);
        // Single vertex computes once then halts, workflow terminates
        assert!(result.supersteps <= 2);
    }

    #[tokio::test]
    async fn test_runtime_message_delivery() {
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();

        runtime.add_vertex(Arc::new(MessageSenderVertex {
            id: VertexId::new("sender"),
            target: VertexId::new("receiver"),
        }));

        runtime.add_vertex(Arc::new(MessageReceiverVertex {
            id: VertexId::new("receiver"),
        }));

        let result = runtime.run(TestState::default()).await.unwrap();
        assert!(result.completed);
        // Sender sends in superstep 0, receiver gets it in superstep 1
        assert!(result.supersteps >= 1);
    }

    #[tokio::test]
    async fn test_runtime_termination_all_halted() {
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();

        // Add two vertices that halt immediately
        runtime.add_vertex(Arc::new(IncrementVertex {
            id: VertexId::new("a"),
            increment: 1,
        }));

        runtime.add_vertex(Arc::new(IncrementVertex {
            id: VertexId::new("b"),
            increment: 1,
        }));

        let result = runtime.run(TestState::default()).await.unwrap();
        assert!(result.completed);
        // All vertices halt, no messages pending -> terminate
        assert!(result.vertex_states.values().all(|s| !s.is_active()));
    }

    #[tokio::test]
    async fn test_runtime_max_supersteps_exceeded() {
        struct InfiniteLoopVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for InfiniteLoopVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                // Always stay active
                ctx.send_message(self.id.clone(), WorkflowMessage::Activate);
                Ok(ComputeResult::active(TestUpdate::empty()))
            }
        }

        let config = PregelConfig::default().with_max_supersteps(5);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime.add_vertex(Arc::new(InfiniteLoopVertex {
            id: VertexId::new("loop"),
        }));

        let result = runtime.run(TestState::default()).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PregelError::MaxSuperstepsExceeded(5)
        ));
    }

    #[tokio::test]
    async fn test_runtime_terminal_state() {
        struct CounterVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for CounterVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                // Keep running until terminal state
                ctx.send_message(self.id.clone(), WorkflowMessage::Activate);
                Ok(ComputeResult::active(TestUpdate::empty()))
            }
        }

        let mut runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();

        runtime.add_vertex(Arc::new(CounterVertex {
            id: VertexId::new("counter"),
        }));

        // Start with counter at 10, which is terminal
        let result = runtime
            .run(TestState {
                counter: 10,
                messages_received: 0,
            })
            .await
            .unwrap();

        assert!(result.completed);
        assert_eq!(result.supersteps, 0); // Terminates immediately
    }

    #[tokio::test]
    async fn test_runtime_parallel_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;

        static EXECUTION_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct SlowVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for SlowVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                EXECUTION_COUNT.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(ComputeResult::halt(TestUpdate::empty()))
            }
        }

        EXECUTION_COUNT.store(0, Ordering::SeqCst);

        let config = PregelConfig::default().with_parallelism(4);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        // Add 4 slow vertices
        for i in 0..4 {
            runtime.add_vertex(Arc::new(SlowVertex {
                id: VertexId::new(format!("slow_{}", i)),
            }));
        }

        let start = Instant::now();
        let result = runtime.run(TestState::default()).await.unwrap();
        let elapsed = start.elapsed();

        assert!(result.completed);
        assert_eq!(EXECUTION_COUNT.load(Ordering::SeqCst), 4);
        // With parallelism=4, should take ~50ms, not ~200ms
        assert!(elapsed < Duration::from_millis(150));
    }

    #[tokio::test]
    async fn test_runtime_add_edge() {
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> = PregelRuntime::new();

        runtime
            .add_vertex(Arc::new(IncrementVertex {
                id: VertexId::new("a"),
                increment: 1,
            }))
            .add_edge("a", "b");

        assert!(runtime.edges.contains_key(&VertexId::new("a")));
    }

    // ============================================
    // C2: Workflow Timeout Tests (RED - should fail)
    // ============================================

    #[tokio::test]
    async fn test_workflow_timeout_enforced() {
        // Vertex that runs forever (simulates slow LLM calls)
        struct SlowForeverVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for SlowForeverVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                // Sleep for a long time but stay active
                tokio::time::sleep(Duration::from_secs(10)).await;
                ctx.send_message(self.id.clone(), WorkflowMessage::Activate);
                Ok(ComputeResult::active(TestUpdate::empty()))
            }
        }

        // Set a very short workflow timeout (100ms)
        let config = PregelConfig::default()
            .with_workflow_timeout(Duration::from_millis(100))
            .with_vertex_timeout(Duration::from_secs(60)) // vertex timeout is longer
            .with_max_supersteps(1000); // high limit so it doesn't hit this first

        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime.add_vertex(Arc::new(SlowForeverVertex {
            id: VertexId::new("slow"),
        }));

        let start = std::time::Instant::now();
        let result = runtime.run(TestState::default()).await;
        let elapsed = start.elapsed();

        // Should timeout within ~200ms (some tolerance)
        assert!(elapsed < Duration::from_millis(500), "Took too long: {:?}", elapsed);

        // Should return WorkflowTimeout error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PregelError::WorkflowTimeout(_)),
            "Expected WorkflowTimeout, got {:?}",
            err
        );
    }

    // ============================================
    // C3: Retry Policy Tests (RED - should fail)
    // ============================================

    #[tokio::test]
    async fn test_retry_policy_with_backoff() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static ATTEMPT_COUNT: AtomicUsize = AtomicUsize::new(0);

        // Vertex that fails first 2 times, then succeeds
        struct FailingThenSuccessVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for FailingThenSuccessVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                let attempt = ATTEMPT_COUNT.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    // Fail with recoverable error
                    Err(PregelError::vertex_error(self.id.clone(), format!("transient failure {}", attempt)))
                } else {
                    // Succeed on 3rd attempt
                    Ok(ComputeResult::halt(TestUpdate {
                        counter_delta: 1,
                        messages_delta: 0,
                    }))
                }
            }
        }

        ATTEMPT_COUNT.store(0, Ordering::SeqCst);

        let config = PregelConfig::default()
            .with_retry_policy(
                super::super::config::RetryPolicy::new(3)
                    .with_backoff_base(Duration::from_millis(10))
            )
            .with_max_supersteps(20);

        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime.add_vertex(Arc::new(FailingThenSuccessVertex {
            id: VertexId::new("flaky"),
        }));

        let result = runtime.run(TestState::default()).await;

        // Should succeed after retries
        assert!(result.is_ok(), "Expected success after retries, got {:?}", result);

        // Should have attempted 3 times (2 failures + 1 success)
        assert_eq!(ATTEMPT_COUNT.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_policy_max_exceeded() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static FAIL_COUNT: AtomicUsize = AtomicUsize::new(0);

        // Vertex that always fails
        struct AlwaysFailsVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for AlwaysFailsVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                FAIL_COUNT.fetch_add(1, Ordering::SeqCst);
                Err(PregelError::vertex_error(self.id.clone(), "always fails"))
            }
        }

        FAIL_COUNT.store(0, Ordering::SeqCst);

        let config = PregelConfig::default()
            .with_retry_policy(super::super::config::RetryPolicy::new(3))
            .with_max_supersteps(100);

        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime.add_vertex(Arc::new(AlwaysFailsVertex {
            id: VertexId::new("failing"),
        }));

        let result = runtime.run(TestState::default()).await;

        // Should fail with MaxRetriesExceeded
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PregelError::MaxRetriesExceeded { .. }),
            "Expected MaxRetriesExceeded, got {:?}",
            err
        );

        // Should have tried exactly max_retries + 1 times (initial + retries)
        assert_eq!(FAIL_COUNT.load(Ordering::SeqCst), 4); // 1 initial + 3 retries
    }

    #[tokio::test]
    async fn test_edge_driven_only_entry_active() {
        use super::super::config::ExecutionMode;

        let config = PregelConfig::default()
            .with_execution_mode(ExecutionMode::EdgeDriven);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("a"), increment: 1 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("b"), increment: 1 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("c"), increment: 1 }))
            .set_entry("a");

        // Only "a" should be Active
        assert!(runtime.vertex_states.get(&VertexId::new("a")).unwrap().is_active(),
            "Entry vertex 'a' should be Active");
        assert!(runtime.vertex_states.get(&VertexId::new("b")).unwrap().is_halted(),
            "Non-entry vertex 'b' should be Halted");
        assert!(runtime.vertex_states.get(&VertexId::new("c")).unwrap().is_halted(),
            "Non-entry vertex 'c' should be Halted");
    }

    #[tokio::test]
    async fn test_message_based_all_active_backward_compat() {
        use super::super::config::ExecutionMode;

        let config = PregelConfig::default()
            .with_execution_mode(ExecutionMode::MessageBased);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("a"), increment: 1 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("b"), increment: 1 }));

        // Both should be Active (backward compatible)
        assert!(runtime.vertex_states.get(&VertexId::new("a")).unwrap().is_active());
        assert!(runtime.vertex_states.get(&VertexId::new("b")).unwrap().is_active());
    }

    #[tokio::test]
    async fn test_edge_driven_auto_activation() {
        use super::super::config::ExecutionMode;
        use std::sync::atomic::{AtomicBool, Ordering};

        // Vertex that halts immediately without sending messages
        struct HaltImmediatelyVertex {
            id: VertexId,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for HaltImmediatelyVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                Ok(ComputeResult::halt(TestUpdate::empty()))
            }
        }

        // Vertex that records if it was activated
        struct RecordActivationVertex {
            id: VertexId,
            activated: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for RecordActivationVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                if ctx.has_messages() {
                    self.activated.store(true, Ordering::SeqCst);
                }
                Ok(ComputeResult::halt(TestUpdate::empty()))
            }
        }

        let activated = Arc::new(AtomicBool::new(false));

        let config = PregelConfig::default()
            .with_execution_mode(ExecutionMode::EdgeDriven);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        runtime
            .add_vertex(Arc::new(HaltImmediatelyVertex { id: VertexId::new("entry") }))
            .add_vertex(Arc::new(RecordActivationVertex {
                id: VertexId::new("target"),
                activated: Arc::clone(&activated),
            }))
            .set_entry("entry")
            .add_edge("entry", "target");

        let result = runtime.run(TestState::default()).await;
        assert!(result.is_ok(), "Workflow should complete successfully");

        // Target should have been activated via edge
        assert!(activated.load(Ordering::SeqCst),
            "Target vertex was not activated via edge - C2 fix not working");
    }

    #[tokio::test]
    async fn test_edge_driven_chain_execution() {
        use super::super::config::ExecutionMode;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Use thread-local to avoid test interference
        thread_local! {
            static EXECUTION_ORDER: AtomicUsize = AtomicUsize::new(0);
        }

        struct OrderedVertex {
            id: VertexId,
            expected_order: usize,
        }

        #[async_trait]
        impl Vertex<TestState, WorkflowMessage> for OrderedVertex {
            fn id(&self) -> &VertexId {
                &self.id
            }

            async fn compute(
                &self,
                _ctx: &mut ComputeContext<'_, TestState, WorkflowMessage>,
            ) -> Result<ComputeResult<TestUpdate>, PregelError> {
                let order = EXECUTION_ORDER.with(|counter| counter.fetch_add(1, Ordering::SeqCst));
                assert_eq!(order, self.expected_order,
                    "Vertex {} executed out of order", self.id);
                Ok(ComputeResult::halt(TestUpdate::empty()))
            }
        }

        EXECUTION_ORDER.with(|counter| counter.store(0, Ordering::SeqCst));

        let config = PregelConfig::default()
            .with_execution_mode(ExecutionMode::EdgeDriven);
        let mut runtime: PregelRuntime<TestState, WorkflowMessage> =
            PregelRuntime::with_config(config);

        // Create chain: A -> B -> C
        runtime
            .add_vertex(Arc::new(OrderedVertex { id: VertexId::new("a"), expected_order: 0 }))
            .add_vertex(Arc::new(OrderedVertex { id: VertexId::new("b"), expected_order: 1 }))
            .add_vertex(Arc::new(OrderedVertex { id: VertexId::new("c"), expected_order: 2 }))
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("b", "c");

        let result = runtime.run(TestState::default()).await;
        assert!(result.is_ok());
        assert_eq!(EXECUTION_ORDER.with(|c| c.load(Ordering::SeqCst)), 3, "All 3 vertices should execute");
    }

    // =========================================================================
    // Visualization Integration Tests
    // =========================================================================

    #[test]
    fn test_to_mermaid_simple_chain() {
        use std::sync::Arc;

        let mut runtime = PregelRuntime::<TestState, WorkflowMessage>::new();

        // Create simple chain: start -> agent -> tool -> end
        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("start"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("agent"), increment: 1 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("tool"), increment: 1 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("end"), increment: 0 }))
            .set_entry("start")
            .add_edge("start", "agent")
            .add_edge("agent", "tool")
            .add_edge("tool", "end");

        let mermaid = runtime.to_mermaid();
        println!("=== Simple Chain Diagram ===\n{}", mermaid);

        // Verify structure
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("start"));
        assert!(mermaid.contains("agent"));
        assert!(mermaid.contains("tool"));
        assert!(mermaid.contains("end"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_to_mermaid_with_state_shows_classes() {
        use std::sync::Arc;

        let mut runtime = PregelRuntime::<TestState, WorkflowMessage>::new();

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("active_node"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("halted_node"), increment: 0 }))
            .set_entry("active_node")
            .add_edge("active_node", "halted_node");

        let mermaid = runtime.to_mermaid_with_state();
        println!("=== State Diagram ===\n{}", mermaid);

        // Entry is Active, others are Halted (MessageBased mode)
        assert!(mermaid.contains(":::active") || mermaid.contains(":::halted"));
        assert!(mermaid.contains("classDef active"));
        assert!(mermaid.contains("classDef halted"));
        assert!(mermaid.contains("classDef completed"));
    }

    #[test]
    fn test_to_mermaid_with_node_kinds() {
        use std::sync::Arc;
        use crate::workflow::NodeKind;

        let mut runtime = PregelRuntime::<TestState, WorkflowMessage>::new();

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("planner"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("search_tool"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("router"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("parallel_split"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("merge_results"), increment: 0 }))
            .set_entry("planner")
            .add_edge("planner", "search_tool")
            .add_edge("search_tool", "router")
            .add_edge("router", "parallel_split")
            .add_edge("parallel_split", "merge_results");

        // Provide NodeKind metadata
        let mut kinds = HashMap::new();
        kinds.insert(VertexId::new("planner"), NodeKind::Agent(Default::default()));
        kinds.insert(VertexId::new("search_tool"), NodeKind::Tool(Default::default()));
        kinds.insert(VertexId::new("router"), NodeKind::Router(Default::default()));
        kinds.insert(VertexId::new("parallel_split"), NodeKind::FanOut(Default::default()));
        kinds.insert(VertexId::new("merge_results"), NodeKind::FanIn(Default::default()));

        let mermaid = runtime.to_mermaid_with_kinds(&kinds);
        println!("=== Node Kinds Diagram ===\n{}", mermaid);

        // Verify different shapes
        assert!(mermaid.contains("[planner]"));       // Agent: rectangle
        assert!(mermaid.contains("[[search_tool]]")); // Tool: subroutine
        assert!(mermaid.contains("{router}"));        // Router: diamond
        assert!(mermaid.contains("[/parallel_split\\]")); // FanOut: parallelogram
        assert!(mermaid.contains("[\\merge_results/]"));  // FanIn: reverse para
    }

    #[test]
    fn test_to_mermaid_research_workflow() {
        use std::sync::Arc;
        use crate::workflow::NodeKind;

        // Create a realistic research workflow
        let mut runtime = PregelRuntime::<TestState, WorkflowMessage>::new();

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("orchestrator"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("planner"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("router"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("researcher"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("web_search"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("synthesizer"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("report_writer"), increment: 0 }))
            .set_entry("orchestrator")
            .add_edge("orchestrator", "planner")
            .add_edge("planner", "router")
            .add_edge("router", "researcher")
            .add_edge("router", "synthesizer")
            .add_edge("researcher", "web_search")
            .add_edge("web_search", "synthesizer")
            .add_edge("synthesizer", "report_writer");

        let mut kinds = HashMap::new();
        kinds.insert(VertexId::new("orchestrator"), NodeKind::Agent(Default::default()));
        kinds.insert(VertexId::new("planner"), NodeKind::Agent(Default::default()));
        kinds.insert(VertexId::new("router"), NodeKind::Router(Default::default()));
        kinds.insert(VertexId::new("researcher"), NodeKind::SubAgent(Default::default()));
        kinds.insert(VertexId::new("web_search"), NodeKind::Tool(Default::default()));
        kinds.insert(VertexId::new("synthesizer"), NodeKind::Agent(Default::default()));
        kinds.insert(VertexId::new("report_writer"), NodeKind::Agent(Default::default()));

        let mermaid = runtime.to_mermaid_with_state_and_kinds(&kinds);
        println!("=== Research Workflow Diagram ===\n{}", mermaid);

        // Verify it's valid mermaid
        assert!(mermaid.starts_with("graph TD"));
        assert!(mermaid.contains("orchestrator"));
        assert!(mermaid.contains("classDef"));
    }

    #[test]
    fn test_log_state_output() {
        use std::sync::Arc;

        let mut runtime = PregelRuntime::<TestState, WorkflowMessage>::new();

        runtime
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("node_a"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("node_b"), increment: 0 }))
            .add_vertex(Arc::new(IncrementVertex { id: VertexId::new("node_c"), increment: 0 }))
            .set_entry("node_a")
            .add_edge("node_a", "node_b")
            .add_edge("node_b", "node_c");

        println!("=== Terminal State Log ===");
        runtime.log_state(0);
        // Visual inspection - should show:
        // [Superstep 0]
        //   ▶ node_a : Active
        //   ⏸ node_b : Halted (or Active in MessageBased)
        //   ⏸ node_c : Halted
    }
}
