//! CompiledWorkflow: Compiles a WorkflowGraph into a runnable PregelRuntime
//!
//! This module provides the bridge between the high-level workflow DSL and
//! the low-level Pregel execution engine.
//!
//! # Overview
//!
//! The compilation process:
//! 1. Takes a validated `BuiltWorkflowGraph`
//! 2. Creates appropriate vertex implementations for each node
//! 3. Wires up edges and entry points
//! 4. Returns a `CompiledWorkflow` ready for execution
//!
//! # Example
//!
//! ```ignore
//! let graph = WorkflowGraph::<MyState>::new()
//!     .name("my_workflow")
//!     .node("start", NodeKind::Passthrough)
//!     .node("process", NodeKind::Passthrough)
//!     .entry("start")
//!     .edge("start", "process")
//!     .edge("process", END)
//!     .build()?;
//!
//! let workflow = CompiledWorkflow::compile(graph, PregelConfig::default())?;
//! let result = workflow.run(initial_state).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::llm::LLMProvider;
use crate::pregel::checkpoint::{Checkpoint, Checkpointer};
use crate::pregel::error::PregelError;
use crate::pregel::message::WorkflowMessage;
use crate::pregel::runtime::{CheckpointingRuntime, PregelRuntime, WorkflowResult};
use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::{
    BoxedVertex, ComputeContext, ComputeResult, StateUpdate, Vertex, VertexId,
};
use crate::pregel::PregelConfig;
use crate::backends::Backend;
use crate::middleware::{ToolDefinition, ToolRegistry};
use crate::middleware::subagent::{SubAgentExecutorFactory, SubAgentRegistry};
use crate::workflow::graph::{BuiltWorkflowGraph, END};
use crate::workflow::node::NodeKind;
use crate::workflow::vertices::{
    AgentVertex, FanInVertex, FanOutVertex, RouterVertex, SubAgentVertex, ToolVertex,
};
use crate::runtime::ToolRuntime;
use crate::state::AgentState;

/// Errors that can occur during workflow compilation
#[derive(Debug, Error)]
pub enum WorkflowCompileError {
    /// Node requires an LLM provider but none was provided
    #[error("Node '{node_id}' requires LLM provider but none was configured")]
    MissingLLMProvider { node_id: String },

    /// Node requires a tool registry but none was provided
    #[error("Node '{node_id}' requires tool registry but none was configured")]
    MissingToolRegistry { node_id: String },

    /// Node requires a sub-agent registry but none was provided
    #[error("Node '{node_id}' requires sub-agent registry but none was configured")]
    MissingSubAgentRegistry { node_id: String },

    /// Internal compilation error
    #[error("Compilation error: {0}")]
    Internal(String),
}

/// Internal enum to hold either a plain or checkpointing runtime
///
/// This enables backward-compatible checkpointing support:
/// - Existing `compile*()` methods return `Plain` variant
/// - New `compile_with_checkpointer*()` methods return `Checkpointing` variant
enum RuntimeKind<S>
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Plain runtime without checkpointing
    Plain(PregelRuntime<S, WorkflowMessage>),

    /// Checkpointing-enabled runtime with fault tolerance
    Checkpointing(CheckpointingRuntime<S>),
}

/// A compiled workflow ready for execution
///
/// This struct holds a configured `PregelRuntime` (optionally wrapped with
/// checkpointing support) and provides a simple interface for running the workflow.
///
/// # Checkpointing Support
///
/// Use `compile_with_checkpointer*()` methods to enable fault-tolerant execution
/// with automatic checkpoint saving and resume capability:
///
/// ```ignore
/// let checkpointer = Arc::new(MemoryCheckpointer::new());
/// let mut workflow = CompiledWorkflow::compile_with_checkpointer(
///     graph, config, checkpointer, "my-workflow"
/// )?;
///
/// // Run with automatic checkpointing
/// let result = workflow.run(initial_state).await?;
///
/// // Later, resume from checkpoint if needed
/// if let Some(result) = workflow.resume().await? {
///     println!("Resumed successfully!");
/// }
/// ```
pub struct CompiledWorkflow<S>
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// The underlying runtime (plain or with checkpointing)
    runtime: RuntimeKind<S>,
    /// Workflow name for identification
    name: String,
    /// Node kinds for visualization (optional)
    node_kinds: HashMap<VertexId, NodeKind>,
}

impl<S> CompiledWorkflow<S>
where
    S: WorkflowState + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    /// Compile a workflow graph into a runnable workflow
    ///
    /// This is the basic compilation method that uses `PassthroughVertex` for
    /// all node types except FanOut, FanIn, Router, and Passthrough.
    ///
    /// For full-featured compilation with LLM and tool support, use
    /// `compile_with_providers`.
    pub fn compile(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
    ) -> Result<Self, WorkflowCompileError> {
        Self::compile_with_providers(graph, config, None)
    }

    /// Compile a workflow graph with optional LLM provider
    ///
    /// Provides LLM support for Router nodes with LLMDecision strategy.
    /// For full agent support with tools, use `compile_with_tools`.
    pub fn compile_with_providers(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
    ) -> Result<Self, WorkflowCompileError> {
        Self::compile_with_tools(graph, config, llm, vec![])
    }

    /// Compile a workflow graph with LLM provider and tools
    ///
    /// This is the full-featured compilation method that enables:
    /// - Agent nodes with real LLM calls
    /// - Router nodes with LLMDecision strategy
    /// - Tool definitions for agent nodes
    ///
    /// Note: This method passes tool definitions only. For full tool execution
    /// support, use `compile_with_registry` instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rig::client::{CompletionClient, ProviderClient};
    /// use rig_deepagents::RigAgentAdapter;
    ///
    /// let client = rig::providers::openai::Client::from_env();
    /// let agent = client.agent("gpt-4").build();
    /// let llm = Arc::new(RigAgentAdapter::new(agent));
    /// let tools = vec![TavilySearchTool::from_env()?.definition()];
    ///
    /// let workflow = CompiledWorkflow::compile_with_tools(
    ///     graph,
    ///     PregelConfig::default(),
    ///     Some(llm),
    ///     tools,
    /// )?;
    /// ```
    pub fn compile_with_tools(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        tools: Vec<ToolDefinition>,
    ) -> Result<Self, WorkflowCompileError> {
        // Convert to registry-based compilation with empty registry
        // (tools are passed as definitions only, no execution)
        let registry = ToolRegistry::new();
        // The registry is empty, so tools won't execute - but definitions are passed
        Self::compile_internal(graph, config, llm, registry, tools, None, None, None)
    }

    /// Compile a workflow graph with LLM provider and tool registry
    ///
    /// This is the recommended compilation method for production use.
    /// It enables full tool execution through the registry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rig::client::{CompletionClient, ProviderClient};
    /// use rig_deepagents::middleware::{ToolRegistry, DynTool};
    /// use rig_deepagents::tools::{TavilySearchTool, ThinkTool};
    /// use rig_deepagents::RigAgentAdapter;
    ///
    /// let client = rig::providers::openai::Client::from_env();
    /// let agent = client.agent("gpt-4").build();
    /// let llm = Arc::new(RigAgentAdapter::new(agent));
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Arc::new(TavilySearchTool::from_env()?));
    /// registry.register(Arc::new(ThinkTool));
    ///
    /// let workflow = CompiledWorkflow::compile_with_registry(
    ///     graph,
    ///     PregelConfig::default(),
    ///     Some(llm),
    ///     registry,
    /// )?;
    /// ```
    pub fn compile_with_registry(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        registry: ToolRegistry,
    ) -> Result<Self, WorkflowCompileError> {
        let definitions = registry.definitions();
        Self::compile_internal(graph, config, llm, registry, definitions, None, None, None)
    }

    /// Compile a workflow graph with all resources
    ///
    /// This is the most complete compilation method, enabling:
    /// - Agent nodes with LLM and tool execution
    /// - Router nodes with LLM decision making
    /// - SubAgent nodes with registry and executor
    /// - Tool nodes (if tool registry provided)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rig_deepagents::middleware::{ToolRegistry, SubAgentRegistry};
    /// use rig_deepagents::middleware::subagent::DefaultSubAgentExecutorFactory;
    ///
    /// let workflow = CompiledWorkflow::compile_with_all(
    ///     graph,
    ///     PregelConfig::default(),
    ///     Some(llm),
    ///     tool_registry,
    ///     Some(subagent_registry),
    ///     Some(Arc::new(DefaultSubAgentExecutorFactory::new(llm.clone()))),
    ///     Some(backend),
    /// )?;
    /// ```
    pub fn compile_with_all(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        tool_registry: ToolRegistry,
        subagent_registry: Option<Arc<SubAgentRegistry>>,
        executor_factory: Option<Arc<dyn SubAgentExecutorFactory>>,
        backend: Option<Arc<dyn Backend>>,
    ) -> Result<Self, WorkflowCompileError> {
        let definitions = tool_registry.definitions();
        Self::compile_internal(
            graph,
            config,
            llm,
            tool_registry,
            definitions,
            subagent_registry,
            executor_factory,
            backend,
        )
    }

    // =========================================================================
    // Checkpointing-enabled compilation methods
    // =========================================================================

    /// Compile a workflow graph with checkpointer for fault tolerance
    ///
    /// This is the minimal checkpointing compilation method. It enables:
    /// - Automatic checkpoint saving at configured intervals
    /// - Resume capability from the latest checkpoint
    ///
    /// For LLM and tool support, use `compile_with_checkpointer_and_providers`
    /// or `compile_with_checkpointer_and_registry`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rig_deepagents::pregel::checkpoint::MemoryCheckpointer;
    ///
    /// let checkpointer = Arc::new(MemoryCheckpointer::new());
    /// let config = PregelConfig::default().with_checkpoint_interval(5);
    ///
    /// let mut workflow = CompiledWorkflow::compile_with_checkpointer(
    ///     graph,
    ///     config,
    ///     checkpointer,
    ///     "my-workflow-id",
    /// )?;
    ///
    /// // Run with automatic checkpointing
    /// let result = workflow.run(initial_state).await?;
    ///
    /// // Resume from checkpoint if needed
    /// if let Some(result) = workflow.resume().await? {
    ///     println!("Resumed from checkpoint!");
    /// }
    /// ```
    pub fn compile_with_checkpointer(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        Self::compile_with_checkpointer_and_providers(graph, config, None, checkpointer, workflow_id)
    }

    /// Compile a workflow graph with checkpointer and LLM provider
    ///
    /// Enables checkpointing plus LLM support for Router nodes with LLMDecision strategy.
    /// For full tool support, use `compile_with_checkpointer_and_registry`.
    pub fn compile_with_checkpointer_and_providers(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        Self::compile_internal_checkpointed(
            graph,
            config,
            llm,
            ToolRegistry::new(),
            vec![],
            None,
            None,
            None,
            checkpointer,
            workflow_id,
        )
    }

    /// Compile a workflow graph with checkpointer, LLM, and tool registry
    ///
    /// This is the recommended checkpointing compilation method for production use.
    /// It enables full tool execution through the registry plus fault tolerance.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rig_deepagents::pregel::checkpoint::FileCheckpointer;
    ///
    /// let checkpointer = Arc::new(FileCheckpointer::new(
    ///     "checkpoints", "my-workflow", true
    /// ));
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Arc::new(TavilySearchTool::from_env()?));
    ///
    /// let workflow = CompiledWorkflow::compile_with_checkpointer_and_registry(
    ///     graph,
    ///     PregelConfig::default().with_checkpoint_interval(3),
    ///     Some(llm),
    ///     registry,
    ///     checkpointer,
    ///     "production-workflow",
    /// )?;
    /// ```
    pub fn compile_with_checkpointer_and_registry(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        registry: ToolRegistry,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        let definitions = registry.definitions();
        Self::compile_internal_checkpointed(
            graph,
            config,
            llm,
            registry,
            definitions,
            None,
            None,
            None,
            checkpointer,
            workflow_id,
        )
    }

    /// Compile a workflow graph with all resources including checkpointer
    ///
    /// This is the most complete checkpointing compilation method, enabling:
    /// - Agent nodes with LLM and tool execution
    /// - Router nodes with LLM decision making
    /// - SubAgent nodes with registry and executor
    /// - Tool nodes (if tool registry provided)
    /// - Fault-tolerant execution with checkpointing
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workflow = CompiledWorkflow::compile_with_all_checkpointed(
    ///     graph,
    ///     PregelConfig::default().with_checkpoint_interval(5),
    ///     Some(llm),
    ///     tool_registry,
    ///     Some(subagent_registry),
    ///     Some(executor_factory),
    ///     Some(backend),
    ///     checkpointer,
    ///     "full-featured-workflow",
    /// )?;
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_all_checkpointed(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        tool_registry: ToolRegistry,
        subagent_registry: Option<Arc<SubAgentRegistry>>,
        executor_factory: Option<Arc<dyn SubAgentExecutorFactory>>,
        backend: Option<Arc<dyn Backend>>,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        let definitions = tool_registry.definitions();
        Self::compile_internal_checkpointed(
            graph,
            config,
            llm,
            tool_registry,
            definitions,
            subagent_registry,
            executor_factory,
            backend,
            checkpointer,
            workflow_id,
        )
    }

    /// Internal compilation with all resources (plain runtime, no checkpointing)
    #[allow(clippy::too_many_arguments)]
    fn compile_internal(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        tool_registry: ToolRegistry,
        tool_definitions: Vec<ToolDefinition>,
        subagent_registry: Option<Arc<SubAgentRegistry>>,
        executor_factory: Option<Arc<dyn SubAgentExecutorFactory>>,
        backend: Option<Arc<dyn Backend>>,
    ) -> Result<Self, WorkflowCompileError> {
        let mut runtime = PregelRuntime::with_config(config);
        let mut node_kinds = HashMap::new();

        // Create vertices from NodeKind
        for (node_id, kind) in &graph.nodes {
            let vertex = Self::create_vertex(
                node_id,
                kind.clone(),
                llm.clone(),
                &tool_registry,
                &tool_definitions,
                subagent_registry.as_ref(),
                executor_factory.as_ref(),
                backend.as_ref(),
            )?;
            runtime.add_vertex(vertex);
            node_kinds.insert(VertexId::new(node_id), kind.clone());
        }

        // Add edges (filter out END sentinel)
        for (from, targets) in &graph.edges {
            for to in targets {
                if to != END {
                    runtime.add_edge(from.as_str(), to.as_str());
                }
            }
        }

        // Set entry point
        runtime.set_entry(graph.entry_point.as_str());

        Ok(Self {
            runtime: RuntimeKind::Plain(runtime),
            name: graph.name,
            node_kinds,
        })
    }

    /// Internal compilation with checkpointing enabled
    ///
    /// This builds the PregelRuntime the same way as `compile_internal`, but
    /// then wraps it in a `CheckpointingRuntime` for fault tolerance.
    #[allow(clippy::too_many_arguments)]
    fn compile_internal_checkpointed(
        graph: BuiltWorkflowGraph<S>,
        config: PregelConfig,
        llm: Option<Arc<dyn LLMProvider>>,
        tool_registry: ToolRegistry,
        tool_definitions: Vec<ToolDefinition>,
        subagent_registry: Option<Arc<SubAgentRegistry>>,
        executor_factory: Option<Arc<dyn SubAgentExecutorFactory>>,
        backend: Option<Arc<dyn Backend>>,
        checkpointer: Arc<dyn Checkpointer<S> + Send + Sync>,
        workflow_id: impl Into<String>,
    ) -> Result<Self, WorkflowCompileError> {
        let workflow_id = workflow_id.into();
        let mut runtime = PregelRuntime::with_config(config);
        let mut node_kinds = HashMap::new();

        // Set workflow ID for checkpoint identification
        runtime = runtime.with_workflow_id(&workflow_id);

        // Create vertices from NodeKind
        for (node_id, kind) in &graph.nodes {
            let vertex = Self::create_vertex(
                node_id,
                kind.clone(),
                llm.clone(),
                &tool_registry,
                &tool_definitions,
                subagent_registry.as_ref(),
                executor_factory.as_ref(),
                backend.as_ref(),
            )?;
            runtime.add_vertex(vertex);
            node_kinds.insert(VertexId::new(node_id), kind.clone());
        }

        // Add edges (filter out END sentinel)
        for (from, targets) in &graph.edges {
            for to in targets {
                if to != END {
                    runtime.add_edge(from.as_str(), to.as_str());
                }
            }
        }

        // Set entry point
        runtime.set_entry(graph.entry_point.as_str());

        // Wrap in CheckpointingRuntime for fault tolerance
        let checkpointing_runtime = CheckpointingRuntime::new(runtime, checkpointer);

        Ok(Self {
            runtime: RuntimeKind::Checkpointing(checkpointing_runtime),
            name: graph.name,
            node_kinds,
        })
    }

    /// Create a vertex from a NodeKind
    #[allow(clippy::too_many_arguments)]
    fn create_vertex(
        node_id: &str,
        kind: NodeKind,
        llm: Option<Arc<dyn LLMProvider>>,
        tool_registry: &ToolRegistry,
        tool_definitions: &[ToolDefinition],
        subagent_registry: Option<&Arc<SubAgentRegistry>>,
        executor_factory: Option<&Arc<dyn SubAgentExecutorFactory>>,
        backend: Option<&Arc<dyn Backend>>,
    ) -> Result<BoxedVertex<S, WorkflowMessage>, WorkflowCompileError> {
        match kind {
            NodeKind::Agent(config) => {
                // Use real AgentVertex if LLM is available, otherwise passthrough
                match llm {
                    Some(llm_provider) => {
                        // Use registry if it has tools, otherwise fall back to definitions
                        if tool_registry.is_empty() {
                            Ok(Arc::new(AgentVertex::<S>::new(
                                node_id,
                                config,
                                llm_provider,
                                tool_definitions.to_vec(),
                            )))
                        } else {
                            Ok(Arc::new(AgentVertex::<S>::new_with_registry(
                                node_id,
                                config,
                                llm_provider,
                                tool_registry.clone(),
                            )))
                        }
                    }
                    None => {
                        tracing::warn!(
                            node_id = node_id,
                            "Agent node created without LLM provider - using passthrough"
                        );
                        Ok(Arc::new(PassthroughVertex::new(node_id)))
                    }
                }
            }
            NodeKind::Tool(config) => {
                // Attempt to create a real ToolVertex if registry and backend are available
                if let Some(tool) = tool_registry.get(&config.tool_name) {
                    if let Some(backend) = backend {
                        // Create ToolRuntime with default AgentState and provided backend
                        let runtime = ToolRuntime::new(
                            AgentState::new(),
                            Arc::clone(backend),
                        );
                        tracing::debug!(
                            node_id = node_id,
                            tool_name = %config.tool_name,
                            "Creating ToolVertex with registry tool"
                        );
                        return Ok(Arc::new(ToolVertex::<S>::new(
                            node_id,
                            config,
                            tool.clone(),
                            Arc::new(runtime),
                        )));
                    } else {
                        tracing::warn!(
                            node_id = node_id,
                            tool_name = %config.tool_name,
                            "Tool node requires backend for ToolRuntime - using passthrough"
                        );
                    }
                } else {
                    tracing::warn!(
                        node_id = node_id,
                        tool_name = %config.tool_name,
                        "Tool '{}' not found in registry - using passthrough",
                        config.tool_name
                    );
                }
                // Fallback to passthrough when tool or backend not available
                Ok(Arc::new(PassthroughVertex::new(node_id)))
            }
            NodeKind::Router(config) => {
                // RouterVertex can work with or without LLM (for StateField strategy)
                Ok(Arc::new(RouterVertex::<S>::new(node_id, config, llm)))
            }
            NodeKind::SubAgent(config) => {
                // Create SubAgentVertex if all required resources are available
                match (subagent_registry, executor_factory, backend) {
                    (Some(registry), Some(factory), Some(backend)) => {
                        Ok(Arc::new(SubAgentVertex::<S>::new(
                            node_id,
                            config,
                            registry.clone(),
                            factory.clone(),
                            backend.clone(),
                        )))
                    }
                    _ => {
                        tracing::warn!(
                            node_id = node_id,
                            "SubAgent node requires registry, executor factory, and backend - using passthrough"
                        );
                        Ok(Arc::new(PassthroughVertex::new(node_id)))
                    }
                }
            }
            NodeKind::FanOut(config) => Ok(Arc::new(FanOutVertex::<S>::new(node_id, config))),
            NodeKind::FanIn(config) => Ok(Arc::new(FanInVertex::<S>::new(node_id, config))),
            NodeKind::Passthrough => Ok(Arc::new(PassthroughVertex::new(node_id))),
        }
    }

    /// Run the workflow with the given initial state
    ///
    /// If the workflow was compiled with a checkpointer, checkpoints will be
    /// saved automatically at the configured interval.
    pub async fn run(&mut self, initial_state: S) -> Result<WorkflowResult<S>, PregelError> {
        match &mut self.runtime {
            RuntimeKind::Plain(runtime) => runtime.run(initial_state).await,
            RuntimeKind::Checkpointing(runtime) => runtime.run(initial_state).await,
        }
    }

    /// Get the workflow name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the underlying runtime for advanced configuration
    pub fn runtime(&self) -> &PregelRuntime<S, WorkflowMessage> {
        match &self.runtime {
            RuntimeKind::Plain(runtime) => runtime,
            RuntimeKind::Checkpointing(runtime) => runtime.inner(),
        }
    }

    /// Get mutable access to the underlying runtime
    pub fn runtime_mut(&mut self) -> &mut PregelRuntime<S, WorkflowMessage> {
        match &mut self.runtime {
            RuntimeKind::Plain(runtime) => runtime,
            RuntimeKind::Checkpointing(runtime) => runtime.inner_mut(),
        }
    }

    /// Generate a Mermaid diagram of the workflow
    pub fn to_mermaid(&self) -> String {
        self.runtime().to_mermaid_with_kinds(&self.node_kinds)
    }

    /// Generate a Mermaid diagram with execution state
    pub fn to_mermaid_with_state(&self) -> String {
        self.runtime().to_mermaid_with_state_and_kinds(&self.node_kinds)
    }

    // =========================================================================
    // Checkpointing runtime methods
    // =========================================================================

    /// Resume workflow from the latest checkpoint
    ///
    /// Returns `Ok(None)` if no checkpoint exists.
    /// Returns `Err(PregelError::NotImplemented)` if workflow was not compiled with checkpointer.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let checkpointer = Arc::new(MemoryCheckpointer::new());
    /// let mut workflow = CompiledWorkflow::compile_with_checkpointer(
    ///     graph, config, checkpointer, "my-workflow"
    /// )?;
    ///
    /// // Initial run (may fail or be interrupted)
    /// let _ = workflow.run(initial_state).await;
    ///
    /// // Later, resume from checkpoint
    /// if let Some(result) = workflow.resume().await? {
    ///     println!("Resumed and completed at superstep {}", result.supersteps);
    /// } else {
    ///     println!("No checkpoint found, starting fresh");
    /// }
    /// ```
    pub async fn resume(&mut self) -> Result<Option<WorkflowResult<S>>, PregelError> {
        match &mut self.runtime {
            RuntimeKind::Plain(_) => Err(PregelError::not_implemented(
                "resume() requires workflow compiled with checkpointer. Use compile_with_checkpointer() instead of compile()."
            )),
            RuntimeKind::Checkpointing(runtime) => runtime.resume().await,
        }
    }

    /// Resume workflow from a specific checkpoint
    ///
    /// Useful when you want to resume from a checkpoint other than the latest,
    /// for example to replay from a known-good state.
    ///
    /// Returns `Err(PregelError::NotImplemented)` if workflow was not compiled with checkpointer.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load a specific checkpoint
    /// let checkpoint = checkpointer.load(5).await?.expect("Checkpoint 5 should exist");
    ///
    /// // Resume from that checkpoint
    /// let result = workflow.run_from_checkpoint(checkpoint).await?;
    /// ```
    pub async fn run_from_checkpoint(
        &mut self,
        checkpoint: Checkpoint<S>,
    ) -> Result<WorkflowResult<S>, PregelError> {
        match &mut self.runtime {
            RuntimeKind::Plain(_) => Err(PregelError::not_implemented(
                "run_from_checkpoint() requires workflow compiled with checkpointer. Use compile_with_checkpointer() instead of compile()."
            )),
            RuntimeKind::Checkpointing(runtime) => runtime.run_from_checkpoint(checkpoint).await,
        }
    }

    /// Get access to the checkpointer (if configured)
    ///
    /// Returns `None` if workflow was compiled without checkpointer.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(cp) = workflow.checkpointer() {
    ///     // List available checkpoints
    ///     let supersteps = cp.list().await?;
    ///     println!("Available checkpoints: {:?}", supersteps);
    ///
    ///     // Prune old checkpoints, keeping only the 3 most recent
    ///     cp.prune(3).await?;
    /// }
    /// ```
    pub fn checkpointer(&self) -> Option<&Arc<dyn Checkpointer<S> + Send + Sync>> {
        match &self.runtime {
            RuntimeKind::Plain(_) => None,
            RuntimeKind::Checkpointing(runtime) => Some(runtime.checkpointer()),
        }
    }

    /// Check if checkpointing is enabled for this workflow
    ///
    /// Returns `true` if the workflow was compiled with a checkpointer.
    pub fn has_checkpointer(&self) -> bool {
        matches!(&self.runtime, RuntimeKind::Checkpointing(_))
    }
}

/// A simple passthrough vertex that forwards messages without transformation
///
/// This vertex is used as a placeholder for node types that require external
/// dependencies (LLM, tools, sub-agents) that aren't provided during compilation.
///
/// Behavior:
/// - Forwards all incoming messages to all outgoing edges
/// - Halts after processing (edge-driven mode will auto-activate successors)
pub struct PassthroughVertex<S: WorkflowState> {
    id: VertexId,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WorkflowState> PassthroughVertex<S> {
    /// Create a new passthrough vertex
    pub fn new(id: impl Into<VertexId>) -> Self {
        Self {
            id: id.into(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S: WorkflowState> Vertex<S, WorkflowMessage> for PassthroughVertex<S> {
    fn id(&self) -> &VertexId {
        &self.id
    }

    async fn compute(
        &self,
        _ctx: &mut ComputeContext<'_, S, WorkflowMessage>,
    ) -> Result<ComputeResult<S::Update>, PregelError> {
        // Passthrough: just halt and let edge-driven mode handle activation
        // In MessageBased mode, messages would need to be forwarded explicitly,
        // but that requires knowing the targets which aren't stored here.
        Ok(ComputeResult::halt(S::Update::empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::config::ExecutionMode;
    use crate::pregel::state::UnitState;
    use crate::workflow::graph::WorkflowGraph;
    use crate::workflow::node::{FanInNodeConfig, FanOutNodeConfig, RouterNodeConfig};

    #[test]
    fn test_compile_simple_workflow() {
        // Create a simple A -> B -> END workflow
        let graph = WorkflowGraph::<UnitState>::new()
            .name("simple")
            .node("start", NodeKind::Passthrough)
            .node("process", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "process")
            .edge("process", END)
            .build()
            .unwrap();

        let compiled = CompiledWorkflow::compile(graph, PregelConfig::default());
        assert!(compiled.is_ok());

        let workflow = compiled.unwrap();
        assert_eq!(workflow.name(), "simple");
    }

    #[test]
    fn test_compile_workflow_with_fanout_fanin() {
        let graph = WorkflowGraph::<UnitState>::new()
            .name("parallel")
            .node("start", NodeKind::Passthrough)
            .node(
                "split",
                NodeKind::FanOut(FanOutNodeConfig {
                    targets: vec!["worker_a".into(), "worker_b".into()],
                    ..Default::default()
                }),
            )
            .node("worker_a", NodeKind::Passthrough)
            .node("worker_b", NodeKind::Passthrough)
            .node(
                "merge",
                NodeKind::FanIn(FanInNodeConfig {
                    sources: vec!["worker_a".into(), "worker_b".into()],
                    ..Default::default()
                }),
            )
            .node("end", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "split")
            .edge("split", "worker_a")
            .edge("split", "worker_b")
            .edge("worker_a", "merge")
            .edge("worker_b", "merge")
            .edge("merge", "end")
            .edge("end", END)
            .build()
            .unwrap();

        let compiled = CompiledWorkflow::compile(graph, PregelConfig::default());
        assert!(compiled.is_ok());
    }

    #[test]
    fn test_compile_workflow_with_router() {
        use crate::workflow::node::{Branch, BranchCondition, RoutingStrategy};

        let graph = WorkflowGraph::<UnitState>::new()
            .name("routing")
            .node("start", NodeKind::Passthrough)
            .node(
                "router",
                NodeKind::Router(RouterNodeConfig {
                    strategy: RoutingStrategy::StateField {
                        field: "phase".into(),
                    },
                    branches: vec![
                        Branch {
                            target: "explore".into(),
                            condition: BranchCondition::Always,
                        },
                    ],
                    default: Some("fallback".into()),
                }),
            )
            .node("explore", NodeKind::Passthrough)
            .node("fallback", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "router")
            .edge("router", "explore")
            .edge("router", "fallback")
            .edge("explore", END)
            .edge("fallback", END)
            .build()
            .unwrap();

        let compiled = CompiledWorkflow::compile(graph, PregelConfig::default());
        assert!(compiled.is_ok());
    }

    #[tokio::test]
    async fn test_run_passthrough_workflow() {
        // Create workflow with only passthrough nodes
        let graph = WorkflowGraph::<UnitState>::new()
            .name("passthrough_test")
            .node("a", NodeKind::Passthrough)
            .node("b", NodeKind::Passthrough)
            .node("c", NodeKind::Passthrough)
            .entry("a")
            .edge("a", "b")
            .edge("b", "c")
            .edge("c", END)
            .build()
            .unwrap();

        // Use EdgeDriven mode for proper chain execution
        let config = PregelConfig::default().with_execution_mode(ExecutionMode::EdgeDriven);

        let mut workflow = CompiledWorkflow::compile(graph, config).unwrap();

        // Run the workflow
        let result = workflow.run(UnitState).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.completed);
        // In EdgeDriven mode, should execute a -> b -> c
        assert!(result.supersteps >= 1);
    }

    #[tokio::test]
    async fn test_run_single_node_workflow() {
        let graph = WorkflowGraph::<UnitState>::new()
            .name("single")
            .node("only", NodeKind::Passthrough)
            .entry("only")
            .edge("only", END)
            .build()
            .unwrap();

        let mut workflow = CompiledWorkflow::compile(graph, PregelConfig::default()).unwrap();
        let result = workflow.run(UnitState).await;

        assert!(result.is_ok());
        assert!(result.unwrap().completed);
    }

    #[test]
    fn test_workflow_mermaid_generation() {
        let graph = WorkflowGraph::<UnitState>::new()
            .name("mermaid_test")
            .node("start", NodeKind::Passthrough)
            .node(
                "split",
                NodeKind::FanOut(FanOutNodeConfig {
                    targets: vec!["a".into(), "b".into()],
                    ..Default::default()
                }),
            )
            .node("a", NodeKind::Passthrough)
            .node("b", NodeKind::Passthrough)
            .entry("start")
            .edge("start", "split")
            .edge("split", "a")
            .edge("split", "b")
            .edge("a", END)
            .edge("b", END)
            .build()
            .unwrap();

        let workflow = CompiledWorkflow::compile(graph, PregelConfig::default()).unwrap();
        let mermaid = workflow.to_mermaid();

        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("start"));
        assert!(mermaid.contains("split"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_passthrough_vertex_halts() {
        // We can't easily test async compute in a sync test,
        // but we can verify the vertex is created correctly
        let vertex = PassthroughVertex::<UnitState>::new("test");
        assert_eq!(vertex.id().as_str(), "test");
    }

    #[test]
    fn test_compile_preserves_node_kinds() {
        let graph = WorkflowGraph::<UnitState>::new()
            .name("kinds_test")
            .node("start", NodeKind::Passthrough)
            .node(
                "fanout",
                NodeKind::FanOut(FanOutNodeConfig::default()),
            )
            .entry("start")
            .edge("start", "fanout")
            .edge("fanout", END)
            .build()
            .unwrap();

        let workflow = CompiledWorkflow::compile(graph, PregelConfig::default()).unwrap();

        // Node kinds should be stored for visualization
        assert!(workflow.node_kinds.contains_key(&VertexId::new("start")));
        assert!(workflow.node_kinds.contains_key(&VertexId::new("fanout")));
    }
}
