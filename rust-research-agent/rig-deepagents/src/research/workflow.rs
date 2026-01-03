//! Pre-built Research Workflow
//!
//! A complete workflow graph for autonomous research following the
//! "breadth-first, then depth" pattern.
//!
//! # Workflow Structure
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        Research Workflow Graph                               │
//! │                                                                              │
//! │  ┌────────────┐                                                              │
//! │  │   START    │                                                              │
//! │  │  (planner) │                                                              │
//! │  └─────┬──────┘                                                              │
//! │        ▼                                                                     │
//! │  ┌────────────┐     ┌────────────────┐                                      │
//! │  │ explorer   │ ──▶ │ phase_router   │                                      │
//! │  │ (Phase 1)  │     │                │                                      │
//! │  └────────────┘     └───────┬────────┘                                      │
//! │                             │                                                │
//! │              ┌──────────────┼──────────────┐                                │
//! │              ▼              ▼              ▼                                │
//! │        ┌──────────┐  ┌──────────┐  ┌──────────┐                            │
//! │        │ directed │  │synthesize│  │  budget  │                            │
//! │        │ (Phase 2)│  │ (Phase 3)│  │ exceeded │                            │
//! │        └────┬─────┘  └────┬─────┘  └────┬─────┘                            │
//! │             │             │             │                                   │
//! │             └──────┬──────┴─────────────┘                                   │
//! │                    ▼                                                        │
//! │              ┌──────────┐                                                   │
//! │              │   END    │                                                   │
//! │              └──────────┘                                                   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::research::{ResearchState, ResearchWorkflowBuilder};
//!
//! let workflow = ResearchWorkflowBuilder::new()
//!     .max_searches(6)
//!     .max_directions(3)
//!     .build()?;
//!
//! let initial_state = ResearchState::new("What is context engineering?");
//! // Execute with PregelRuntime...
//! ```

use crate::workflow::{
    AgentNodeConfig, Branch, BranchCondition, NodeKind, RouterNodeConfig, RoutingStrategy,
    StopCondition, WorkflowBuildError, WorkflowGraph, END,
};

use super::prompts::ResearchPrompts;
use super::state::{ResearchPhase, ResearchState, ResearchUpdate};

/// Builder for constructing research workflows with configurable parameters.
#[derive(Debug, Clone)]
pub struct ResearchWorkflowBuilder {
    /// Name of the workflow
    name: String,

    /// Maximum number of searches allowed
    max_searches: usize,

    /// Maximum number of research directions to explore
    max_directions: usize,

    /// Maximum iterations for the explorer agent
    max_explorer_iterations: usize,

    /// Maximum iterations for the directed research agent
    max_directed_iterations: usize,

    /// Maximum iterations for the synthesizer agent
    max_synthesizer_iterations: usize,
}

impl Default for ResearchWorkflowBuilder {
    fn default() -> Self {
        Self {
            name: "research_workflow".to_string(),
            max_searches: 6,
            max_directions: 3,
            max_explorer_iterations: 5,
            max_directed_iterations: 8,
            max_synthesizer_iterations: 3,
        }
    }
}

impl ResearchWorkflowBuilder {
    /// Create a new research workflow builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the maximum number of searches allowed.
    ///
    /// Default: 6 (2 exploratory + 4 directed)
    pub fn max_searches(mut self, max: usize) -> Self {
        self.max_searches = max;
        self
    }

    /// Set the maximum number of research directions to explore.
    ///
    /// Default: 3
    pub fn max_directions(mut self, max: usize) -> Self {
        self.max_directions = max;
        self
    }

    /// Set the maximum iterations for the explorer agent.
    ///
    /// Default: 5
    pub fn max_explorer_iterations(mut self, max: usize) -> Self {
        self.max_explorer_iterations = max;
        self
    }

    /// Set the maximum iterations for the directed research agent.
    ///
    /// Default: 8
    pub fn max_directed_iterations(mut self, max: usize) -> Self {
        self.max_directed_iterations = max;
        self
    }

    /// Set the maximum iterations for the synthesizer agent.
    ///
    /// Default: 3
    pub fn max_synthesizer_iterations(mut self, max: usize) -> Self {
        self.max_synthesizer_iterations = max;
        self
    }

    /// Build the research workflow graph.
    pub fn build(self) -> Result<WorkflowGraph<ResearchState>, WorkflowBuildError> {
        // Create agent configurations
        let planner_config = AgentNodeConfig {
            system_prompt: ResearchPrompts::planner(),
            max_iterations: 3,
            stop_conditions: vec![StopCondition::NoToolCalls],
            ..Default::default()
        };

        let explorer_config = AgentNodeConfig {
            system_prompt: format!(
                "{}\n\n## Budget\nMax searches for this phase: 2",
                ResearchPrompts::researcher()
            ),
            max_iterations: self.max_explorer_iterations,
            stop_conditions: vec![
                StopCondition::NoToolCalls,
                StopCondition::ContainsText {
                    pattern: "PHASE_COMPLETE".to_string(),
                },
            ],
            ..Default::default()
        };

        let directed_config = AgentNodeConfig {
            system_prompt: format!(
                "{}\n\n## Budget\nMax searches for this phase: {}",
                ResearchPrompts::researcher(),
                self.max_searches.saturating_sub(2) // Reserve 2 for exploratory
            ),
            max_iterations: self.max_directed_iterations,
            stop_conditions: vec![
                StopCondition::NoToolCalls,
                StopCondition::ContainsText {
                    pattern: "PHASE_COMPLETE".to_string(),
                },
            ],
            ..Default::default()
        };

        let synthesizer_config = AgentNodeConfig {
            system_prompt: ResearchPrompts::synthesizer(),
            max_iterations: self.max_synthesizer_iterations,
            stop_conditions: vec![StopCondition::NoToolCalls],
            ..Default::default()
        };

        // Create phase router configuration
        let phase_router_config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "phase".to_string(),
            },
            branches: vec![
                Branch {
                    target: "explorer".to_string(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("Exploratory"),
                    },
                },
                Branch {
                    target: "directed".to_string(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("Directed"),
                    },
                },
                Branch {
                    target: "synthesizer".to_string(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("Synthesis"),
                    },
                },
                Branch {
                    target: END.to_string(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("Complete"),
                    },
                },
            ],
            default: Some(END.to_string()),
        };

        // Create budget check router
        let budget_router_config = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "can_continue".to_string(),
            },
            branches: vec![
                Branch {
                    target: "phase_router".to_string(),
                    condition: BranchCondition::IsTruthy,
                },
                Branch {
                    target: "synthesizer".to_string(),
                    condition: BranchCondition::IsFalsy,
                },
            ],
            default: Some("synthesizer".to_string()),
        };

        // Build the workflow graph
        let graph = WorkflowGraph::<ResearchState>::new()
            .name(&self.name)
            // Entry point: planner analyzes the query
            .node("planner", NodeKind::Agent(planner_config))
            // Phase router: directs to appropriate phase
            .node("phase_router", NodeKind::Router(phase_router_config))
            // Phase 1: Exploratory search
            .node("explorer", NodeKind::Agent(explorer_config))
            // Budget check after exploration
            .node("budget_check", NodeKind::Router(budget_router_config))
            // Phase 2: Directed research
            .node("directed", NodeKind::Agent(directed_config))
            // Phase 3: Synthesis
            .node("synthesizer", NodeKind::Agent(synthesizer_config))
            // Edges
            .entry("planner")
            .edge("planner", "phase_router")
            .edge("explorer", "budget_check")
            .edge("directed", "budget_check")
            .edge("synthesizer", END);

        Ok(graph)
    }
}

/// Configuration for research workflow execution.
#[derive(Debug, Clone)]
pub struct ResearchConfig {
    /// Maximum total searches across all phases
    pub max_searches: usize,

    /// Maximum research directions to explore in Phase 2
    pub max_directions: usize,

    /// Whether to enable parallel direction exploration
    pub parallel_directions: bool,

    /// Timeout for the entire workflow in seconds
    pub timeout_secs: Option<u64>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_searches: 6,
            max_directions: 3,
            parallel_directions: false,
            timeout_secs: None,
        }
    }
}

impl ResearchConfig {
    /// Create a new research configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum searches.
    pub fn with_max_searches(mut self, max: usize) -> Self {
        self.max_searches = max;
        self
    }

    /// Set maximum directions.
    pub fn with_max_directions(mut self, max: usize) -> Self {
        self.max_directions = max;
        self
    }

    /// Enable parallel direction exploration.
    pub fn with_parallel_directions(mut self, enabled: bool) -> Self {
        self.parallel_directions = enabled;
        self
    }

    /// Set workflow timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

/// Helper function to check if research can continue based on budget and phase.
///
/// This delegates to the state's computed `can_continue` field for consistency.
/// The field is automatically updated after each state update.
pub fn can_continue_research(state: &ResearchState) -> bool {
    state.can_continue
}

/// Determine the next phase based on current state.
pub fn determine_next_phase(state: &ResearchState) -> ResearchPhase {
    match state.phase {
        ResearchPhase::Exploratory => {
            // Move to Directed if we have directions to explore
            if !state.directions.is_empty() {
                ResearchPhase::Directed
            } else {
                // Skip to Synthesis if no directions found
                ResearchPhase::Synthesis
            }
        }
        ResearchPhase::Directed => {
            // Move to Synthesis when all directions explored or budget exceeded
            if state.unexplored_directions().is_empty() || !state.can_search() {
                ResearchPhase::Synthesis
            } else {
                ResearchPhase::Directed
            }
        }
        ResearchPhase::Synthesis => ResearchPhase::Complete,
        ResearchPhase::Complete => ResearchPhase::Complete,
    }
}

/// Create a phase transition update.
pub fn phase_transition_update(current: &ResearchState) -> ResearchUpdate {
    let next_phase = determine_next_phase(current);
    ResearchUpdate::transition_to(next_phase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pregel::WorkflowState;
    use crate::research::ResearchDirection;

    #[test]
    fn test_workflow_builder_default() {
        let builder = ResearchWorkflowBuilder::new();

        assert_eq!(builder.max_searches, 6);
        assert_eq!(builder.max_directions, 3);
    }

    #[test]
    fn test_workflow_builder_custom() {
        let builder = ResearchWorkflowBuilder::new()
            .name("custom_research")
            .max_searches(10)
            .max_directions(5)
            .max_explorer_iterations(10);

        assert_eq!(builder.name, "custom_research");
        assert_eq!(builder.max_searches, 10);
        assert_eq!(builder.max_directions, 5);
        assert_eq!(builder.max_explorer_iterations, 10);
    }

    #[test]
    fn test_workflow_builder_build() {
        let result = ResearchWorkflowBuilder::new().build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_research_config_default() {
        let config = ResearchConfig::default();

        assert_eq!(config.max_searches, 6);
        assert_eq!(config.max_directions, 3);
        assert!(!config.parallel_directions);
        assert!(config.timeout_secs.is_none());
    }

    #[test]
    fn test_research_config_builder() {
        let config = ResearchConfig::new()
            .with_max_searches(10)
            .with_max_directions(5)
            .with_parallel_directions(true)
            .with_timeout(300);

        assert_eq!(config.max_searches, 10);
        assert_eq!(config.max_directions, 5);
        assert!(config.parallel_directions);
        assert_eq!(config.timeout_secs, Some(300));
    }

    #[test]
    fn test_can_continue_research_budget() {
        let mut state = ResearchState::new("test").with_max_searches(3);

        // Can continue when under budget
        assert!(can_continue_research(&state));

        // Cannot continue when at budget
        state.search_count = 3;
        state.refresh_can_continue(); // Refresh after direct mutation
        assert!(!can_continue_research(&state));
    }

    #[test]
    fn test_can_continue_research_terminal() {
        let mut state = ResearchState::new("test");

        // Can continue in non-terminal phase
        assert!(can_continue_research(&state));

        // Cannot continue in terminal phase
        state.phase = ResearchPhase::Complete;
        state.refresh_can_continue(); // Refresh after direct mutation
        assert!(!can_continue_research(&state));
    }

    #[test]
    fn test_can_continue_research_directions() {
        let mut state = ResearchState::new("test");
        state.phase = ResearchPhase::Directed;
        state.refresh_can_continue(); // Refresh after direct mutation

        // Cannot continue without directions
        assert!(!can_continue_research(&state));

        // Can continue with unexplored directions
        state.directions.push(ResearchDirection::new("Dir A", "Reason", 5));
        state.refresh_can_continue(); // Refresh after direct mutation
        assert!(can_continue_research(&state));

        // Cannot continue when all explored
        state.directions[0].explored = true;
        state.refresh_can_continue(); // Refresh after direct mutation
        assert!(!can_continue_research(&state));
    }

    #[test]
    fn test_determine_next_phase_exploratory() {
        let mut state = ResearchState::new("test");
        state.phase = ResearchPhase::Exploratory;

        // Without directions, skip to Synthesis
        assert_eq!(determine_next_phase(&state), ResearchPhase::Synthesis);

        // With directions, go to Directed
        state.directions.push(ResearchDirection::new("Dir", "Reason", 5));
        assert_eq!(determine_next_phase(&state), ResearchPhase::Directed);
    }

    #[test]
    fn test_determine_next_phase_directed() {
        let mut state = ResearchState::new("test");
        state.phase = ResearchPhase::Directed;
        state.directions.push(ResearchDirection::new("Dir", "Reason", 5));

        // With unexplored directions, stay in Directed
        assert_eq!(determine_next_phase(&state), ResearchPhase::Directed);

        // With all explored, go to Synthesis
        state.directions[0].explored = true;
        assert_eq!(determine_next_phase(&state), ResearchPhase::Synthesis);
    }

    #[test]
    fn test_determine_next_phase_directed_budget() {
        let mut state = ResearchState::new("test").with_max_searches(2);
        state.phase = ResearchPhase::Directed;
        state.directions.push(ResearchDirection::new("Dir", "Reason", 5));
        state.search_count = 2; // Budget exhausted

        // Even with unexplored directions, go to Synthesis due to budget
        assert_eq!(determine_next_phase(&state), ResearchPhase::Synthesis);
    }

    #[test]
    fn test_determine_next_phase_synthesis() {
        let mut state = ResearchState::new("test");
        state.phase = ResearchPhase::Synthesis;

        assert_eq!(determine_next_phase(&state), ResearchPhase::Complete);
    }

    #[test]
    fn test_phase_transition_update() {
        let mut state = ResearchState::new("test");
        state.phase = ResearchPhase::Exploratory;
        state.directions.push(ResearchDirection::new("Dir", "Reason", 5));

        let update = phase_transition_update(&state);

        assert_eq!(update.phase_transition, Some(ResearchPhase::Directed));
    }

    #[test]
    fn test_workflow_state_trait_impl() {
        // Verify ResearchState implements WorkflowState
        fn requires_workflow_state<S: WorkflowState>() {}
        requires_workflow_state::<ResearchState>();
    }
}
