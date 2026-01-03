//! Research Workflow Module
//!
//! Provides a pre-built three-phase research workflow based on the
//! "breadth-first, then depth" pattern from DeepAgents.
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Research Workflow                         │
//! │                                                              │
//! │  Phase 1: Exploratory                                        │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │ • Broad searches (1-2)                               │    │
//! │  │ • Identify key concepts, players, trends             │    │
//! │  │ • Discover 2-3 promising research directions         │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                          ▼                                   │
//! │  Phase 2: Directed                                           │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │ • Focused searches per direction (1-2 each)          │    │
//! │  │ • Deep dive into promising areas                     │    │
//! │  │ • Collect detailed findings with sources             │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                          ▼                                   │
//! │  Phase 3: Synthesis                                          │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │ • Combine findings into coherent response            │    │
//! │  │ • Analyze source agreement/disagreement              │    │
//! │  │ • Structure final report                             │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rig_deepagents::research::{ResearchState, ResearchPhase, Finding};
//!
//! // Create initial state for a research query
//! let state = ResearchState::new("What are the latest developments in AI safety?")
//!     .with_max_searches(6);
//!
//! // Check research budget
//! if state.can_search() {
//!     println!("Remaining searches: {}", state.remaining_searches());
//! }
//!
//! // Create updates as research progresses
//! let update = ResearchUpdate::with_findings(vec![
//!     Finding::new("Key insight", "Details...", 0.9, ResearchPhase::Exploratory)
//! ]);
//! let new_state = state.apply_update(update);
//! ```
//!
//! # Module Structure
//!
//! - `state` - State and update types for tracking research progress
//! - `prompts` - Pre-built prompt templates for each research phase
//! - `workflow` - Pre-built workflow graph for autonomous research

pub mod prompts;
pub mod state;
pub mod workflow;

// Re-exports for convenience
pub use state::{
    Finding, ResearchDirection, ResearchPhase, ResearchState, ResearchUpdate, Source,
    SourceAgreement,
};
pub use prompts::{PromptBuilder, ResearchPrompts};
pub use workflow::{
    can_continue_research, determine_next_phase, phase_transition_update, ResearchConfig,
    ResearchWorkflowBuilder,
};
