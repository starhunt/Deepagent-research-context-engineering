//! Research workflow state definition
//!
//! Implements `WorkflowState` trait for the three-phase research pattern:
//! 1. Exploratory (breadth-first discovery)
//! 2. Directed (deep dives into promising directions)
//! 3. Synthesis (combining findings into a coherent response)
//!
//! Python Reference: research_agent/researcher/prompts.py

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::pregel::state::WorkflowState;
use crate::pregel::vertex::StateUpdate;

/// Research workflow phases following the "breadth-first, then depth" pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ResearchPhase {
    /// Initial phase: broad exploration to understand the landscape
    #[default]
    Exploratory,
    /// Deep dive into promising directions identified in exploration
    Directed,
    /// Combining findings into a coherent response
    Synthesis,
    /// Research is complete
    Complete,
}

impl ResearchPhase {
    /// Get the next phase in the workflow
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Exploratory => Some(Self::Directed),
            Self::Directed => Some(Self::Synthesis),
            Self::Synthesis => Some(Self::Complete),
            Self::Complete => None,
        }
    }

    /// Check if this is a terminal phase
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete)
    }
}

/// A research direction identified during exploration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResearchDirection {
    /// Name/label for the direction
    pub name: String,
    /// Reason this direction was identified as promising
    pub reason: String,
    /// Priority (higher = more important)
    pub priority: u8,
    /// Whether this direction has been explored
    pub explored: bool,
}

impl ResearchDirection {
    /// Create a new research direction
    pub fn new(name: impl Into<String>, reason: impl Into<String>, priority: u8) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
            priority,
            explored: false,
        }
    }
}

/// A source used during research
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// Source URL
    pub url: String,
    /// Source title
    pub title: String,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
    /// Optional snippet/summary from the source
    pub snippet: Option<String>,
}

impl Source {
    /// Create a new source
    pub fn new(url: impl Into<String>, title: impl Into<String>, relevance: f32) -> Self {
        Self {
            url: url.into(),
            title: title.into(),
            relevance: relevance.clamp(0.0, 1.0),
            snippet: None,
        }
    }

    /// Add a snippet to the source
    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into());
        self
    }
}

/// A research finding with supporting sources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// Title of the finding
    pub title: String,
    /// Detailed description/content
    pub content: String,
    /// Indices of supporting sources
    pub source_indices: Vec<usize>,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Phase when this finding was discovered
    pub phase: ResearchPhase,
    /// Direction this finding belongs to (for directed phase)
    pub direction: Option<String>,
}

impl Finding {
    /// Create a new finding
    pub fn new(
        title: impl Into<String>,
        content: impl Into<String>,
        confidence: f32,
        phase: ResearchPhase,
    ) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            source_indices: vec![],
            confidence: confidence.clamp(0.0, 1.0),
            phase,
            direction: None,
        }
    }

    /// Add source references
    pub fn with_sources(mut self, indices: Vec<usize>) -> Self {
        self.source_indices = indices;
        self
    }

    /// Associate with a research direction
    pub fn with_direction(mut self, direction: impl Into<String>) -> Self {
        self.direction = Some(direction.into());
        self
    }
}

/// Source agreement analysis result
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceAgreement {
    /// Topics where sources strongly agree
    pub high_agreement: Vec<String>,
    /// Topics with conflicting or uncertain information
    pub disagreement: Vec<String>,
}

/// The complete research workflow state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResearchState {
    /// Original research query/topic
    pub query: String,

    /// Current phase of research
    pub phase: ResearchPhase,

    /// Research directions identified (Phase 1 output)
    pub directions: Vec<ResearchDirection>,

    /// Collected sources
    pub sources: Vec<Source>,

    /// Research findings organized by discovery
    pub findings: Vec<Finding>,

    /// Source agreement analysis (Phase 3 output)
    pub agreement: SourceAgreement,

    /// Search count (for enforcing limits)
    pub search_count: usize,

    /// Maximum allowed searches (default: 6)
    pub max_searches: usize,

    /// Queries that have been executed (for deduplication)
    pub executed_queries: HashSet<String>,

    /// Any errors encountered during research
    pub errors: Vec<String>,

    /// Whether research can continue (computed field for router decisions)
    /// This is automatically updated after each state update.
    #[serde(default = "default_can_continue")]
    pub can_continue: bool,
}

/// Default value for can_continue - new states start as continuable
fn default_can_continue() -> bool {
    true
}

impl ResearchState {
    /// Create a new research state for a query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            phase: ResearchPhase::Exploratory,
            max_searches: 6,
            can_continue: true, // New states can always continue
            ..Default::default()
        }
    }

    /// Refreshes the `can_continue` computed field based on current state.
    /// Call this after directly mutating state fields (outside of `apply_update`).
    ///
    /// In normal workflow operation, `can_continue` is automatically updated
    /// via `apply_update()`. This method is primarily for testing or manual
    /// state manipulation.
    pub fn refresh_can_continue(&mut self) {
        self.can_continue = self.compute_can_continue();
    }

    /// Compute whether research can continue based on current state.
    /// This checks: budget availability, terminal phase, and unexplored directions.
    fn compute_can_continue(&self) -> bool {
        // Check if we've exceeded search budget
        if self.search_count >= self.max_searches {
            return false;
        }

        // Check if we're in a terminal phase
        if self.phase.is_terminal() {
            return false;
        }

        // Check if all directions have been explored in Directed phase
        if self.phase == ResearchPhase::Directed && self.unexplored_directions().is_empty() {
            return false;
        }

        true
    }

    /// Configure the maximum number of searches
    pub fn with_max_searches(mut self, max: usize) -> Self {
        self.max_searches = max;
        self
    }

    /// Check if more searches are allowed
    pub fn can_search(&self) -> bool {
        self.search_count < self.max_searches
    }

    /// Get remaining search budget
    pub fn remaining_searches(&self) -> usize {
        self.max_searches.saturating_sub(self.search_count)
    }

    /// Check if a query has already been executed
    pub fn has_executed_query(&self, query: &str) -> bool {
        self.executed_queries.contains(query)
    }

    /// Get unexplored directions sorted by priority
    pub fn unexplored_directions(&self) -> Vec<&ResearchDirection> {
        let mut dirs: Vec<_> = self.directions.iter().filter(|d| !d.explored).collect();
        dirs.sort_by(|a, b| b.priority.cmp(&a.priority));
        dirs
    }

    /// Get findings for a specific direction
    pub fn findings_for_direction(&self, direction: &str) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| f.direction.as_deref() == Some(direction))
            .collect()
    }

    /// Get findings from the exploratory phase
    pub fn exploratory_findings(&self) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| f.phase == ResearchPhase::Exploratory)
            .collect()
    }

    /// Generate a formatted source list for citations
    pub fn format_sources(&self) -> String {
        self.sources
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {}: {}", i + 1, s.title, s.url))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Update to the research state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResearchUpdate {
    /// New findings to add
    pub new_findings: Vec<Finding>,

    /// New sources to add
    pub new_sources: Vec<Source>,

    /// New directions identified
    pub new_directions: Vec<ResearchDirection>,

    /// Directions that have been explored (mark as complete)
    pub explored_directions: HashSet<String>,

    /// Queries that were executed
    pub executed_queries: HashSet<String>,

    /// Number of searches performed
    pub searches_performed: usize,

    /// Phase transition (if any)
    pub phase_transition: Option<ResearchPhase>,

    /// Source agreement update
    pub agreement_update: Option<SourceAgreement>,

    /// Errors encountered
    pub errors: Vec<String>,
}

impl ResearchUpdate {
    /// Create an update with new findings
    pub fn with_findings(findings: Vec<Finding>) -> Self {
        Self {
            new_findings: findings,
            ..Default::default()
        }
    }

    /// Create an update for a phase transition
    pub fn transition_to(phase: ResearchPhase) -> Self {
        Self {
            phase_transition: Some(phase),
            ..Default::default()
        }
    }

    /// Add a search execution record
    pub fn with_search(mut self, query: impl Into<String>) -> Self {
        self.executed_queries.insert(query.into());
        self.searches_performed += 1;
        self
    }

    /// Add new directions
    pub fn with_directions(mut self, directions: Vec<ResearchDirection>) -> Self {
        self.new_directions = directions;
        self
    }

    /// Mark directions as explored
    pub fn with_explored(mut self, directions: impl IntoIterator<Item = String>) -> Self {
        self.explored_directions.extend(directions);
        self
    }

    /// Add new sources
    pub fn with_sources(mut self, sources: Vec<Source>) -> Self {
        self.new_sources = sources;
        self
    }

    /// Set agreement analysis
    pub fn with_agreement(mut self, agreement: SourceAgreement) -> Self {
        self.agreement_update = Some(agreement);
        self
    }

    /// Add an error
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.errors.push(error.into());
        self
    }
}

impl StateUpdate for ResearchUpdate {
    fn empty() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.new_findings.is_empty()
            && self.new_sources.is_empty()
            && self.new_directions.is_empty()
            && self.explored_directions.is_empty()
            && self.executed_queries.is_empty()
            && self.searches_performed == 0
            && self.phase_transition.is_none()
            && self.agreement_update.is_none()
            && self.errors.is_empty()
    }
}

impl WorkflowState for ResearchState {
    type Update = ResearchUpdate;

    fn apply_update(&self, update: Self::Update) -> Self {
        let mut new_state = self.clone();

        // Add new findings
        new_state.findings.extend(update.new_findings);

        // Add new sources (dedup by URL)
        for source in update.new_sources {
            if !new_state.sources.iter().any(|s| s.url == source.url) {
                new_state.sources.push(source);
            }
        }

        // Add new directions (dedup by name)
        for direction in update.new_directions {
            if !new_state
                .directions
                .iter()
                .any(|d| d.name == direction.name)
            {
                new_state.directions.push(direction);
            }
        }

        // Mark explored directions
        for dir_name in &update.explored_directions {
            if let Some(dir) = new_state
                .directions
                .iter_mut()
                .find(|d| &d.name == dir_name)
            {
                dir.explored = true;
            }
        }

        // Record executed queries
        new_state.executed_queries.extend(update.executed_queries);

        // Update search count
        new_state.search_count += update.searches_performed;

        // Apply phase transition
        if let Some(new_phase) = update.phase_transition {
            new_state.phase = new_phase;
        }

        // Update agreement analysis
        if let Some(agreement) = update.agreement_update {
            new_state.agreement = agreement;
        }

        // Collect errors
        new_state.errors.extend(update.errors);

        // Recompute can_continue based on new state
        new_state.can_continue = new_state.compute_can_continue();

        new_state
    }

    fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
        let mut merged = ResearchUpdate::default();

        for update in updates {
            merged.new_findings.extend(update.new_findings);
            merged.new_sources.extend(update.new_sources);
            merged.new_directions.extend(update.new_directions);
            merged.explored_directions.extend(update.explored_directions);
            merged.executed_queries.extend(update.executed_queries);
            merged.searches_performed += update.searches_performed;
            merged.errors.extend(update.errors);

            // Last phase transition wins
            if update.phase_transition.is_some() {
                merged.phase_transition = update.phase_transition;
            }

            // Last agreement update wins
            if update.agreement_update.is_some() {
                merged.agreement_update = update.agreement_update;
            }
        }

        merged
    }

    fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_phase_progression() {
        assert_eq!(ResearchPhase::Exploratory.next(), Some(ResearchPhase::Directed));
        assert_eq!(ResearchPhase::Directed.next(), Some(ResearchPhase::Synthesis));
        assert_eq!(ResearchPhase::Synthesis.next(), Some(ResearchPhase::Complete));
        assert_eq!(ResearchPhase::Complete.next(), None);

        assert!(!ResearchPhase::Exploratory.is_terminal());
        assert!(!ResearchPhase::Directed.is_terminal());
        assert!(!ResearchPhase::Synthesis.is_terminal());
        assert!(ResearchPhase::Complete.is_terminal());
    }

    #[test]
    fn test_research_state_new() {
        let state = ResearchState::new("test query");

        assert_eq!(state.query, "test query");
        assert_eq!(state.phase, ResearchPhase::Exploratory);
        assert_eq!(state.max_searches, 6);
        assert_eq!(state.search_count, 0);
        assert!(state.findings.is_empty());
        assert!(state.sources.is_empty());
    }

    #[test]
    fn test_research_state_search_budget() {
        let mut state = ResearchState::new("test").with_max_searches(3);

        assert!(state.can_search());
        assert_eq!(state.remaining_searches(), 3);

        state.search_count = 2;
        assert!(state.can_search());
        assert_eq!(state.remaining_searches(), 1);

        state.search_count = 3;
        assert!(!state.can_search());
        assert_eq!(state.remaining_searches(), 0);
    }

    #[test]
    fn test_research_direction() {
        let dir = ResearchDirection::new("AI Safety", "Important emerging field", 5);

        assert_eq!(dir.name, "AI Safety");
        assert!(!dir.explored);
        assert_eq!(dir.priority, 5);
    }

    #[test]
    fn test_source_creation() {
        let source = Source::new("https://example.com", "Example", 0.95)
            .with_snippet("Relevant excerpt...");

        assert_eq!(source.url, "https://example.com");
        assert_eq!(source.relevance, 0.95);
        assert!(source.snippet.is_some());
    }

    #[test]
    fn test_source_relevance_clamping() {
        let high = Source::new("url", "title", 1.5);
        assert_eq!(high.relevance, 1.0);

        let low = Source::new("url", "title", -0.5);
        assert_eq!(low.relevance, 0.0);
    }

    #[test]
    fn test_finding_creation() {
        let finding = Finding::new(
            "Test Finding",
            "Detailed content here",
            0.85,
            ResearchPhase::Exploratory,
        )
        .with_sources(vec![0, 1])
        .with_direction("AI Safety");

        assert_eq!(finding.title, "Test Finding");
        assert_eq!(finding.confidence, 0.85);
        assert_eq!(finding.source_indices, vec![0, 1]);
        assert_eq!(finding.direction, Some("AI Safety".to_string()));
    }

    #[test]
    fn test_research_update_empty() {
        let update = ResearchUpdate::empty();
        assert!(update.is_empty());

        let with_finding = ResearchUpdate::with_findings(vec![Finding::new(
            "Test",
            "Content",
            0.5,
            ResearchPhase::Exploratory,
        )]);
        assert!(!with_finding.is_empty());
    }

    #[test]
    fn test_research_state_apply_update() {
        let state = ResearchState::new("test query");

        let update = ResearchUpdate {
            new_findings: vec![Finding::new(
                "Finding 1",
                "Content",
                0.9,
                ResearchPhase::Exploratory,
            )],
            new_sources: vec![Source::new("https://a.com", "Source A", 0.8)],
            new_directions: vec![ResearchDirection::new("Dir A", "Reason", 3)],
            searches_performed: 1,
            executed_queries: ["query 1".to_string()].into_iter().collect(),
            phase_transition: Some(ResearchPhase::Directed),
            ..Default::default()
        };

        let new_state = state.apply_update(update);

        assert_eq!(new_state.findings.len(), 1);
        assert_eq!(new_state.sources.len(), 1);
        assert_eq!(new_state.directions.len(), 1);
        assert_eq!(new_state.search_count, 1);
        assert!(new_state.has_executed_query("query 1"));
        assert_eq!(new_state.phase, ResearchPhase::Directed);
    }

    #[test]
    fn test_research_state_source_dedup() {
        let state = ResearchState::new("test");

        let update1 = ResearchUpdate {
            new_sources: vec![Source::new("https://a.com", "A", 0.8)],
            ..Default::default()
        };

        let update2 = ResearchUpdate {
            new_sources: vec![
                Source::new("https://a.com", "A duplicate", 0.9), // Same URL
                Source::new("https://b.com", "B", 0.7),
            ],
            ..Default::default()
        };

        let state = state.apply_update(update1).apply_update(update2);

        assert_eq!(state.sources.len(), 2); // Deduped by URL
        assert_eq!(state.sources[0].title, "A"); // Original kept
    }

    #[test]
    fn test_research_state_merge_updates() {
        let updates = vec![
            ResearchUpdate {
                new_findings: vec![Finding::new("F1", "C1", 0.8, ResearchPhase::Exploratory)],
                searches_performed: 1,
                ..Default::default()
            },
            ResearchUpdate {
                new_findings: vec![Finding::new("F2", "C2", 0.7, ResearchPhase::Directed)],
                searches_performed: 2,
                phase_transition: Some(ResearchPhase::Directed),
                ..Default::default()
            },
            ResearchUpdate {
                phase_transition: Some(ResearchPhase::Synthesis), // This one wins
                ..Default::default()
            },
        ];

        let merged = ResearchState::merge_updates(updates);

        assert_eq!(merged.new_findings.len(), 2);
        assert_eq!(merged.searches_performed, 3);
        assert_eq!(merged.phase_transition, Some(ResearchPhase::Synthesis));
    }

    #[test]
    fn test_research_state_terminal() {
        let mut state = ResearchState::new("test");
        assert!(!state.is_terminal());

        state.phase = ResearchPhase::Complete;
        assert!(state.is_terminal());
    }

    #[test]
    fn test_unexplored_directions() {
        let mut state = ResearchState::new("test");
        state.directions = vec![
            ResearchDirection {
                name: "Low".to_string(),
                reason: "R".to_string(),
                priority: 1,
                explored: false,
            },
            ResearchDirection {
                name: "High".to_string(),
                reason: "R".to_string(),
                priority: 5,
                explored: false,
            },
            ResearchDirection {
                name: "Done".to_string(),
                reason: "R".to_string(),
                priority: 10,
                explored: true,
            },
        ];

        let unexplored = state.unexplored_directions();
        assert_eq!(unexplored.len(), 2);
        assert_eq!(unexplored[0].name, "High"); // Sorted by priority desc
        assert_eq!(unexplored[1].name, "Low");
    }

    #[test]
    fn test_findings_by_direction() {
        let mut state = ResearchState::new("test");
        state.findings = vec![
            Finding::new("F1", "C1", 0.8, ResearchPhase::Directed)
                .with_direction("AI Safety"),
            Finding::new("F2", "C2", 0.7, ResearchPhase::Directed)
                .with_direction("Ethics"),
            Finding::new("F3", "C3", 0.9, ResearchPhase::Directed)
                .with_direction("AI Safety"),
        ];

        let ai_findings = state.findings_for_direction("AI Safety");
        assert_eq!(ai_findings.len(), 2);
        assert!(ai_findings.iter().all(|f| f.direction == Some("AI Safety".to_string())));
    }

    #[test]
    fn test_format_sources() {
        let mut state = ResearchState::new("test");
        state.sources = vec![
            Source::new("https://a.com", "Source A", 0.9),
            Source::new("https://b.com", "Source B", 0.8),
        ];

        let formatted = state.format_sources();
        assert!(formatted.contains("[1] Source A: https://a.com"));
        assert!(formatted.contains("[2] Source B: https://b.com"));
    }
}
