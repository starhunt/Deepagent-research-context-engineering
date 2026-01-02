//! State isolation for SubAgent execution
//!
//! This module implements the state isolation pattern from Python DeepAgents,
//! where subagents receive filtered state to prevent context contamination.
//!
//! Key concept: When a subagent is invoked, it should NOT see:
//! - Parent's message history (gets fresh HumanMessage with task description)
//! - Parent's todos (gets fresh todo list)
//! - Parent's structured_response
//!
//! But it SHOULD see:
//! - Parent's files (shared filesystem context)
//!
//! Python Reference: deepagents/middleware/subagents.py (_EXCLUDED_STATE_KEYS)

use std::collections::HashMap;

use crate::middleware::StateUpdate;
use crate::state::{AgentState, FileData, Message};

/// Keys excluded when passing state to subagents
///
/// These state fields are NOT passed to subagents:
/// - messages: Subagent gets fresh conversation starting with task description
/// - todos: Subagent gets its own todo list
/// - structured_response: Each agent manages its own structured output
pub const EXCLUDED_STATE_KEYS: &[&str] = &["messages", "todos", "structured_response"];

/// Isolated state for subagent execution
///
/// Contains only the state that should be passed to a subagent,
/// primarily the shared filesystem context.
#[derive(Debug, Clone, Default)]
pub struct IsolatedState {
    /// Files carried over from parent (shared context)
    pub files: HashMap<String, FileData>,
}

impl IsolatedState {
    /// Create an empty isolated state
    pub fn new() -> Self {
        Self::default()
    }

    /// Create isolated state from parent AgentState
    ///
    /// This filters out messages, todos, and structured_response,
    /// keeping only files as shared context.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let parent_state = AgentState::with_messages(vec![Message::user("Hello")]);
    /// let isolated = IsolatedState::from_parent(&parent_state);
    ///
    /// // isolated has parent's files but NOT messages
    /// ```
    pub fn from_parent(parent: &AgentState) -> Self {
        Self {
            files: parent.files.clone(),
        }
    }

    /// Convert to AgentState for subagent execution
    ///
    /// Creates a new AgentState with:
    /// - A single HumanMessage containing the task prompt
    /// - Files from the isolated state
    /// - Empty todos and no structured response
    ///
    /// # Arguments
    ///
    /// * `prompt` - The task description to send to the subagent
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let isolated = IsolatedState::from_parent(&parent_state);
    /// let subagent_state = isolated.to_agent_state("Research quantum computing");
    ///
    /// // subagent_state.messages = [HumanMessage("Research quantum computing")]
    /// ```
    pub fn to_agent_state(self, prompt: &str) -> AgentState {
        let mut state = AgentState::with_messages(vec![Message::user(prompt)]);
        state.files = self.files;
        state
    }

    /// Extract file updates from subagent result state
    ///
    /// After subagent execution, we want to propagate file changes
    /// back to the parent. This extracts the files as a StateUpdate.
    ///
    /// # Arguments
    ///
    /// * `result` - The AgentState after subagent execution
    ///
    /// # Returns
    ///
    /// A StateUpdate::UpdateFiles containing the subagent's file state
    pub fn extract_file_update(result: &AgentState) -> StateUpdate {
        let file_updates: HashMap<String, Option<FileData>> = result
            .files
            .iter()
            .map(|(k, v)| (k.clone(), Some(v.clone())))
            .collect();

        StateUpdate::UpdateFiles(file_updates)
    }

    /// Merge subagent files with parent files
    ///
    /// Applies subagent file changes to parent state.
    /// New/modified files are added, but files are never deleted.
    ///
    /// # Arguments
    ///
    /// * `parent_files` - Parent's current files
    /// * `subagent_files` - Files from subagent execution
    ///
    /// # Returns
    ///
    /// Merged file map with subagent changes applied
    pub fn merge_files(
        parent_files: &HashMap<String, FileData>,
        subagent_files: &HashMap<String, FileData>,
    ) -> HashMap<String, FileData> {
        let mut merged = parent_files.clone();

        for (path, file_data) in subagent_files {
            merged.insert(path.clone(), file_data.clone());
        }

        merged
    }

    /// Check if a state key should be excluded from subagent
    ///
    /// # Arguments
    ///
    /// * `key` - The state key to check
    ///
    /// # Returns
    ///
    /// true if the key should be excluded from subagent state
    pub fn should_exclude(key: &str) -> bool {
        EXCLUDED_STATE_KEYS.contains(&key)
    }
}

/// Builder for creating isolated state with custom configuration
pub struct IsolatedStateBuilder {
    state: IsolatedState,
    include_files: bool,
}

impl IsolatedStateBuilder {
    /// Create a new builder from parent state
    pub fn from_parent(parent: &AgentState) -> Self {
        Self {
            state: IsolatedState {
                files: parent.files.clone(),
            },
            include_files: true,
        }
    }

    /// Exclude files from isolated state
    ///
    /// Use this when you want the subagent to have a completely
    /// fresh context without any inherited files.
    pub fn without_files(mut self) -> Self {
        self.include_files = false;
        self.state.files.clear();
        self
    }

    /// Include only specific files
    ///
    /// Use this to pass only a subset of files to the subagent.
    pub fn with_files(mut self, files: HashMap<String, FileData>) -> Self {
        self.state.files = files;
        self.include_files = true;
        self
    }

    /// Build the isolated state
    pub fn build(self) -> IsolatedState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Todo;

    fn create_test_parent_state() -> AgentState {
        let mut state = AgentState::new();

        // Add messages
        state.messages.push(Message::user("What is quantum computing?"));
        state
            .messages
            .push(Message::assistant("Quantum computing is..."));

        // Add todos
        state.todos.push(Todo::new("Research quantum computers"));

        // Add files
        state.files.insert(
            "/notes.md".to_string(),
            FileData::new("# Research Notes\n\n..."),
        );
        state.files.insert(
            "/report.md".to_string(),
            FileData::new("# Final Report\n\n..."),
        );

        // Add structured response
        state.structured_response = Some(serde_json::json!({"status": "in_progress"}));

        state
    }

    #[test]
    fn test_isolated_state_from_parent() {
        let parent = create_test_parent_state();
        let isolated = IsolatedState::from_parent(&parent);

        // Files should be preserved
        assert_eq!(isolated.files.len(), 2);
        assert!(isolated.files.contains_key("/notes.md"));
        assert!(isolated.files.contains_key("/report.md"));
    }

    #[test]
    fn test_to_agent_state() {
        let parent = create_test_parent_state();
        let isolated = IsolatedState::from_parent(&parent);

        let subagent_state = isolated.to_agent_state("Research quantum entanglement");

        // Should have single user message with prompt
        assert_eq!(subagent_state.messages.len(), 1);
        assert_eq!(
            subagent_state.messages[0].content,
            "Research quantum entanglement"
        );

        // Should have empty todos
        assert!(subagent_state.todos.is_empty());

        // Should have no structured response
        assert!(subagent_state.structured_response.is_none());

        // Should have files from parent
        assert_eq!(subagent_state.files.len(), 2);
    }

    #[test]
    fn test_extract_file_update() {
        let mut result_state = AgentState::new();
        result_state
            .files
            .insert("/new_file.md".to_string(), FileData::new("New content"));

        let update = IsolatedState::extract_file_update(&result_state);

        match update {
            StateUpdate::UpdateFiles(files) => {
                assert_eq!(files.len(), 1);
                assert!(files.contains_key("/new_file.md"));
            }
            _ => panic!("Expected UpdateFiles variant"),
        }
    }

    #[test]
    fn test_merge_files() {
        let mut parent_files = HashMap::new();
        parent_files.insert("/existing.md".to_string(), FileData::new("Existing"));

        let mut subagent_files = HashMap::new();
        subagent_files.insert("/new.md".to_string(), FileData::new("New"));
        subagent_files.insert(
            "/existing.md".to_string(),
            FileData::new("Modified existing"),
        );

        let merged = IsolatedState::merge_files(&parent_files, &subagent_files);

        // Should have both files
        assert_eq!(merged.len(), 2);

        // Existing file should be overwritten
        assert_eq!(
            merged.get("/existing.md").unwrap().as_string(),
            "Modified existing"
        );

        // New file should be added
        assert!(merged.contains_key("/new.md"));
    }

    #[test]
    fn test_should_exclude() {
        assert!(IsolatedState::should_exclude("messages"));
        assert!(IsolatedState::should_exclude("todos"));
        assert!(IsolatedState::should_exclude("structured_response"));
        assert!(!IsolatedState::should_exclude("files"));
        assert!(!IsolatedState::should_exclude("unknown"));
    }

    #[test]
    fn test_builder_without_files() {
        let parent = create_test_parent_state();

        let isolated = IsolatedStateBuilder::from_parent(&parent)
            .without_files()
            .build();

        assert!(isolated.files.is_empty());
    }

    #[test]
    fn test_builder_with_custom_files() {
        let parent = create_test_parent_state();
        let mut custom_files = HashMap::new();
        custom_files.insert("/custom.md".to_string(), FileData::new("Custom content"));

        let isolated = IsolatedStateBuilder::from_parent(&parent)
            .with_files(custom_files)
            .build();

        assert_eq!(isolated.files.len(), 1);
        assert!(isolated.files.contains_key("/custom.md"));
    }

    #[test]
    fn test_empty_isolated_state() {
        let isolated = IsolatedState::new();
        assert!(isolated.files.is_empty());

        let state = isolated.to_agent_state("Test prompt");
        assert_eq!(state.messages.len(), 1);
        assert!(state.files.is_empty());
    }
}
