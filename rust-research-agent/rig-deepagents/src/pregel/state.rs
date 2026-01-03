//! Workflow state abstraction for Pregel runtime
//!
//! Defines how workflow state is updated and merged during supersteps.
//! The runtime collects updates from all vertices and applies them atomically
//! at the end of each superstep.

use super::vertex::StateUpdate;

/// Trait for workflow state managed by the Pregel runtime
///
/// The workflow state represents the shared data that vertices can read
/// and update during computation. Updates are collected and merged at the
/// end of each superstep.
///
/// # Example
///
/// ```ignore
/// #[derive(Clone, Default)]
/// struct ResearchState {
///     findings: Vec<Finding>,
///     phase: ResearchPhase,
///     completed_topics: HashSet<String>,
/// }
///
/// impl WorkflowState for ResearchState {
///     type Update = ResearchUpdate;
///
///     fn apply_update(&self, update: Self::Update) -> Self {
///         let mut new = self.clone();
///         new.findings.extend(update.new_findings);
///         new.completed_topics.extend(update.completed);
///         if let Some(phase) = update.phase_transition {
///             new.phase = phase;
///         }
///         new
///     }
///
///     fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
///         ResearchUpdate {
///             new_findings: updates.iter().flat_map(|u| u.new_findings.clone()).collect(),
///             completed: updates.iter().flat_map(|u| u.completed.clone()).collect(),
///             phase_transition: updates.iter().find_map(|u| u.phase_transition.clone()),
///         }
///     }
///
///     fn is_terminal(&self) -> bool {
///         self.phase == ResearchPhase::Complete
///     }
/// }
/// ```
pub trait WorkflowState: Clone + Send + Sync + 'static {
    /// The update type produced by vertices
    type Update: StateUpdate;

    /// Apply an update to produce a new state
    ///
    /// This should be a pure function - the original state is not modified.
    fn apply_update(&self, update: Self::Update) -> Self;

    /// Merge multiple updates into a single update
    ///
    /// Called when multiple vertices produce updates in the same superstep.
    /// The merge should be deterministic (order-independent for correctness).
    fn merge_updates(updates: Vec<Self::Update>) -> Self::Update;

    /// Check if the state represents a terminal condition
    ///
    /// When true, the workflow will terminate regardless of vertex states.
    fn is_terminal(&self) -> bool {
        false
    }

    /// Apply multiple updates in sequence
    ///
    /// Default implementation merges updates then applies the result.
    fn apply_updates(&self, updates: Vec<Self::Update>) -> Self {
        if updates.is_empty() {
            return self.clone();
        }
        let merged = Self::merge_updates(updates);
        self.apply_update(merged)
    }
}

/// A simple unit state for workflows that don't need shared state
///
/// Useful for workflows where all communication is via messages.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UnitState;

/// Unit update that has no effect
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UnitUpdate;

impl StateUpdate for UnitUpdate {
    fn empty() -> Self {
        UnitUpdate
    }

    fn is_empty(&self) -> bool {
        true
    }
}

impl WorkflowState for UnitState {
    type Update = UnitUpdate;

    fn apply_update(&self, _update: Self::Update) -> Self {
        UnitState
    }

    fn merge_updates(_updates: Vec<Self::Update>) -> Self::Update {
        UnitUpdate
    }

    fn is_terminal(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Counter-based state for testing
    #[derive(Clone, Default, Debug, PartialEq)]
    struct CounterState {
        count: i32,
    }

    #[derive(Clone, Debug)]
    struct CounterUpdate {
        delta: i32,
    }

    impl StateUpdate for CounterUpdate {
        fn empty() -> Self {
            CounterUpdate { delta: 0 }
        }

        fn is_empty(&self) -> bool {
            self.delta == 0
        }
    }

    impl WorkflowState for CounterState {
        type Update = CounterUpdate;

        fn apply_update(&self, update: Self::Update) -> Self {
            CounterState {
                count: self.count + update.delta,
            }
        }

        fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
            CounterUpdate {
                delta: updates.iter().map(|u| u.delta).sum(),
            }
        }

        fn is_terminal(&self) -> bool {
            self.count >= 100
        }
    }

    #[test]
    fn test_state_update_merge() {
        let updates = vec![
            CounterUpdate { delta: 5 },
            CounterUpdate { delta: 3 },
            CounterUpdate { delta: -2 },
        ];
        let merged = CounterState::merge_updates(updates);
        assert_eq!(merged.delta, 6);
    }

    #[test]
    fn test_state_apply_update() {
        let state = CounterState { count: 10 };
        let update = CounterUpdate { delta: 5 };
        let new_state = state.apply_update(update);
        assert_eq!(new_state.count, 15);
        // Original state unchanged (immutable)
        assert_eq!(state.count, 10);
    }

    #[test]
    fn test_state_apply_updates() {
        let state = CounterState { count: 0 };
        let updates = vec![
            CounterUpdate { delta: 10 },
            CounterUpdate { delta: 20 },
            CounterUpdate { delta: 5 },
        ];
        let new_state = state.apply_updates(updates);
        assert_eq!(new_state.count, 35);
    }

    #[test]
    fn test_state_terminal_condition() {
        let non_terminal = CounterState { count: 50 };
        assert!(!non_terminal.is_terminal());

        let terminal = CounterState { count: 100 };
        assert!(terminal.is_terminal());

        let over_terminal = CounterState { count: 150 };
        assert!(over_terminal.is_terminal());
    }

    #[test]
    fn test_empty_updates() {
        let state = CounterState { count: 42 };
        let new_state = state.apply_updates(vec![]);
        assert_eq!(new_state.count, 42);
    }

    // More complex state for testing
    #[derive(Clone, Default, Debug)]
    struct CollectionState {
        items: Vec<String>,
        seen: HashSet<String>,
    }

    #[derive(Clone, Debug)]
    struct CollectionUpdate {
        new_items: Vec<String>,
    }

    impl StateUpdate for CollectionUpdate {
        fn empty() -> Self {
            CollectionUpdate { new_items: vec![] }
        }

        fn is_empty(&self) -> bool {
            self.new_items.is_empty()
        }
    }

    impl WorkflowState for CollectionState {
        type Update = CollectionUpdate;

        fn apply_update(&self, update: Self::Update) -> Self {
            let mut items = self.items.clone();
            let mut seen = self.seen.clone();

            for item in update.new_items {
                if !seen.contains(&item) {
                    seen.insert(item.clone());
                    items.push(item);
                }
            }

            CollectionState { items, seen }
        }

        fn merge_updates(updates: Vec<Self::Update>) -> Self::Update {
            CollectionUpdate {
                new_items: updates.into_iter().flat_map(|u| u.new_items).collect(),
            }
        }
    }

    #[test]
    fn test_collection_state_dedup() {
        let state = CollectionState::default();
        let updates = vec![
            CollectionUpdate {
                new_items: vec!["a".to_string(), "b".to_string()],
            },
            CollectionUpdate {
                new_items: vec!["b".to_string(), "c".to_string()],
            },
        ];

        let new_state = state.apply_updates(updates);
        // "b" should only appear once due to dedup in apply_update
        assert_eq!(new_state.items.len(), 3);
        assert_eq!(new_state.seen.len(), 3);
    }

    #[test]
    fn test_unit_state() {
        let state = UnitState;
        let update = UnitUpdate;

        assert!(update.is_empty());
        assert!(UnitUpdate::empty().is_empty());

        let new_state = state.apply_update(update);
        assert!(!new_state.is_terminal());

        let merged = UnitState::merge_updates(vec![UnitUpdate, UnitUpdate]);
        assert!(merged.is_empty());
    }
}
