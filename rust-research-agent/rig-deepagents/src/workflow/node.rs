//! Node Types and Configuration for Workflow Graphs
//!
//! Defines the different types of nodes that can exist in a workflow graph.
//! Each node type has specific behavior and configuration options.
//!
//! # Node Types
//!
//! - **Agent**: LLM-based processing with tool calling capabilities
//! - **Tool**: Single tool execution with static or dynamic arguments
//! - **Router**: Conditional branching based on state or LLM decisions
//! - **SubAgent**: Delegation to nested workflows with recursion protection
//! - **FanOut**: Parallel dispatch to multiple targets
//! - **FanIn**: Synchronization point waiting for multiple sources
//! - **Passthrough**: Simple data forwarding (identity transformation)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// The kind of node in a workflow graph.
///
/// Each variant represents a different computation pattern.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeKind {
    /// An LLM-based agent that can process messages and call tools
    Agent(AgentNodeConfig),

    /// A single tool execution node
    Tool(ToolNodeConfig),

    /// Conditional routing based on state or LLM decisions
    Router(RouterNodeConfig),

    /// Delegation to a sub-workflow
    SubAgent(SubAgentNodeConfig),

    /// Parallel dispatch to multiple targets
    FanOut(FanOutNodeConfig),

    /// Synchronization point waiting for multiple sources
    FanIn(FanInNodeConfig),

    /// Simple passthrough (identity transformation)
    #[default]
    Passthrough,
}

/// Configuration for an Agent node.
///
/// Agents use LLMs to process messages and can optionally call tools.
/// They iterate until a stop condition is met.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNodeConfig {
    /// System prompt for the agent
    pub system_prompt: String,

    /// Maximum iterations before forcing termination
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// Conditions that cause the agent to stop iterating
    #[serde(default)]
    pub stop_conditions: Vec<StopCondition>,

    /// Tools the agent is allowed to use (None = all tools)
    #[serde(default)]
    pub allowed_tools: Option<HashSet<String>>,

    /// Timeout for each LLM call
    #[serde(default, with = "humantime_serde")]
    pub llm_timeout: Option<Duration>,

    /// Temperature for LLM calls
    #[serde(default)]
    pub temperature: Option<f32>,
}

impl Default for AgentNodeConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            max_iterations: 10,
            stop_conditions: vec![StopCondition::NoToolCalls],
            allowed_tools: None,
            llm_timeout: None,
            temperature: None,
        }
    }
}

fn default_max_iterations() -> usize {
    10
}

/// Conditions that cause an agent to stop iterating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StopCondition {
    /// Stop when the LLM produces no tool calls
    NoToolCalls,

    /// Stop when a specific tool is called
    OnTool { tool_name: String },

    /// Stop when the message contains specific text
    ContainsText { pattern: String },

    /// Stop when a state field matches a condition
    StateMatch { field: String, value: serde_json::Value },

    /// Stop after a certain number of iterations
    MaxIterations { count: usize },
}

/// Configuration for a Tool node.
///
/// Executes a single tool with arguments from static config or state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolNodeConfig {
    /// Name of the tool to execute
    #[serde(default)]
    pub tool_name: String,

    /// Static arguments (can be overridden by state paths)
    #[serde(default)]
    pub static_args: HashMap<String, serde_json::Value>,

    /// Map from argument name to state path for dynamic arguments
    #[serde(default)]
    pub state_arg_paths: HashMap<String, String>,

    /// Path in state where the result should be stored
    #[serde(default)]
    pub result_path: Option<String>,

    /// Timeout for tool execution
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

/// Configuration for a Router node.
///
/// Determines next node based on state inspection or LLM decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterNodeConfig {
    /// How to make the routing decision
    pub strategy: RoutingStrategy,

    /// Branches to evaluate (in order for StateField strategy)
    pub branches: Vec<Branch>,

    /// Default branch if no conditions match (required for StateField)
    #[serde(default)]
    pub default: Option<String>,
}

impl Default for RouterNodeConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::StateField {
                field: String::new(),
            },
            branches: Vec::new(),
            default: None,
        }
    }
}

/// Strategy for making routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// Route based on a state field value
    StateField {
        /// Path to the state field to inspect
        field: String,
    },

    /// Route based on LLM classification
    LLMDecision {
        /// Prompt describing the options to the LLM
        prompt: String,
        /// Model to use (optional, uses default if not specified)
        model: Option<String>,
    },
}

/// A branch in a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Target node for this branch
    pub target: String,

    /// Condition that must be true for this branch
    pub condition: BranchCondition,
}

/// Condition for a routing branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum BranchCondition {
    /// Value equals expected
    Equals { value: serde_json::Value },

    /// Value is in set of options
    In { values: Vec<serde_json::Value> },

    /// Value matches regex pattern
    Matches { pattern: String },

    /// Value is truthy (non-null, non-empty, non-false)
    IsTruthy,

    /// Value is falsy
    IsFalsy,

    /// Always true (used for catch-all branches)
    Always,
}

/// Configuration for a SubAgent node.
///
/// Delegates work to a nested workflow with recursion protection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentNodeConfig {
    /// Name of the sub-agent to invoke
    pub agent_name: String,

    /// Maximum recursion depth (prevents infinite nesting)
    #[serde(default = "default_max_recursion")]
    pub max_recursion: usize,

    /// Mapping from parent state paths to sub-agent input
    #[serde(default)]
    pub input_mapping: HashMap<String, String>,

    /// Mapping from sub-agent output to parent state paths
    #[serde(default)]
    pub output_mapping: HashMap<String, String>,

    /// Timeout for the entire sub-agent execution
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

impl Default for SubAgentNodeConfig {
    fn default() -> Self {
        Self {
            agent_name: String::new(),
            max_recursion: 5,
            input_mapping: HashMap::new(),
            output_mapping: HashMap::new(),
            timeout: None,
        }
    }
}

fn default_max_recursion() -> usize {
    5
}

/// Configuration for a FanOut node.
///
/// Broadcasts messages to multiple targets in parallel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanOutNodeConfig {
    /// Target nodes to send to
    pub targets: Vec<String>,

    /// How to split the work among targets
    #[serde(default)]
    pub split_strategy: SplitStrategy,

    /// Path to array in state to split (for Split strategy)
    #[serde(default)]
    pub split_path: Option<String>,
}

impl Default for FanOutNodeConfig {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            split_strategy: SplitStrategy::Broadcast,
            split_path: None,
        }
    }
}

/// Strategy for splitting work in a FanOut node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitStrategy {
    /// Send the same message to all targets
    #[default]
    Broadcast,

    /// Split an array and send one element to each target
    Split,

    /// Round-robin distribution
    RoundRobin,
}

/// Configuration for a FanIn node.
///
/// Waits for messages from multiple sources and merges them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanInNodeConfig {
    /// Source nodes to wait for
    pub sources: Vec<String>,

    /// How to merge results from sources
    #[serde(default)]
    pub merge_strategy: MergeStrategy,

    /// Path in state where merged results are stored
    #[serde(default)]
    pub result_path: Option<String>,

    /// Timeout for waiting for all sources
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

impl Default for FanInNodeConfig {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            merge_strategy: MergeStrategy::Collect,
            result_path: None,
            timeout: None,
        }
    }
}

/// Strategy for merging results in a FanIn node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Collect all results into an array
    #[default]
    Collect,

    /// Use first result that arrives
    First,

    /// Use last result (all must complete)
    Last,

    /// Concatenate string results
    Concat,

    /// Merge object results (later values overwrite)
    Merge,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_kind_serialization() {
        let agent = NodeKind::Agent(AgentNodeConfig {
            system_prompt: "You are helpful.".into(),
            ..Default::default()
        });

        let json = serde_json::to_string(&agent).unwrap();
        let deserialized: NodeKind = serde_json::from_str(&json).unwrap();

        match deserialized {
            NodeKind::Agent(config) => {
                assert_eq!(config.system_prompt, "You are helpful.");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tool_node_config() {
        let tool = ToolNodeConfig {
            tool_name: "search".into(),
            static_args: [("query".into(), serde_json::json!("test"))].into(),
            state_arg_paths: [("max_results".into(), "config.limit".into())].into(),
            result_path: Some("search_results".into()),
            timeout: Some(Duration::from_secs(30)),
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("search"));
        assert!(json.contains("query"));
    }

    #[test]
    fn test_router_config_with_branches() {
        let router = RouterNodeConfig {
            strategy: RoutingStrategy::StateField {
                field: "phase".into(),
            },
            branches: vec![
                Branch {
                    target: "explore".into(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("exploratory"),
                    },
                },
                Branch {
                    target: "synthesize".into(),
                    condition: BranchCondition::Equals {
                        value: serde_json::json!("synthesis"),
                    },
                },
            ],
            default: Some("done".into()),
        };

        let json = serde_json::to_string(&router).unwrap();
        let deserialized: RouterNodeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.branches.len(), 2);
        assert_eq!(deserialized.default, Some("done".into()));
    }

    #[test]
    fn test_stop_conditions() {
        let conditions = vec![
            StopCondition::NoToolCalls,
            StopCondition::OnTool {
                tool_name: "submit".into(),
            },
            StopCondition::ContainsText {
                pattern: "DONE".into(),
            },
            StopCondition::MaxIterations { count: 5 },
        ];

        let json = serde_json::to_string(&conditions).unwrap();
        let deserialized: Vec<StopCondition> = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.len(), 4);
        assert_eq!(deserialized[0], StopCondition::NoToolCalls);
    }

    #[test]
    fn test_subagent_config() {
        let config = SubAgentNodeConfig {
            agent_name: "researcher".into(),
            max_recursion: 3,
            input_mapping: [("query".into(), "research.topic".into())].into(),
            output_mapping: [("findings".into(), "research.findings".into())].into(),
            timeout: Some(Duration::from_secs(300)),
        };

        assert_eq!(config.agent_name, "researcher");
        assert_eq!(config.max_recursion, 3);
    }

    #[test]
    fn test_fanout_fanin_config() {
        let fanout = FanOutNodeConfig {
            targets: vec!["a".into(), "b".into(), "c".into()],
            split_strategy: SplitStrategy::Broadcast,
            split_path: None,
        };

        let fanin = FanInNodeConfig {
            sources: vec!["a".into(), "b".into(), "c".into()],
            merge_strategy: MergeStrategy::Collect,
            result_path: Some("results".into()),
            timeout: Some(Duration::from_secs(60)),
        };

        assert_eq!(fanout.targets, fanin.sources);
    }

    #[test]
    fn test_branch_conditions() {
        let conditions = vec![
            BranchCondition::Equals {
                value: serde_json::json!("active"),
            },
            BranchCondition::In {
                values: vec![serde_json::json!(1), serde_json::json!(2)],
            },
            BranchCondition::Matches {
                pattern: "^done.*".into(),
            },
            BranchCondition::IsTruthy,
            BranchCondition::Always,
        ];

        for condition in &conditions {
            let json = serde_json::to_string(condition).unwrap();
            let _: BranchCondition = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_node_kind_variants() {
        // Ensure all 7 variants can be created
        let _agent = NodeKind::Agent(Default::default());
        let _tool = NodeKind::Tool(Default::default());
        let _router = NodeKind::Router(Default::default());
        let _subagent = NodeKind::SubAgent(Default::default());
        let _fanout = NodeKind::FanOut(Default::default());
        let _fanin = NodeKind::FanIn(Default::default());
        let _passthrough = NodeKind::Passthrough;

        // Ensure default is Passthrough
        assert!(matches!(NodeKind::default(), NodeKind::Passthrough));
    }
}
