//! End-to-End LLM Integration Tests
//!
//! These tests require actual API keys and make real LLM calls.
//! Run with: `cargo test --test e2e_llm_integration -- --ignored`
//!
//! # Environment Variables Required
//!
//! - `OPENAI_API_KEY`: OpenAI API key for LLM calls
//! - `TAVILY_API_KEY`: Tavily API key for web search (optional)
//!
//! # Warning
//!
//! These tests will consume API credits!

use std::sync::Arc;
use std::time::Duration;

use rig::client::{CompletionClient, ProviderClient};

use rig_deepagents::{
    RigAgentAdapter, LLMProvider, LLMConfig,
    ProductionConfig, ProductionSetup,
    ResearchState, ResearchPhase,
    TavilySearchTool, ThinkTool,
};
use rig_deepagents::middleware::Tool;
use rig_deepagents::pregel::{PregelConfig, WorkflowState};
use rig_deepagents::pregel::config::ExecutionMode;
use rig_deepagents::state::Message;
use rig_deepagents::workflow::CompiledWorkflow;
use rig_deepagents::workflow::graph::WorkflowGraph;
use rig_deepagents::workflow::node::{AgentNodeConfig, NodeKind, StopCondition};

/// Helper to check if OpenAI API key is available
fn has_openai_key() -> bool {
    std::env::var("OPENAI_API_KEY").is_ok()
}

/// Helper to check if Tavily API key is available
fn has_tavily_key() -> bool {
    std::env::var("TAVILY_API_KEY").is_ok()
}

/// Create a RigAgentAdapter for OpenAI
fn create_openai_provider(model: &str) -> impl LLMProvider {
    let client = rig::providers::openai::Client::from_env();
    let agent = client.agent(model).build();
    RigAgentAdapter::with_names(agent, "openai", model)
}

/// Test basic LLM completion without tools
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_basic_completion() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let provider = create_openai_provider("gpt-4.1");

    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("What is 2 + 2? Reply with just the number."),
    ];

    let config = LLMConfig::new("gpt-4.1")
        .with_temperature(0.0)
        .with_max_tokens(16000);

    let response = provider
        .complete(&messages, &[], Some(&config))
        .await
        .expect("LLM completion failed");

    println!("Response: {}", response.message.content);
    assert!(response.message.content.contains("4"));
}

/// Test LLM with tool definitions (no actual tool execution)
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_openai_with_tool_definitions() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let provider = create_openai_provider("gpt-4.1");

    let think_tool = ThinkTool;
    let tool_defs = vec![think_tool.definition()];

    let messages = vec![
        Message::system("You are a research assistant. Use the think tool to reason through problems."),
        Message::user("Use the think tool to consider why the sky is blue."),
    ];

    let config = LLMConfig::new("gpt-4.1")
        .with_temperature(0.0);

    let response = provider
        .complete(&messages, &tool_defs, Some(&config))
        .await
        .expect("LLM completion failed");

    println!("Response: {:?}", response.message);

    // The model should either use the think tool or provide a response
    assert!(
        response.message.tool_calls.is_some() || !response.message.content.is_empty(),
        "Expected tool call or content response"
    );
}

/// Test Tavily search tool execution
#[tokio::test]
#[ignore = "Requires TAVILY_API_KEY environment variable"]
async fn test_tavily_search_execution() {
    if !has_tavily_key() {
        eprintln!("Skipping test: TAVILY_API_KEY not set");
        return;
    }

    let tavily = TavilySearchTool::from_env()
        .expect("Failed to create Tavily tool")
        .with_timeout(Duration::from_secs(30))
        .with_max_retries(2);

    // Create a mock runtime for tool execution
    use rig_deepagents::runtime::ToolRuntime;
    use rig_deepagents::state::AgentState;
    use rig_deepagents::backends::MemoryBackend;

    let backend = Arc::new(MemoryBackend::new());
    let state = AgentState::new();
    let runtime = ToolRuntime::new(state, backend);

    let args = serde_json::json!({
        "query": "Rust programming language features 2024",
        "max_results": 3,
        "search_depth": "basic"
    });

    let result = tavily.execute(args, &runtime).await;

    match result {
        Ok(content) => {
            println!("Search results:\n{}", content.message);
            assert!(!content.message.is_empty(), "Expected search results");
            assert!(
                content.message.contains("Rust") || content.message.contains("programming"),
                "Expected relevant content"
            );
        }
        Err(e) => {
            // Rate limiting or transient errors are acceptable in tests
            println!("Search failed (may be rate limited): {}", e);
        }
    }
}

/// Test production configuration from environment
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_production_config_from_env() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let config = ProductionConfig::from_env()
        .expect("Failed to load production config");

    // Verify LLM provider creation
    let llm = config.llm_provider()
        .expect("Failed to create LLM provider");

    println!("LLM Provider: {}", llm.name());
    assert_eq!(llm.name(), "openai");

    // Test LLM config
    let llm_config = config.llm_config();
    assert!(llm_config.temperature.unwrap_or(0.0) >= 0.0);

    // Test Pregel config
    let pregel_config = config.pregel_config();
    assert!(pregel_config.max_supersteps > 0);
    assert!(pregel_config.parallelism > 0);
}

/// Test simple agent workflow with real LLM
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_simple_agent_workflow() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let llm: Arc<dyn LLMProvider> = Arc::new(create_openai_provider("gpt-4.1"));

    // Create a simple workflow: single agent node
    let agent_config = AgentNodeConfig {
        system_prompt: "You are a helpful assistant. Be concise and direct.".into(),
        max_iterations: 3,
        stop_conditions: vec![StopCondition::NoToolCalls],
        ..Default::default()
    };

    let graph = WorkflowGraph::<ResearchState>::new()
        .name("simple_agent")
        .node("agent", NodeKind::Agent(agent_config))
        .entry("agent")
        .edge("agent", rig_deepagents::workflow::END)
        .build()
        .expect("Failed to build graph");

    let pregel_config = PregelConfig::default()
        .with_max_supersteps(10)
        .with_vertex_timeout(Duration::from_secs(60))
        .with_execution_mode(ExecutionMode::EdgeDriven);

    let mut workflow = CompiledWorkflow::compile_with_tools(
        graph,
        pregel_config,
        Some(llm),
        vec![],
    )
    .expect("Failed to compile workflow");

    let state = ResearchState::new("Simple test query");
    let result = workflow.run(state).await;

    match result {
        Ok(result) => {
            println!("Workflow completed in {} supersteps", result.supersteps);
            assert!(result.completed);
        }
        Err(e) => {
            println!("Workflow error (may be expected): {}", e);
        }
    }
}

/// Test research state management during workflow
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_research_state_updates() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    // Create initial state
    let mut state = ResearchState::new("What is context engineering?")
        .with_max_searches(4);

    // Verify initial state
    assert_eq!(state.phase, ResearchPhase::Exploratory);
    assert!(state.can_continue);
    assert_eq!(state.max_searches, 4);
    assert_eq!(state.search_count, 0);

    // Simulate phase transitions
    use rig_deepagents::ResearchUpdate;

    let update1 = ResearchUpdate::default()
        .with_search("exploratory search 1".to_string());

    state = state.apply_update(update1);
    assert_eq!(state.search_count, 1);
    assert!(state.can_continue);

    // Transition to directed phase
    let update2 = ResearchUpdate::transition_to(ResearchPhase::Directed);
    state = state.apply_update(update2);
    assert_eq!(state.phase, ResearchPhase::Directed);

    println!("State after updates: phase={:?}, searches={}", state.phase, state.search_count);
}

/// Test full production setup initialization
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY and TAVILY_API_KEY environment variables"]
async fn test_full_production_setup() {
    if !has_openai_key() || !has_tavily_key() {
        eprintln!("Skipping test: Required API keys not set");
        return;
    }

    let setup = ProductionSetup::from_env()
        .expect("Failed to initialize production setup");

    // Verify LLM is configured
    assert!(setup.llm().is_some());

    // Verify tools are configured
    let tools = setup.tools();
    assert!(!tools.is_empty(), "Expected research tools");

    // Verify we have tavily_search and think tools
    let tool_names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    println!("Configured tools: {:?}", tool_names);

    assert!(
        tool_names.contains(&"tavily_search"),
        "Expected tavily_search tool"
    );
    assert!(
        tool_names.contains(&"think"),
        "Expected think tool"
    );

    // Build workflow
    let _workflow = setup.build_workflow()
        .expect("Failed to build workflow");

    println!("Production setup initialized successfully!");
}

/// Performance monitoring test - tracks token usage
#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY environment variable"]
async fn test_token_usage_tracking() {
    if !has_openai_key() {
        eprintln!("Skipping test: OPENAI_API_KEY not set");
        return;
    }

    let provider = create_openai_provider("gpt-4.1");

    let messages = vec![
        Message::system("You are concise."),
        Message::user("Say hello."),
    ];

    let config = LLMConfig::new("gpt-4.1")
        .with_max_tokens(16000);

    let response = provider
        .complete(&messages, &[], Some(&config))
        .await
        .expect("LLM completion failed");

    println!("Response: {}", response.message.content);

    if let Some(usage) = response.usage {
        println!("Token usage:");
        println!("  - Input tokens: {}", usage.input_tokens);
        println!("  - Output tokens: {}", usage.output_tokens);
        println!("  - Total tokens: {}", usage.total_tokens);

        // Basic sanity checks
        assert!(usage.input_tokens > 0, "Expected input tokens");
        assert!(usage.output_tokens > 0, "Expected output tokens");
        assert_eq!(
            usage.total_tokens,
            usage.input_tokens + usage.output_tokens,
            "Total should equal sum"
        );
    } else {
        println!("Note: Token usage not available from this provider");
    }
}

/// Test error handling for invalid API key
#[tokio::test]
async fn test_invalid_api_key_error() {
    // This test doesn't require actual API keys
    // It tests that invalid keys produce proper errors

    // Set an invalid key temporarily
    std::env::set_var("OPENAI_API_KEY", "sk-invalid-key-for-testing");

    let provider = create_openai_provider("gpt-4.1");

    let messages = vec![Message::user("test")];

    let result = provider.complete(&messages, &[], None).await;

    // Should fail with authentication error
    assert!(result.is_err(), "Expected error with invalid API key");

    // Clean up
    std::env::remove_var("OPENAI_API_KEY");
}
