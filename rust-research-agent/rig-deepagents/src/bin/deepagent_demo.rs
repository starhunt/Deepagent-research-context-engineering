//! DeepAgent Interactive Demo
//!
//! Rust DeepAgent가 Python 버전처럼 동작하는 것을 시각적으로 확인할 수 있는
//! 인터랙티브 CLI 데모입니다.
//!
//! 이 데모는 Rig 프레임워크의 네이티브 에이전트 시스템을 사용하여
//! 실제 LLM API를 호출하고 도구를 실행합니다.
//!
//! # 필수 환경 변수
//!
//! - `OPENAI_API_KEY`: OpenAI API 키
//!
//! # Usage
//!
//! ```bash
//! # 기본 실행
//! cargo run --bin deepagent-demo
//!
//! # 커스텀 쿼리
//! cargo run --bin deepagent-demo -- --query "What is 2+2?"
//!
//! # 도구 사용 유도
//! cargo run --bin deepagent-demo -- --query "Think step by step: what is the capital of France?"
//! ```

use std::time::{Duration, Instant};

use clap::Parser;
use colored::Colorize;

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::openai::Client;
use rig::completion::Prompt;
use rig::tools::think::ThinkTool;

/// DeepAgent Demo CLI
#[derive(Parser, Debug)]
#[command(name = "deepagent-demo")]
#[command(about = "Interactive demo for Rust DeepAgent using Rig framework")]
#[command(version)]
struct Args {
    /// 연구 쿼리
    #[arg(short, long)]
    query: Option<String>,

    /// 사용할 모델
    #[arg(short, long, default_value = "gpt-4.1")]
    model: String,
}

// =============================================================================
// .env 파일 로딩
// =============================================================================

fn load_dotenv() {
    for path in [".env", "../.env", "../../.env", "../../../.env"] {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((key, value)) = line.split_once('=') {
                    let key = key.trim();
                    let value = value.trim().trim_matches('"').trim_matches('\'');
                    if std::env::var(key).is_err() {
                        std::env::set_var(key, value);
                    }
                }
            }
            break;
        }
    }
}

// =============================================================================
// Output Formatting
// =============================================================================

fn print_header(args: &Args) {
    let separator = "━".repeat(60);

    println!();
    println!("{}", separator.cyan());
    println!("{}", "🚀 Rig-DeepAgents Demo - Native Rig Agent".cyan().bold());
    println!("{}", separator.cyan());
    println!();

    println!("{}", "📋 Configuration:".white().bold());
    println!("   Model: {}", args.model.green());
    println!("   Framework: {} (with native tool support)", "Rig".cyan());
    println!();

    println!("{}", "🔑 API Keys:".white().bold());
    if std::env::var("OPENAI_API_KEY").is_ok() {
        println!("   OPENAI_API_KEY: {}", "✓ Found".green());
    } else {
        println!("   OPENAI_API_KEY: {}", "✗ Missing".red());
    }
    println!();
}

fn truncate_to_width(s: &str, max_width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > max_width {
        chars[..max_width.saturating_sub(3)].iter().collect::<String>() + "..."
    } else {
        s.to_string()
    }
}

fn print_section(title: &str, emoji: &str) {
    let separator = "═".repeat(60);
    println!();
    println!("{}", separator.white());
    println!(" {} {}", emoji, title.white().bold());
    println!("{}", separator.white());
    println!();
}

fn print_box(role: &str, emoji: &str, title: &str, content: &str) {
    let title_colored = match role {
        "ai" => title.green().to_string(),
        "human" => title.blue().to_string(),
        "tool" => title.yellow().to_string(),
        "system" => title.cyan().to_string(),
        _ => title.white().to_string(),
    };

    let border_len = 55usize.saturating_sub(title.len());
    let border = "-".repeat(border_len);

    println!("+- {} {} {}+", emoji, title_colored, border);

    for line in content.lines() {
        println!("| {}", truncate_to_width(line, 55));
    }

    println!("+{}+", "-".repeat(58));
    println!();
}

fn print_stats(duration: Duration, response_len: usize) {
    let separator = "━".repeat(60);

    println!();
    println!("{}", separator.cyan());
    println!("{}", "✅ Demo Complete!".green().bold());
    println!("{}", separator.cyan());
    println!();

    println!("{}", "📊 Execution Statistics:".white().bold());
    println!("   ├─ Duration: {:.2}s", duration.as_secs_f64());
    println!("   └─ Response Length: {} chars", response_len);
    println!();
}

// =============================================================================
// Demo Runner
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load environment from .env file
    load_dotenv();

    let args = Args::parse();

    // Check for required API key
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("{}", "❌ Error: OPENAI_API_KEY environment variable not set!".red().bold());
        eprintln!();
        eprintln!("Please set your OpenAI API key:");
        eprintln!("  export OPENAI_API_KEY=sk-...");
        eprintln!();
        std::process::exit(1);
    }

    print_header(&args);

    // Default query
    let query = args.query.unwrap_or_else(|| {
        "Think step by step about what makes Rust a good language for systems programming. Use the think tool to reason through this.".to_string()
    });

    // ==========================================================================
    // Initialize Rig Agent
    // ==========================================================================

    print_section("Agent Initialization", "🤖");

    println!("{}", "[INFO] Initializing OpenAI client...".dimmed());
    let openai_client = Client::from_env();

    println!("{}", format!("[INFO] Building agent with model: {}", args.model).dimmed());
    println!("{}", "[INFO] Adding ThinkTool for reasoning...".dimmed());

    // Create agent with ThinkTool
    let agent = openai_client
        .agent(&args.model)
        .preamble("You are an intelligent research assistant. \
                   When asked to think through problems, use the think tool to reason step by step. \
                   Be concise and clear in your responses.")
        .tool(ThinkTool)
        .temperature(0.0)
        .build();

    print_box(
        "system", "🔧", "Agent Configuration",
        &format!(
            "Model: {}\n\
             Temperature: 0.0\n\
             \n\
             Available Tools:\n\
             └─ think: Reason through complex problems step by step\n\
             \n\
             System Prompt:\n\
             You are an intelligent research assistant...",
            args.model
        )
    );

    // ==========================================================================
    // Execute Query
    // ==========================================================================

    print_section("Query Execution", "💬");

    print_box(
        "human", "📨", "User Query",
        &format!("\"{}\"", truncate_to_width(&query, 200))
    );

    println!("{}", "[INFO] Sending query to OpenAI...".dimmed());
    println!("{}", "[INFO] (Tool calls will be handled automatically by Rig)".dimmed());
    println!();

    let start_time = Instant::now();

    // Execute the prompt - Rig handles tool calling automatically
    let response = agent.prompt(&query).await?;

    let duration = start_time.elapsed();

    // ==========================================================================
    // Display Response
    // ==========================================================================

    print_section("Agent Response", "🤖");

    print_box(
        "ai", "💡", "Final Answer",
        &response
    );

    // Print the full response
    println!("{}", "📝 Complete Response:".white().bold());
    println!("{}", "-".repeat(60));
    for line in response.lines() {
        println!("  {}", line);
    }
    println!("{}", "-".repeat(60));

    print_stats(duration, response.len());

    println!("{}", "💡 Tip:".yellow().bold());
    println!("   Rig automatically handles tool calls internally.");
    println!("   The ThinkTool allows the model to reason step by step.");
    println!("   Try different queries to see how the agent responds!");
    println!();

    Ok(())
}
