# `Rust` - Research Agent

A production-ready AI research agent built with **Rust** and the **Rig framework**. This project is designed for a YouTube tutorial teaching beginners how to build their first AI agent.

## Features

- **Local LLM Support** - Uses Ollama for privacy-friendly, free AI inference
- **Web Search** - DuckDuckGo integration (no API key required!)
- **Tool-Using Agent** - Demonstrates agentic AI patterns
- **Beginner Friendly** - Extensive comments explaining Rust patterns
- **Production Ready** - Proper error handling, logging, and CLI

## Quick Start

### Prerequisites

1. **Install Rust** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install Ollama**:
   - Visit [ollama.ai](https://ollama.ai) and follow installation instructions
   - Or on Linux: `curl -fsSL https://ollama.com/install.sh | sh`

3. **Pull a model**:
   ```bash
   ollama pull devstral-small-2
   ```

4. **Start Ollama**:
   ```bash
   ollama serve
   ```

### Installation

```bash
# Copy environment template
cp .env.example .env

# Build the project
cargo build --release
```

### Usage

```bash
# Basic research query
cargo run -- "What are the latest developments in Rust async runtime?"

# Quick search mode (no AI synthesis)
cargo run --release -- --quick "Rust web frameworks 2024"

# Use a specific model
cargo run -- --model deepseek-v3.2 "Machine learning in Rust"

# Verbose output
cargo run -- --verbose "WebAssembly trends"

# Show help
cargo run -- --help
```

## Project Structure

```
rust-research-agent/
├── Cargo.toml          # Project dependencies and metadata
├── .env.example        # Environment variable template
├── README.md           # This file
└── src/
    ├── main.rs         # CLI entry point and application logic
    ├── config.rs       # Configuration management
    ├── agent.rs        # Research agent implementation
    └── tools.rs        # Web search tool (DuckDuckGo)
```

## Configuration

Edit `.env` to customize the agent:

```bash
# Model to use (must be installed in Ollama)
OLLAMA_MODEL=llama3.2

# Ollama server URL
OLLAMA_HOST=http://localhost:11434

# Response creativity (0.0 = focused, 1.0 = creative)
TEMPERATURE=0.7

# Number of search results to analyze
MAX_SEARCH_RESULTS=5

# Logging level
RUST_LOG=info
```

## Learning Rust Concepts

This codebase demonstrates these Rust concepts with inline comments:

| Concept | File | Description |
|---------|------|-------------|
| **Structs & Enums** | `config.rs` | Data types and pattern matching |
| **Traits** | `tools.rs` | Implementing the Rig `Tool` trait |
| **Ownership & Borrowing** | `agent.rs` | Memory safety without GC |
| **Async/Await** | `agent.rs`, `tools.rs` | Non-blocking I/O |
| **Error Handling** | All files | `Result`, `?` operator, `anyhow` |
| **Derive Macros** | All files | `Debug`, `Clone`, `Serialize` |
| **Unit Tests** | All files | The `#[cfg(test)]` pattern |

## Extending the Agent

### Adding a New Tool

1. Create a new struct in `tools.rs`:
   ```rust
   pub struct MyNewTool {
       // fields
   }
   ```

2. Implement the `Tool` trait:
   ```rust
   impl Tool for MyNewTool {
       const NAME: &'static str = "my_tool";
       // ... implement required methods
   }
   ```

3. Register with the agent in `agent.rs`:
   ```rust
   let agent = client
       .agent(&model)
       .tool(web_search_tool)
       .tool(my_new_tool)  // Add here
       .build();
   ```

Then set `OLLAMA_MODEL` in `.env` or use `--model` flag.

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_config
```


## Acknowledgments

- [Rig Framework](https://rig.rs) - The Rust AI framework
- [Ollama](https://ollama.ai) - Local LLM runner
- [DuckDuckGo](https://duckduckgo.com) - Privacy-respecting search
