//! LLM Provider abstractions for DeepAgents
//!
//! This module provides provider-agnostic interfaces for LLM completion,
//! bridging DeepAgents with various LLM providers via Rig framework.
//!
//! # Architecture
//!
//! The LLM module follows a layered design inspired by LangChain:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │          DeepAgents Executor            │
//! └─────────────────┬───────────────────────┘
//!                   │ uses
//!                   ▼
//! ┌─────────────────────────────────────────┐
//! │        LLMProvider (trait)              │
//! │  - complete(messages, tools, config)    │
//! │  - stream(messages, tools, config)      │
//! └─────────────────┬───────────────────────┘
//!                   │ implemented by
//!          ┌────────┴────────┐
//!          ▼                 ▼
//! ┌─────────────────┐ ┌─────────────────┐
//! │ OpenAIProvider  │ │AnthropicProvider│
//! │ (via rig-core)  │ │ (via rig-core)  │
//! └─────────────────┘ └─────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use rig_deepagents::llm::{LLMProvider, OpenAIProvider, LLMConfig};
//!
//! // Create provider from environment
//! let provider = OpenAIProvider::from_env()?;
//!
//! // Or with custom configuration
//! let provider = OpenAIProvider::new("api-key", "gpt-4.1");
//!
//! // Generate completion
//! let messages = vec![Message::user("Hello!")];
//! let response = provider.complete(&messages, &[], None).await?;
//! ```

mod config;
mod provider;
mod message;
mod openai;
mod anthropic;

pub use config::{LLMConfig, TokenUsage};
pub use provider::{LLMProvider, LLMResponse, LLMResponseStream, MessageChunk};
pub use message::{MessageConverter, ToolConverter, convert_messages, convert_tools};
pub use openai::OpenAIProvider;
pub use anthropic::AnthropicProvider;
