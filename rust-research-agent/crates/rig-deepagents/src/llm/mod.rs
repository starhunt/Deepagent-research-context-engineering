//! LLM Provider abstractions for DeepAgents
//!
//! This module provides provider-agnostic interfaces for LLM completion,
//! bridging DeepAgents with Rig framework's native providers.
//!
//! # Architecture
//!
//! The LLM module follows a layered design:
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
//!                   ▼
//! ┌─────────────────────────────────────────┐
//! │         RigAgentAdapter                 │
//! │   (wraps any Rig Agent<M>)              │
//! └─────────────────────────────────────────┘
//!                   │ delegates to
//!          ┌────────┴────────┐
//!          ▼                 ▼
//! ┌─────────────────┐ ┌─────────────────┐
//! │  Rig OpenAI     │ │ Rig Anthropic   │
//! │   (20+ more)    │ │                 │
//! └─────────────────┘ └─────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use rig::providers::openai::Client;
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig_deepagents::compat::RigAgentAdapter;
//! use rig_deepagents::llm::LLMProvider;
//!
//! // Create Rig agent
//! let client = Client::from_env();
//! let agent = client.agent("gpt-4").build();
//!
//! // Wrap in adapter
//! let provider = RigAgentAdapter::new(agent);
//!
//! // Generate completion
//! let messages = vec![Message::user("Hello!")];
//! let response = provider.complete(&messages, &[], None).await?;
//! ```
//!
//! # Migration from Legacy Providers
//!
//! The legacy `OpenAIProvider` and `AnthropicProvider` have been removed.
//! Use `RigAgentAdapter` with Rig's native providers instead:
//!
//! ```rust,ignore
//! // Old (removed):
//! // let provider = OpenAIProvider::from_env()?;
//!
//! // New:
//! let client = rig::providers::openai::Client::from_env();
//! let agent = client.agent("gpt-4").build();
//! let provider = RigAgentAdapter::new(agent);
//! ```

mod config;
mod provider;
mod message;

pub use config::{LLMConfig, TokenUsage};
pub use provider::{LLMProvider, LLMResponse, LLMResponseStream, MessageChunk};
pub use message::{MessageConverter, ToolConverter, convert_messages, convert_tools};

// Re-export message utilities
pub use message::extract_system_preamble;
