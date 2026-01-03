//! Compatibility layer for Rig framework integration
//!
//! This module provides adapters to bridge between rig-deepagents and Rig's
//! native abstractions, enabling seamless use of both ecosystems.
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    rig-deepagents                           │
//! │  ┌──────────────────┐       ┌──────────────────────────┐   │
//! │  │ Tool trait       │       │ LLMProvider trait        │   │
//! │  │ (dynamic, JSON)  │       │ (messages + tools)       │   │
//! │  └────────┬─────────┘       └────────────┬─────────────┘   │
//! │           │                              │                  │
//! │           │ adapts                       │ adapts           │
//! │           ▼                              ▼                  │
//! │  ┌──────────────────┐       ┌──────────────────────────┐   │
//! │  │ RigToolAdapter   │       │ RigAgentAdapter          │   │
//! │  │ (wraps Rig Tool) │       │ (wraps Rig Agent)        │   │
//! │  └────────┬─────────┘       └────────────┬─────────────┘   │
//! └───────────│──────────────────────────────│──────────────────┘
//!             │                              │
//!             ▼                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Rig Framework                            │
//! │  ┌──────────────────┐       ┌──────────────────────────┐   │
//! │  │ Tool trait       │       │ Agent<M>                 │   │
//! │  │ (static types)   │       │ (CompletionModel)        │   │
//! │  └──────────────────┘       └──────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ## Using Rig Tools in rig-deepagents
//!
//! ```rust,ignore
//! use rig::tools::think::ThinkTool;
//! use rig_deepagents::compat::RigToolAdapter;
//!
//! // Wrap Rig's ThinkTool for use in rig-deepagents
//! let think = RigToolAdapter::new(ThinkTool).await;
//! middleware.add_tool(Arc::new(think));
//! ```
//!
//! ## Using Rig Agent as LLMProvider
//!
//! ```rust,ignore
//! use rig::providers::openai::Client;
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig_deepagents::compat::RigAgentAdapter;
//!
//! let client = Client::from_env();
//! let agent = client.agent("gpt-4").build();
//! let provider = RigAgentAdapter::new(agent);
//!
//! // Now use provider with AgentExecutor
//! let executor = AgentExecutor::new(Arc::new(provider), middleware, backend);
//! ```

mod rig_tool_adapter;
mod rig_agent_adapter;

pub use rig_tool_adapter::RigToolAdapter;
pub use rig_agent_adapter::RigAgentAdapter;
