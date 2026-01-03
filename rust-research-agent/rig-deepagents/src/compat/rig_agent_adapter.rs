//! Adapter for using Rig Agents as LLMProvider in rig-deepagents
//!
//! This module provides `RigAgentAdapter` which wraps a Rig Agent to implement
//! the `LLMProvider` trait, enabling integration with rig-deepagents' AgentExecutor.
//!
//! # Architecture Note
//!
//! Rig and rig-deepagents have different tool calling architectures:
//!
//! - **Rig**: Tools are configured at agent build time, and the agent's internal
//!   `PromptRequest` handles the tool calling loop automatically.
//!
//! - **rig-deepagents**: Tools are dynamic per-request, and `AgentExecutor` handles
//!   the tool calling loop externally.
//!
//! This adapter bridges these by:
//! 1. Using Rig's completion API for LLM responses and tool call metadata
//! 2. Letting rig-deepagents' AgentExecutor handle tool execution
//!
//! # Usage
//!
//! ```rust,ignore
//! use rig::providers::openai::Client;
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig_deepagents::compat::RigAgentAdapter;
//! use rig_deepagents::AgentExecutor;
//!
//! // Build a Rig agent (tools here are optional, AgentExecutor handles tool calls)
//! let client = Client::from_env();
//! let agent = client.agent("gpt-4").preamble("You are helpful.").build();
//!
//! // Wrap in adapter
//! let provider = RigAgentAdapter::new(agent);
//!
//! // Use with AgentExecutor
//! let executor = AgentExecutor::new(Arc::new(provider), middleware, backend);
//! ```
//!
//! # Limitations
//!
//! - Tool definitions passed to `complete()` are forwarded to Rig's completion API
//!   so the model can emit tool calls, but execution remains external.
//! - Streaming emits text chunks only; tool call streaming is ignored.

use async_trait::async_trait;
use std::sync::Arc;

use futures::StreamExt;

use rig::agent::Agent;
use rig::completion::{
    Completion, CompletionModel, GetTokenUsage, Message as RigMessage, ToolDefinition as RigToolDefinition,
};
use rig::message::{AssistantContent, ToolCall as RigToolCall};
use rig::streaming::StreamedAssistantContent;
use rig::OneOrMany;

use crate::error::DeepAgentError;
use crate::llm::{LLMConfig, LLMProvider, LLMResponse, LLMResponseStream, MessageChunk, TokenUsage};
use crate::middleware::ToolDefinition;
use crate::state::{Message, Role, ToolCall};

/// Adapter that wraps a Rig `Agent<M>` to implement `LLMProvider`.
///
/// This allows using Rig's rich agent ecosystem (with its 20+ LLM providers)
/// within rig-deepagents' middleware and executor framework.
///
/// # Type Parameters
///
/// - `M`: The Rig CompletionModel type (e.g., `OpenAI`, `Anthropic`)
pub struct RigAgentAdapter<M>
where
    M: CompletionModel + Send + Sync,
{
    agent: Arc<Agent<M>>,
    provider_name: String,
    model_name: String,
}

impl<M> RigAgentAdapter<M>
where
    M: CompletionModel + Send + Sync,
{
    /// Create a new adapter wrapping a Rig agent.
    ///
    /// # Arguments
    ///
    /// * `agent` - The Rig agent to wrap
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = rig::providers::openai::Client::from_env();
    /// let agent = client.agent("gpt-4").build();
    /// let provider = RigAgentAdapter::new(agent);
    /// ```
    pub fn new(agent: Agent<M>) -> Self {
        Self {
            agent: Arc::new(agent),
            provider_name: "rig".to_string(),
            model_name: "rig-agent".to_string(),
        }
    }

    /// Create adapter with custom provider/model names for logging.
    pub fn with_names(
        agent: Agent<M>,
        provider_name: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            agent: Arc::new(agent),
            provider_name: provider_name.into(),
            model_name: model_name.into(),
        }
    }

    /// Get a reference to the inner Rig agent.
    pub fn agent(&self) -> &Agent<M> {
        &self.agent
    }
}

#[async_trait]
impl<M> LLMProvider for RigAgentAdapter<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, DeepAgentError> {
        let conversation = build_rig_conversation(messages);
        let mut builder = self
            .agent
            .completion(conversation.prompt, conversation.history)
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Rig agent error: {}", e)))?;

        if let Some(system_preamble) = conversation.preamble {
            let preamble = match self.agent.preamble.as_deref() {
                Some(agent_preamble) => format!("{}\n\n{}", agent_preamble, system_preamble),
                None => system_preamble,
            };
            builder = builder.preamble(preamble);
        }

        if let Some(cfg) = config {
            if let Some(temperature) = cfg.temperature {
                builder = builder.temperature(temperature);
            }
            if let Some(max_tokens) = cfg.max_tokens {
                builder = builder.max_tokens(max_tokens);
            }
        }

        let rig_tools = to_rig_tool_definitions(tools);
        if !rig_tools.is_empty() {
            builder = builder.tools(rig_tools);
        }

        let response = builder
            .send()
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Rig agent error: {}", e)))?;

        let message = message_from_rig_choice(&response.choice);
        let usage = TokenUsage::from_rig_usage(&response.usage);

        let mut llm_response = LLMResponse::new(message);
        if usage.total_tokens > 0 {
            llm_response = llm_response.with_usage(usage);
        }

        Ok(llm_response)
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponseStream, DeepAgentError> {
        let conversation = build_rig_conversation(messages);
        let mut builder = self
            .agent
            .completion(conversation.prompt, conversation.history)
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Rig agent error: {}", e)))?;

        if let Some(system_preamble) = conversation.preamble {
            let preamble = match self.agent.preamble.as_deref() {
                Some(agent_preamble) => format!("{}\n\n{}", agent_preamble, system_preamble),
                None => system_preamble,
            };
            builder = builder.preamble(preamble);
        }

        if let Some(cfg) = config {
            if let Some(temperature) = cfg.temperature {
                builder = builder.temperature(temperature);
            }
            if let Some(max_tokens) = cfg.max_tokens {
                builder = builder.max_tokens(max_tokens);
            }
        }

        let rig_tools = to_rig_tool_definitions(tools);
        if !rig_tools.is_empty() {
            builder = builder.tools(rig_tools);
        }

        let stream = builder
            .stream()
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Rig agent error: {}", e)))?;

        let mapped = stream.filter_map(|item| async move {
            match item {
                Ok(StreamedAssistantContent::Text(text)) => Some(Ok(MessageChunk {
                    content: text.text,
                    is_final: false,
                    usage: None,
                })),
                Ok(StreamedAssistantContent::Final(response)) => {
                    let usage = response
                        .token_usage()
                        .map(|usage| TokenUsage::from_rig_usage(&usage))
                        .filter(|usage| usage.total_tokens > 0);
                    Some(Ok(MessageChunk {
                        content: String::new(),
                        is_final: true,
                        usage,
                    }))
                }
                Ok(_) => None,
                Err(err) => Some(Err(DeepAgentError::LlmError(format!(
                    "Rig agent error: {}",
                    err
                )))),
            }
        });

        Ok(LLMResponseStream::new(mapped))
    }

    fn name(&self) -> &str {
        &self.provider_name
    }

    fn default_model(&self) -> &str {
        &self.model_name
    }
}

struct RigConversation {
    prompt: RigMessage,
    history: Vec<RigMessage>,
    preamble: Option<String>,
}

fn build_rig_conversation(messages: &[Message]) -> RigConversation {
    let mut system_parts = Vec::new();
    let mut rig_messages = Vec::new();

    for message in messages {
        match message.role {
            Role::System => {
                if !message.content.trim().is_empty() {
                    system_parts.push(message.content.clone());
                }
            }
            Role::User => rig_messages.push(RigMessage::user(message.content.clone())),
            Role::Assistant => rig_messages.push(convert_assistant_message(message)),
            Role::Tool => rig_messages.push(convert_tool_message(message)),
        }
    }

    let prompt = rig_messages
        .pop()
        .unwrap_or_else(|| RigMessage::user(""));

    let preamble = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    RigConversation {
        prompt,
        history: rig_messages,
        preamble,
    }
}

fn convert_assistant_message(message: &Message) -> RigMessage {
    let mut contents = Vec::new();

    if !message.content.is_empty() {
        contents.push(AssistantContent::text(message.content.clone()));
    }

    if let Some(tool_calls) = &message.tool_calls {
        for call in tool_calls {
            contents.push(AssistantContent::tool_call(
                call.id.clone(),
                call.name.clone(),
                call.arguments.clone(),
            ));
        }
    }

    let content = if contents.is_empty() {
        OneOrMany::one(AssistantContent::text(""))
    } else {
        OneOrMany::many(contents).unwrap_or_else(|_| OneOrMany::one(AssistantContent::text("")))
    };

    RigMessage::Assistant { id: None, content }
}

fn convert_tool_message(message: &Message) -> RigMessage {
    let tool_id = message
        .tool_call_id
        .clone()
        .unwrap_or_else(|| "tool".to_string());
    RigMessage::tool_result(tool_id, message.content.clone())
}

fn to_rig_tool_definitions(tools: &[ToolDefinition]) -> Vec<RigToolDefinition> {
    tools
        .iter()
        .map(|tool| RigToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect()
}

fn message_from_rig_choice(choice: &OneOrMany<AssistantContent>) -> Message {
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for item in choice.iter() {
        match item {
            AssistantContent::Text(text) => content_parts.push(text.text.clone()),
            AssistantContent::ToolCall(tool_call) => {
                tool_calls.push(convert_rig_tool_call(tool_call));
            }
            AssistantContent::Reasoning(_) => {}
            AssistantContent::Image(_) => {}
        }
    }

    let content = content_parts.join("");

    if tool_calls.is_empty() {
        Message::assistant(&content)
    } else {
        Message::assistant_with_tool_calls(&content, tool_calls)
    }
}

fn convert_rig_tool_call(tool_call: &RigToolCall) -> ToolCall {
    ToolCall {
        id: tool_call.id.clone(),
        name: tool_call.function.name.clone(),
        arguments: tool_call.function.arguments.clone(),
    }
}

impl<M> std::fmt::Debug for RigAgentAdapter<M>
where
    M: CompletionModel + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RigAgentAdapter")
            .field("provider_name", &self.provider_name)
            .field("model_name", &self.model_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::message::UserContent;

    fn rig_message_text(message: &RigMessage) -> Option<String> {
        match message {
            RigMessage::User { content } => content.iter().find_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            }),
            RigMessage::Assistant { content, .. } => content.iter().find_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            }),
        }
    }

    #[test]
    fn test_build_rig_conversation_history_and_preamble() {
        let messages = vec![
            Message::system("system rules"),
            Message::user("hello"),
            Message::assistant("hi"),
            Message::user("next"),
        ];

        let conversation = build_rig_conversation(&messages);

        assert_eq!(conversation.preamble, Some("system rules".to_string()));
        assert_eq!(conversation.history.len(), 2);
        assert_eq!(rig_message_text(&conversation.history[0]).unwrap(), "hello");
        assert_eq!(rig_message_text(&conversation.history[1]).unwrap(), "hi");
        assert_eq!(rig_message_text(&conversation.prompt).unwrap(), "next");
    }

    #[test]
    fn test_message_from_rig_choice_with_tool_call() {
        let choice = OneOrMany::many(vec![
            AssistantContent::text(""),
            AssistantContent::tool_call(
                "call_1",
                "search",
                serde_json::json!({"query": "rust"}),
            ),
        ])
        .unwrap();

        let message = message_from_rig_choice(&choice);

        assert_eq!(message.role, Role::Assistant);
        assert!(message.tool_calls.is_some());

        let calls = message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "search");
    }
}
