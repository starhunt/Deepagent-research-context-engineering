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
//! 1. Using Rig's agent for LLM completion calls
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
//! - Tool definitions passed to `complete()` are used for schema injection but
//!   actual tool execution is handled by AgentExecutor
//! - Streaming is not yet implemented (falls back to complete)

use async_trait::async_trait;
use std::sync::Arc;

use rig::agent::Agent;
use rig::completion::{CompletionModel, Prompt};

use crate::error::DeepAgentError;
use crate::llm::{LLMConfig, LLMProvider, LLMResponse, LLMResponseStream};
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
        _config: Option<&LLMConfig>,
    ) -> Result<LLMResponse, DeepAgentError> {
        // Build the prompt from messages
        // For now, we take the last user message as the prompt
        // and include tool schemas in the prompt for LLM awareness
        let prompt = build_prompt_with_tools(messages, tools);

        // Call Rig agent's prompt method
        let response = self
            .agent
            .prompt(&prompt)
            .await
            .map_err(|e| DeepAgentError::LlmError(format!("Rig agent error: {}", e)))?;

        // Parse the response for potential tool calls
        let message = parse_response_for_tool_calls(&response);

        Ok(LLMResponse::new(message))
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        config: Option<&LLMConfig>,
    ) -> Result<LLMResponseStream, DeepAgentError> {
        // Fallback to complete for now
        // TODO: Implement proper streaming using Rig's streaming API
        let response = self.complete(messages, tools, config).await?;
        Ok(LLMResponseStream::from_complete(response))
    }

    fn name(&self) -> &str {
        &self.provider_name
    }

    fn default_model(&self) -> &str {
        &self.model_name
    }
}

/// Build a prompt string from messages, optionally including tool schemas.
fn build_prompt_with_tools(messages: &[Message], tools: &[ToolDefinition]) -> String {
    let mut prompt_parts = Vec::new();

    // Add tool schemas if present
    if !tools.is_empty() {
        let tools_section = build_tools_section(tools);
        prompt_parts.push(tools_section);
    }

    // Find the last user message
    let last_user_msg = messages
        .iter()
        .rfind(|m| m.role == Role::User)
        .map(|m| m.content.clone())
        .unwrap_or_default();

    prompt_parts.push(last_user_msg);

    prompt_parts.join("\n\n")
}

/// Build a tools description section for the prompt.
fn build_tools_section(tools: &[ToolDefinition]) -> String {
    let mut section = String::from("You have access to the following tools:\n\n");

    for tool in tools {
        section.push_str(&format!(
            "**{}**: {}\nParameters: {}\n\n",
            tool.name,
            tool.description,
            serde_json::to_string_pretty(&tool.parameters).unwrap_or_default()
        ));
    }

    section.push_str(
        "To use a tool, respond with a JSON object in this format:\n\
         {\"tool_calls\": [{\"id\": \"unique_id\", \"name\": \"tool_name\", \"arguments\": {...}}]}\n\n\
         Or respond normally if no tool is needed.",
    );

    section
}

/// Parse LLM response for potential tool calls.
///
/// If the response contains a valid tool_calls JSON structure, extract it.
/// Otherwise, treat the response as a normal text response.
fn parse_response_for_tool_calls(response: &str) -> Message {
    // Try to parse as JSON with tool_calls
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(tool_calls_val) = parsed.get("tool_calls") {
            if let Ok(tool_calls) = serde_json::from_value::<Vec<ToolCallJson>>(tool_calls_val.clone()) {
                if !tool_calls.is_empty() {
                    let calls: Vec<ToolCall> = tool_calls
                        .into_iter()
                        .map(|tc| ToolCall {
                            id: tc.id,
                            name: tc.name,
                            arguments: tc.arguments,
                        })
                        .collect();
                    return Message::assistant_with_tool_calls("", calls);
                }
            }
        }
    }

    // Default: treat as normal text response
    Message::assistant(response)
}

/// JSON structure for parsing tool calls from response.
#[derive(serde::Deserialize)]
struct ToolCallJson {
    id: String,
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
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

    #[test]
    fn test_build_tools_section() {
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }];

        let section = build_tools_section(&tools);
        assert!(section.contains("search"));
        assert!(section.contains("Search the web"));
        assert!(section.contains("tool_calls"));
    }

    #[test]
    fn test_parse_response_text() {
        let response = "Hello, I'm here to help!";
        let message = parse_response_for_tool_calls(response);

        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.content, response);
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn test_parse_response_with_tool_calls() {
        let response = r#"{"tool_calls": [{"id": "call_1", "name": "search", "arguments": {"query": "rust"}}]}"#;
        let message = parse_response_for_tool_calls(response);

        assert_eq!(message.role, Role::Assistant);
        assert!(message.tool_calls.is_some());

        let calls = message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let response = "This is not JSON {invalid}";
        let message = parse_response_for_tool_calls(response);

        // Should treat as normal text
        assert_eq!(message.content, response);
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn test_build_prompt_no_tools() {
        let messages = vec![Message::user("Hello!")];
        let prompt = build_prompt_with_tools(&messages, &[]);

        assert_eq!(prompt, "Hello!");
    }

    #[test]
    fn test_build_prompt_with_tools() {
        let messages = vec![Message::user("Search for Rust")];
        let tools = vec![ToolDefinition {
            name: "search".to_string(),
            description: "Search".to_string(),
            parameters: serde_json::json!({}),
        }];

        let prompt = build_prompt_with_tools(&messages, &tools);

        assert!(prompt.contains("search"));
        assert!(prompt.contains("Search for Rust"));
    }
}
