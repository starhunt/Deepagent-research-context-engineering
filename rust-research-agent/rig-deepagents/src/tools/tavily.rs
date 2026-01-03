//! Tavily Search Tool - Web search for research agents
//!
//! Provides web search capabilities using the Tavily Search API.
//! Supports configurable search depth, result limits, and topic filtering.
//!
//! # Production Features
//!
//! - Type-safe enums for search_depth and topic
//! - HTTP timeout and retry with exponential backoff
//! - Typed error handling for rate limits and timeouts
//! - Complete JSON schema for LLM function calling

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, warn};

use crate::error::MiddlewareError;
use crate::middleware::{Tool, ToolDefinition, ToolResult};
use crate::runtime::ToolRuntime;

/// Default timeout for Tavily API requests
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Maximum retry attempts for transient failures
const MAX_RETRIES: u32 = 3;

/// Base delay for exponential backoff (milliseconds)
const RETRY_BASE_DELAY_MS: u64 = 1000;

/// Search depth for Tavily API
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SearchDepth {
    /// Fast search with basic results
    #[default]
    Basic,
    /// More thorough search with detailed results
    Advanced,
}

impl SearchDepth {
    fn as_str(&self) -> &'static str {
        match self {
            SearchDepth::Basic => "basic",
            SearchDepth::Advanced => "advanced",
        }
    }
}

/// Topic filter for Tavily API
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Topic {
    /// General web search
    #[default]
    General,
    /// Recent news articles
    News,
}

impl Topic {
    fn as_str(&self) -> &'static str {
        match self {
            Topic::General => "general",
            Topic::News => "news",
        }
    }
}

/// Tavily Search Tool for web research
///
/// # Example
/// ```ignore
/// let tool = TavilySearchTool::new("your-api-key");
/// let result = tool.execute(json!({
///     "query": "Rust async programming",
///     "max_results": 5
/// }), &runtime).await?;
/// ```
pub struct TavilySearchTool {
    api_key: String,
    client: Client,
    timeout: Duration,
    max_retries: u32,
}

impl TavilySearchTool {
    /// Create a new TavilySearchTool with the given API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_retries: MAX_RETRIES,
        }
    }

    /// Create from environment variable TAVILY_API_KEY
    pub fn from_env() -> Result<Self, MiddlewareError> {
        let api_key = std::env::var("TAVILY_API_KEY").map_err(|_| {
            MiddlewareError::ToolExecution(
                "TAVILY_API_KEY environment variable not set".to_string(),
            )
        })?;
        Ok(Self::new(api_key))
    }

    /// Set custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom max retries
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Execute HTTP request with retry and backoff
    async fn execute_with_retry(
        &self,
        request: &TavilyRequest,
    ) -> Result<TavilyResponse, TavilyError> {
        let mut last_error = TavilyError::Unknown("No attempts made".to_string());

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(RETRY_BASE_DELAY_MS * 2u64.pow(attempt - 1));
                debug!(attempt, delay_ms = delay.as_millis(), "Retrying Tavily request");
                tokio::time::sleep(delay).await;
            }

            match self.execute_single_request(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    // Only retry on transient errors
                    if !e.is_retryable() {
                        return Err(e);
                    }
                    warn!(attempt, error = %e, "Tavily request failed, will retry");
                    last_error = e;
                }
            }
        }

        Err(last_error)
    }

    /// Execute a single HTTP request
    async fn execute_single_request(
        &self,
        request: &TavilyRequest,
    ) -> Result<TavilyResponse, TavilyError> {
        let response = self
            .client
            .post("https://api.tavily.com/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(self.timeout)
            .json(request)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    TavilyError::Timeout
                } else if e.is_connect() {
                    TavilyError::Connection(e.to_string())
                } else {
                    TavilyError::Network(e.to_string())
                }
            })?;

        let status = response.status();

        if status.is_success() {
            let tavily_response: TavilyResponse = response
                .json()
                .await
                .map_err(|e| TavilyError::ParseError(e.to_string()))?;
            return Ok(tavily_response);
        }

        // Handle specific HTTP errors
        let error_text = response.text().await.unwrap_or_default();

        match status.as_u16() {
            401 => Err(TavilyError::Unauthorized),
            429 => Err(TavilyError::RateLimited),
            400 => Err(TavilyError::BadRequest(error_text)),
            500..=599 => Err(TavilyError::ServerError(status.as_u16(), error_text)),
            _ => Err(TavilyError::HttpError(status.as_u16(), error_text)),
        }
    }
}

/// Typed errors for Tavily API
#[derive(Debug, thiserror::Error)]
pub enum TavilyError {
    #[error("Request timed out")]
    Timeout,

    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Unauthorized - check API key")]
    Unauthorized,

    #[error("Rate limited - too many requests")]
    RateLimited,

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Server error ({0}): {1}")]
    ServerError(u16, String),

    #[error("HTTP error ({0}): {1}")]
    HttpError(u16, String),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl TavilyError {
    /// Check if this error is retryable
    fn is_retryable(&self) -> bool {
        matches!(
            self,
            TavilyError::Timeout
                | TavilyError::Connection(_)
                | TavilyError::RateLimited
                | TavilyError::ServerError(_, _)
        )
    }
}

impl From<TavilyError> for MiddlewareError {
    fn from(e: TavilyError) -> Self {
        MiddlewareError::ToolExecution(format!("Tavily API error: {}", e))
    }
}

/// Arguments for the tavily_search tool
#[derive(Debug, Deserialize)]
struct TavilySearchArgs {
    /// The search query
    query: String,

    /// Maximum number of results (default: 5)
    #[serde(default = "default_max_results")]
    max_results: u32,

    /// Search depth (default: basic)
    #[serde(default)]
    search_depth: SearchDepth,

    /// Topic filter (default: general)
    #[serde(default)]
    topic: Topic,

    /// Include AI-generated answer in response
    #[serde(default)]
    include_answer: bool,

    /// Include raw HTML content in results
    #[serde(default)]
    include_raw_content: bool,
}

fn default_max_results() -> u32 {
    5
}

/// Request body for Tavily API
#[derive(Debug, Serialize)]
struct TavilyRequest {
    query: String,
    max_results: u32,
    search_depth: String,
    topic: String,
    include_answer: bool,
    include_raw_content: bool,
}

/// Response from Tavily API
#[derive(Debug, Deserialize)]
struct TavilyResponse {
    /// AI-generated answer (if requested)
    answer: Option<String>,

    /// Search results
    results: Vec<TavilyResult>,
}

/// Individual search result
#[derive(Debug, Deserialize)]
struct TavilyResult {
    /// Page title
    title: String,

    /// Page URL
    url: String,

    /// Extracted content/snippet
    content: String,

    /// Relevance score (0-1)
    score: f64,

    /// Raw HTML content (if requested)
    raw_content: Option<String>,
}

impl TavilyResult {
    /// Format as markdown for LLM consumption
    fn to_markdown(&self, include_raw: bool) -> String {
        let mut output = format!(
            "### [{}]({})\n**Relevance:** {:.0}%\n\n{}\n",
            self.title,
            self.url,
            self.score * 100.0,
            self.content
        );

        if include_raw {
            if let Some(ref raw) = self.raw_content {
                // Truncate raw content to avoid token explosion
                let truncated = if raw.len() > 2000 {
                    format!("{}...[truncated]", &raw[..2000])
                } else {
                    raw.clone()
                };
                output.push_str(&format!("\n<details>\n<summary>Raw Content</summary>\n\n```html\n{}\n```\n</details>\n", truncated));
            }
        }

        output
    }
}

#[async_trait]
impl Tool for TavilySearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "tavily_search".to_string(),
            description: "Search the web using Tavily Search API. Returns relevant web pages with titles, URLs, and content snippets.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute",
                        "maxLength": 400
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth - 'basic' for fast results, 'advanced' for more thorough search",
                        "default": "basic"
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "description": "Topic filter - 'general' for all content, 'news' for recent news",
                        "default": "general"
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Include an AI-generated answer summarizing the results",
                        "default": false
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Include raw HTML content in results (increases response size)",
                        "default": false
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        runtime: &ToolRuntime,
    ) -> Result<ToolResult, MiddlewareError> {
        // Log tool execution for tracing
        if let Some(tool_call_id) = runtime.tool_call_id() {
            debug!(tool_call_id, "Executing tavily_search");
        }

        let args: TavilySearchArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        // Validate query length
        if args.query.len() > 400 {
            return Err(MiddlewareError::ToolExecution(
                "Query too long (max 400 characters)".to_string(),
            ));
        }

        // Validate and clamp max_results
        let max_results = args.max_results.clamp(1, 20);

        // Build request with type-safe enums
        let request = TavilyRequest {
            query: args.query.clone(),
            max_results,
            search_depth: args.search_depth.as_str().to_string(),
            topic: args.topic.as_str().to_string(),
            include_answer: args.include_answer,
            include_raw_content: args.include_raw_content,
        };

        // Execute with retry
        let tavily_response = self.execute_with_retry(&request).await?;

        // Format results as markdown
        let mut output = format!("## Search Results for: \"{}\"\n\n", args.query);

        // Include AI answer if present
        if let Some(answer) = tavily_response.answer {
            output.push_str("### AI Summary\n");
            output.push_str(&answer);
            output.push_str("\n\n---\n\n");
        }

        // Add results
        if tavily_response.results.is_empty() {
            output.push_str("No results found.\n");
        } else {
            output.push_str(&format!(
                "Found {} results:\n\n",
                tavily_response.results.len()
            ));
            for result in &tavily_response.results {
                output.push_str(&result.to_markdown(args.include_raw_content));
                output.push('\n');
            }
        }

        Ok(ToolResult::new(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Unit Tests ====================

    #[test]
    fn test_search_depth_serialization() {
        assert_eq!(SearchDepth::Basic.as_str(), "basic");
        assert_eq!(SearchDepth::Advanced.as_str(), "advanced");

        let json = serde_json::to_string(&SearchDepth::Advanced).unwrap();
        assert_eq!(json, r#""advanced""#);

        let parsed: SearchDepth = serde_json::from_str(r#""basic""#).unwrap();
        assert_eq!(parsed, SearchDepth::Basic);
    }

    #[test]
    fn test_topic_serialization() {
        assert_eq!(Topic::General.as_str(), "general");
        assert_eq!(Topic::News.as_str(), "news");

        let json = serde_json::to_string(&Topic::News).unwrap();
        assert_eq!(json, r#""news""#);

        let parsed: Topic = serde_json::from_str(r#""general""#).unwrap();
        assert_eq!(parsed, Topic::General);
    }

    #[test]
    fn test_tavily_tool_definition() {
        let tool = TavilySearchTool::new("test-key");
        let def = tool.definition();

        assert_eq!(def.name, "tavily_search");
        assert!(def.description.contains("Search the web"));

        // Verify required parameters
        let params = &def.parameters;
        let required = params["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("query")));

        // Verify include_raw_content is in schema (was missing before)
        assert!(params["properties"]["include_raw_content"].is_object());

        // Verify additionalProperties is false
        assert_eq!(params["additionalProperties"], serde_json::json!(false));

        // Verify constraints
        assert_eq!(params["properties"]["max_results"]["maximum"], 20);
        assert_eq!(params["properties"]["query"]["maxLength"], 400);
    }

    #[test]
    fn test_tavily_args_defaults() {
        let args: TavilySearchArgs = serde_json::from_str(r#"{"query": "test"}"#).unwrap();

        assert_eq!(args.query, "test");
        assert_eq!(args.max_results, 5);
        assert_eq!(args.search_depth, SearchDepth::Basic);
        assert_eq!(args.topic, Topic::General);
        assert!(!args.include_answer);
        assert!(!args.include_raw_content);
    }

    #[test]
    fn test_tavily_args_with_enums() {
        let args: TavilySearchArgs = serde_json::from_str(
            r#"{
                "query": "Rust async",
                "max_results": 10,
                "search_depth": "advanced",
                "topic": "news",
                "include_answer": true,
                "include_raw_content": true
            }"#,
        )
        .unwrap();

        assert_eq!(args.query, "Rust async");
        assert_eq!(args.max_results, 10);
        assert_eq!(args.search_depth, SearchDepth::Advanced);
        assert_eq!(args.topic, Topic::News);
        assert!(args.include_answer);
        assert!(args.include_raw_content);
    }

    #[test]
    fn test_tavily_args_invalid_enum() {
        let result: Result<TavilySearchArgs, _> = serde_json::from_str(
            r#"{"query": "test", "search_depth": "invalid"}"#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_tavily_result_to_markdown_basic() {
        let result = TavilyResult {
            title: "Test Title".to_string(),
            url: "https://example.com".to_string(),
            content: "This is test content.".to_string(),
            score: 0.95,
            raw_content: None,
        };

        let md = result.to_markdown(false);
        assert!(md.contains("### [Test Title](https://example.com)"));
        assert!(md.contains("**Relevance:** 95%"));
        assert!(md.contains("This is test content."));
        assert!(!md.contains("<details>"));
    }

    #[test]
    fn test_tavily_result_to_markdown_with_raw() {
        let result = TavilyResult {
            title: "Test".to_string(),
            url: "https://example.com".to_string(),
            content: "Content".to_string(),
            score: 0.9,
            raw_content: Some("<html><body>Raw HTML</body></html>".to_string()),
        };

        let md = result.to_markdown(true);
        assert!(md.contains("<details>"));
        assert!(md.contains("Raw HTML"));
    }

    #[test]
    fn test_tavily_result_raw_content_truncation() {
        let long_html = "x".repeat(3000);
        let result = TavilyResult {
            title: "Test".to_string(),
            url: "https://example.com".to_string(),
            content: "Content".to_string(),
            score: 0.9,
            raw_content: Some(long_html),
        };

        let md = result.to_markdown(true);
        assert!(md.contains("...[truncated]"));
        assert!(md.len() < 3500); // Should be truncated
    }

    #[test]
    fn test_from_env_missing_key() {
        // Ensure the env var is not set for this test
        std::env::remove_var("TAVILY_API_KEY");
        let result = TavilySearchTool::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let tool = TavilySearchTool::new("test-key")
            .with_timeout(Duration::from_secs(60))
            .with_max_retries(5);

        assert_eq!(tool.timeout, Duration::from_secs(60));
        assert_eq!(tool.max_retries, 5);
    }

    // ==================== Error Tests ====================

    #[test]
    fn test_tavily_error_retryable() {
        assert!(TavilyError::Timeout.is_retryable());
        assert!(TavilyError::RateLimited.is_retryable());
        assert!(TavilyError::ServerError(500, "".to_string()).is_retryable());
        assert!(TavilyError::Connection("failed".to_string()).is_retryable());

        assert!(!TavilyError::Unauthorized.is_retryable());
        assert!(!TavilyError::BadRequest("invalid".to_string()).is_retryable());
    }

    #[test]
    fn test_tavily_error_to_middleware_error() {
        let error: MiddlewareError = TavilyError::RateLimited.into();
        assert!(error.to_string().contains("Rate limited"));
    }
}

/// HTTP Integration tests with mocked server
#[cfg(test)]
mod http_tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Create a TavilySearchTool that uses a custom base URL (for mocking)
    struct MockableTavilyTool {
        api_key: String,
        client: Client,
        timeout: Duration,
        max_retries: u32,
        base_url: String,
    }

    impl MockableTavilyTool {
        fn new(api_key: impl Into<String>, base_url: String) -> Self {
            Self {
                api_key: api_key.into(),
                client: Client::new(),
                timeout: Duration::from_secs(5),
                max_retries: 0, // No retries for most tests
                base_url,
            }
        }

        fn with_retries(mut self, retries: u32) -> Self {
            self.max_retries = retries;
            self
        }

        async fn execute_request(
            &self,
            request: &TavilyRequest,
        ) -> Result<TavilyResponse, TavilyError> {
            let mut last_error = TavilyError::Unknown("No attempts made".to_string());

            for attempt in 0..=self.max_retries {
                if attempt > 0 {
                    let delay = Duration::from_millis(100 * 2u64.pow(attempt - 1));
                    tokio::time::sleep(delay).await;
                }

                match self.execute_single(request).await {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        if !e.is_retryable() {
                            return Err(e);
                        }
                        last_error = e;
                    }
                }
            }

            Err(last_error)
        }

        async fn execute_single(&self, request: &TavilyRequest) -> Result<TavilyResponse, TavilyError> {
            let response = self
                .client
                .post(&format!("{}/search", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .timeout(self.timeout)
                .json(request)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        TavilyError::Timeout
                    } else if e.is_connect() {
                        TavilyError::Connection(e.to_string())
                    } else {
                        TavilyError::Network(e.to_string())
                    }
                })?;

            let status = response.status();

            if status.is_success() {
                return response
                    .json()
                    .await
                    .map_err(|e| TavilyError::ParseError(e.to_string()));
            }

            let error_text = response.text().await.unwrap_or_default();
            match status.as_u16() {
                401 => Err(TavilyError::Unauthorized),
                429 => Err(TavilyError::RateLimited),
                400 => Err(TavilyError::BadRequest(error_text)),
                500..=599 => Err(TavilyError::ServerError(status.as_u16(), error_text)),
                _ => Err(TavilyError::HttpError(status.as_u16(), error_text)),
            }
        }
    }

    fn sample_success_response() -> serde_json::Value {
        serde_json::json!({
            "answer": "Rust is a systems programming language.",
            "results": [
                {
                    "title": "Rust Programming Language",
                    "url": "https://rust-lang.org",
                    "content": "Rust is a systems programming language focused on safety.",
                    "score": 0.95,
                    "raw_content": null
                },
                {
                    "title": "Learn Rust",
                    "url": "https://doc.rust-lang.org/book/",
                    "content": "The Rust Programming Language book.",
                    "score": 0.88,
                    "raw_content": null
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_http_successful_search() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .and(header("Authorization", "Bearer test-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_success_response()))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-api-key", mock_server.uri());
        let request = TavilyRequest {
            query: "Rust programming".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: true,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.answer.is_some());
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].title, "Rust Programming Language");
    }

    #[tokio::test]
    async fn test_http_unauthorized_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Invalid API key"))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("bad-key", mock_server.uri());
        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::Unauthorized)));
    }

    #[tokio::test]
    async fn test_http_rate_limited() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(429).set_body_string("Rate limit exceeded"))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri());
        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::RateLimited)));
    }

    #[tokio::test]
    async fn test_http_server_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal server error"))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri());
        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::ServerError(500, _))));
    }

    #[tokio::test]
    async fn test_http_bad_request() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(400).set_body_string("Invalid query parameter"))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri());
        let request = TavilyRequest {
            query: "".to_string(), // Empty query
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::BadRequest(_))));
    }

    #[tokio::test]
    async fn test_http_retry_on_server_error() {
        let mock_server = MockServer::start().await;

        // First two calls fail with 500, third succeeds
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_success_response()))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri())
            .with_retries(3);

        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        // Should succeed after retries
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_http_no_retry_on_unauthorized() {
        let mock_server = MockServer::start().await;

        // 401 should not trigger retry
        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1) // Should only be called once
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("bad-key", mock_server.uri())
            .with_retries(3);

        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::Unauthorized)));
    }

    #[tokio::test]
    async fn test_http_empty_results() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "answer": null,
                "results": []
            })))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri());
        let request = TavilyRequest {
            query: "nonexistent topic xyz123".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.answer.is_none());
        assert!(response.results.is_empty());
    }

    #[tokio::test]
    async fn test_http_malformed_response() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
            .mount(&mock_server)
            .await;

        let tool = MockableTavilyTool::new("test-key", mock_server.uri());
        let request = TavilyRequest {
            query: "test".to_string(),
            max_results: 5,
            search_depth: "basic".to_string(),
            topic: "general".to_string(),
            include_answer: false,
            include_raw_content: false,
        };

        let result = tool.execute_request(&request).await;

        assert!(matches!(result, Err(TavilyError::ParseError(_))));
    }
}
