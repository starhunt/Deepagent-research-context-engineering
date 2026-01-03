//! ToolResult eviction helpers for oversized tool outputs.

use std::collections::HashMap;

use crate::backends::Backend;
use crate::middleware::{StateUpdate, ToolResult};
use crate::state::FileData;

pub(crate) const DEFAULT_TOOL_RESULT_TOKEN_LIMIT: usize = 20000;
const TOOL_RESULT_EVICT_CHAR_MULTIPLIER: usize = 4;
const LARGE_TOOL_RESULT_DIR: &str = "/large_tool_results";
const TOOL_RESULT_SAMPLE_LINES: usize = 10;
const TOOL_RESULT_SAMPLE_LINE_LEN: usize = 1000;

#[derive(Debug, Clone)]
pub(crate) struct ToolResultEvictor {
    token_limit: Option<usize>,
}

impl ToolResultEvictor {
    pub(crate) fn new(token_limit: Option<usize>) -> Self {
        Self { token_limit }
    }

    pub(crate) async fn maybe_evict(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        result: ToolResult,
        backend: &dyn Backend,
    ) -> ToolResult {
        let Some(limit) = self.token_limit else {
            return result;
        };
        if should_skip_tool_result_eviction(tool_name) {
            return result;
        }

        let threshold = limit.saturating_mul(TOOL_RESULT_EVICT_CHAR_MULTIPLIER);
        if result.message.len() <= threshold {
            return result;
        }

        let sanitized_id = sanitize_tool_call_id(tool_call_id);
        let file_path = format!("{}/{}", LARGE_TOOL_RESULT_DIR, sanitized_id);
        let write_result = match backend.write(&file_path, &result.message).await {
            Ok(write_result) => write_result,
            Err(err) => {
                tracing::warn!(
                    tool_call_id = %tool_call_id,
                    error = %err,
                    "Failed to evict large tool result"
                );
                return result;
            }
        };

        if !write_result.is_ok() {
            tracing::warn!(
                tool_call_id = %tool_call_id,
                error = ?write_result.error,
                "Failed to evict large tool result"
            );
            return result;
        }

        let sample = format_content_sample(&result.message);
        let message = format!(
            "Tool result was too large. The result of tool call {} was saved to: {}\n\
read_file can be used to read the file with offset/limit for pagination.\n\n\
First {} lines:\n{}",
            tool_call_id,
            file_path,
            TOOL_RESULT_SAMPLE_LINES,
            sample
        );

        let mut updates = result.updates.clone();
        if let Some(files_update) = write_result.files_update {
            let files: HashMap<String, Option<FileData>> = files_update
                .into_iter()
                .map(|(path, data)| (path, Some(data)))
                .collect();
            updates.push(StateUpdate::UpdateFiles(files));
        }

        ToolResult { message, updates }
    }
}

impl Default for ToolResultEvictor {
    fn default() -> Self {
        Self::new(Some(DEFAULT_TOOL_RESULT_TOKEN_LIMIT))
    }
}

fn should_skip_tool_result_eviction(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "ls" | "read_file" | "write_file" | "edit_file" | "glob" | "grep"
    )
}

fn sanitize_tool_call_id(id: &str) -> String {
    let mut sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    while sanitized.starts_with('_') {
        sanitized.remove(0);
    }
    while sanitized.ends_with('_') {
        sanitized.pop();
    }

    if sanitized.is_empty() {
        "tool_call".to_string()
    } else {
        sanitized
    }
}

fn format_content_sample(content: &str) -> String {
    let mut lines = Vec::new();
    for (index, line) in content.lines().take(TOOL_RESULT_SAMPLE_LINES).enumerate() {
        let trimmed = if line.len() > TOOL_RESULT_SAMPLE_LINE_LEN {
            let mut part = line.to_string();
            part.truncate(TOOL_RESULT_SAMPLE_LINE_LEN);
            part
        } else {
            line.to_string()
        };
        lines.push(format!("{}\t{}", index + 1, trimmed));
    }

    if lines.is_empty() {
        "(empty)".to_string()
    } else {
        lines.join("\n")
    }
}
