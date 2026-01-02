//! Skills middleware implementing AgentMiddleware trait
//!
//! Injects skill summaries into system prompt and provides
//! the `use_skill` tool for on-demand skill loading.

use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::loader::SkillLoader;
use super::types::{SkillMetadata, SkillSource};
use crate::error::MiddlewareError;
use crate::middleware::{AgentMiddleware, DynTool, Tool, ToolDefinition};
use crate::runtime::ToolRuntime;

/// Skills middleware for progressive skill disclosure
///
/// Implements the progressive disclosure pattern:
/// 1. On initialization: Loads skill metadata (fast)
/// 2. In system prompt: Injects skill summaries
/// 3. On tool call: Loads full skill content (lazy)
pub struct SkillsMiddleware {
    loader: Arc<SkillLoader>,
    /// Pre-computed skill summaries for sync access in modify_system_prompt
    cached_summaries: Arc<RwLock<Option<String>>>,
}

impl SkillsMiddleware {
    /// Create new middleware with a skill loader
    ///
    /// Note: Call `refresh_cache()` after creating to populate the prompt cache.
    pub fn new(loader: Arc<SkillLoader>) -> Self {
        Self {
            loader,
            cached_summaries: Arc::new(RwLock::new(None)),
        }
    }

    /// Create middleware with default loader (from environment)
    pub async fn from_env() -> Result<Self, MiddlewareError> {
        let loader = Arc::new(SkillLoader::from_env());
        loader.initialize().await?;
        let middleware = Self::new(loader);
        middleware.refresh_cache().await;
        Ok(middleware)
    }

    /// Create middleware with pre-cached summaries (for testing or pre-initialization)
    pub async fn with_loader(loader: Arc<SkillLoader>) -> Self {
        let middleware = Self::new(loader);
        middleware.refresh_cache().await;
        middleware
    }

    /// Get the underlying loader
    pub fn loader(&self) -> &SkillLoader {
        &self.loader
    }

    /// Refresh the cached skill summaries
    pub async fn refresh_cache(&self) {
        let skills = self.loader.list_skills().await;
        let summary = Self::build_skill_section(&skills);

        let mut cache = self.cached_summaries.write().await;
        *cache = summary;
    }

    /// Build skill section for system prompt
    fn build_skill_section(skills: &[(SkillMetadata, SkillSource)]) -> Option<String> {
        if skills.is_empty() {
            return None;
        }

        Some(format!(
            r#"

## Available Skills

You have access to the following skills. Use the `use_skill` tool to load full instructions:

{}

To use a skill, call: `use_skill({{"name": "skill-name"}})`
"#,
            skills
                .iter()
                .map(|(meta, source)| format!(
                    "- **{}** ({}): {}",
                    meta.name,
                    source.as_str(),
                    meta.description
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }

    /// Get cached summaries synchronously (for modify_system_prompt)
    fn get_cached_summaries_sync(&self) -> Option<String> {
        // Try to read without blocking using try_read
        self.cached_summaries
            .try_read()
            .ok()
            .and_then(|guard| guard.clone())
    }
}

#[async_trait]
impl AgentMiddleware for SkillsMiddleware {
    fn name(&self) -> &str {
        "skills"
    }

    fn tools(&self) -> Vec<DynTool> {
        vec![Arc::new(UseSkillTool {
            loader: Arc::clone(&self.loader),
        })]
    }

    fn modify_system_prompt(&self, prompt: String) -> String {
        // Use pre-cached summaries (populated by refresh_cache)
        match self.get_cached_summaries_sync() {
            Some(section) => format!("{}{}", prompt, section),
            None => prompt,
        }
    }
}

/// Tool for loading skill content on-demand
struct UseSkillTool {
    loader: Arc<SkillLoader>,
}

#[derive(Debug, Deserialize)]
struct UseSkillArgs {
    name: String,
}

#[async_trait]
impl Tool for UseSkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "use_skill".to_string(),
            description: "Load full instructions for a skill. Use this when you need to apply a specific skill's methodology.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load (e.g., 'academic-search', 'report-writing')"
                    }
                },
                "required": ["name"]
            }),
        }
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        _runtime: &ToolRuntime,
    ) -> Result<String, MiddlewareError> {
        let args: UseSkillArgs = serde_json::from_value(args)
            .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid arguments: {}", e)))?;

        let skill = self.loader.load_skill(&args.name).await?;

        Ok(skill.full_content())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MemoryBackend;
    use crate::state::AgentState;
    use std::path::PathBuf;

    /// Creates a test loader with temporary skill files.
    /// Returns both the loader and the TempDir to keep it alive during the test.
    /// The TempDir is automatically cleaned up when dropped at the end of the test.
    async fn create_test_loader() -> (Arc<SkillLoader>, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let skill_dir = temp_dir.path().join("test-skill");
        std::fs::create_dir_all(&skill_dir).unwrap();

        std::fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: test-skill
description: A test skill for unit testing
---
# Test Skill

This is the full skill content.

## Instructions

1. Do this
2. Then that
"#,
        )
        .unwrap();

        // Create a second skill
        let skill_dir2 = temp_dir.path().join("another-skill");
        std::fs::create_dir_all(&skill_dir2).unwrap();

        std::fs::write(
            skill_dir2.join("SKILL.md"),
            r#"---
name: another-skill
description: Another test skill
---
# Another Skill

Different content here.
"#,
        )
        .unwrap();

        let path = temp_dir.path().to_path_buf();
        let loader = Arc::new(SkillLoader::new(None, Some(path)));
        loader.initialize().await.unwrap();

        // Return both loader and temp_dir - caller must keep temp_dir alive
        (loader, temp_dir)
    }

    #[tokio::test]
    async fn test_middleware_provides_tool() {
        let (loader, _temp_dir) = create_test_loader().await;
        let middleware = SkillsMiddleware::with_loader(loader).await;

        let tools = middleware.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition().name, "use_skill");
    }

    #[tokio::test]
    async fn test_middleware_modifies_prompt() {
        let (loader, _temp_dir) = create_test_loader().await;
        let middleware = SkillsMiddleware::with_loader(loader).await;

        let base_prompt = "You are a helpful assistant.";
        let modified = middleware.modify_system_prompt(base_prompt.to_string());

        assert!(modified.contains("You are a helpful assistant"));
        assert!(modified.contains("Available Skills"));
        assert!(modified.contains("test-skill"));
        assert!(modified.contains("another-skill"));
        assert!(modified.contains("use_skill"));
    }

    #[tokio::test]
    async fn test_use_skill_tool() {
        let (loader, _temp_dir) = create_test_loader().await;
        let tool = UseSkillTool {
            loader: Arc::clone(&loader),
        };

        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);

        let result = tool
            .execute(serde_json::json!({"name": "test-skill"}), &runtime)
            .await
            .unwrap();

        assert!(result.contains("# Skill: test-skill"));
        assert!(result.contains("This is the full skill content"));
        assert!(result.contains("1. Do this"));
    }

    #[tokio::test]
    async fn test_use_skill_not_found() {
        let loader = Arc::new(SkillLoader::new(None, None));
        loader.initialize().await.unwrap();

        let tool = UseSkillTool { loader };

        let backend = Arc::new(MemoryBackend::new());
        let state = AgentState::new();
        let runtime = ToolRuntime::new(state, backend);

        let result = tool
            .execute(serde_json::json!({"name": "nonexistent"}), &runtime)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_middleware_empty_skills() {
        let loader = Arc::new(SkillLoader::new(None, Some(PathBuf::from("/nonexistent"))));
        loader.initialize().await.unwrap();

        let middleware = SkillsMiddleware::with_loader(loader).await;

        let base_prompt = "Base prompt";
        let modified = middleware.modify_system_prompt(base_prompt.to_string());

        // No modification when no skills
        assert_eq!(modified, "Base prompt");
    }
}
