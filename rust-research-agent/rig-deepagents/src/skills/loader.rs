//! Skill loader with lazy loading and YAML frontmatter parsing
//!
//! Implements progressive disclosure pattern:
//! - `list_skills()`: Returns only metadata (fast, for system prompt)
//! - `load_skill()`: Returns full content on-demand (lazy)
//!
//! Skill files must be named SKILL.md and located in:
//! - User skills: ~/.claude/skills/{skill-name}/SKILL.md
//! - Project skills: {PROJECT_ROOT}/skills/{skill-name}/SKILL.md

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use super::types::{SkillContent, SkillMetadata, SkillSource};
use crate::error::MiddlewareError;

/// Type alias for metadata cache entry (metadata, file path, source)
type MetadataCacheEntry = (SkillMetadata, PathBuf, SkillSource);

/// Skill loader with caching support
pub struct SkillLoader {
    /// User skills directory (e.g., ~/.claude/skills)
    user_dir: Option<PathBuf>,

    /// Project skills directory (e.g., ./skills)
    project_dir: Option<PathBuf>,

    /// Cached metadata (loaded eagerly on init)
    metadata_cache: Arc<RwLock<HashMap<String, MetadataCacheEntry>>>,

    /// Cached full content (loaded lazily on demand)
    content_cache: Arc<RwLock<HashMap<String, SkillContent>>>,
}

impl SkillLoader {
    /// Create a new skill loader with specified directories
    pub fn new(user_dir: Option<PathBuf>, project_dir: Option<PathBuf>) -> Self {
        Self {
            user_dir,
            project_dir,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            content_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create loader from environment defaults
    ///
    /// - User: ~/.claude/skills
    /// - Project: ./skills (relative to current working directory)
    pub fn from_env() -> Self {
        let user_dir = dirs::home_dir().map(|h| h.join(".claude").join("skills"));
        let project_dir = Some(PathBuf::from("skills"));

        Self::new(user_dir, project_dir)
    }

    /// Create loader with specific project root
    pub fn with_project_root(project_root: &Path) -> Self {
        let user_dir = dirs::home_dir().map(|h| h.join(".claude").join("skills"));
        let project_dir = Some(project_root.join("skills"));

        Self::new(user_dir, project_dir)
    }

    /// Scan directories and populate metadata cache
    pub async fn initialize(&self) -> Result<(), MiddlewareError> {
        let mut cache = self.metadata_cache.write().await;
        cache.clear();

        // Scan user skills first (lower priority)
        if let Some(user_dir) = &self.user_dir {
            if user_dir.exists() {
                self.scan_directory(user_dir, SkillSource::User, &mut cache)
                    .await?;
            }
        }

        // Scan project skills (higher priority, can override user skills)
        if let Some(project_dir) = &self.project_dir {
            if project_dir.exists() {
                self.scan_directory(project_dir, SkillSource::Project, &mut cache)
                    .await?;
            }
        }

        debug!("Loaded {} skill metadata entries", cache.len());
        Ok(())
    }

    /// Scan a directory for SKILL.md files
    async fn scan_directory(
        &self,
        dir: &Path,
        source: SkillSource,
        cache: &mut HashMap<String, (SkillMetadata, PathBuf, SkillSource)>,
    ) -> Result<(), MiddlewareError> {
        // Use tokio::fs for non-blocking directory reading
        let mut entries = match tokio::fs::read_dir(dir).await {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to read skills directory {:?}: {}", dir, e);
                return Ok(());
            }
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            // Use tokio::fs::metadata for non-blocking check
            if let Ok(metadata) = tokio::fs::metadata(&path).await {
                if metadata.is_dir() {
                    let skill_file = path.join("SKILL.md");
                    // Check if SKILL.md exists using async metadata
                    if tokio::fs::metadata(&skill_file).await.is_ok() {
                        match self.parse_metadata(&skill_file).await {
                            Ok(skill_meta) => {
                                debug!(
                                    "Loaded skill metadata: {} from {:?} ({})",
                                    skill_meta.name,
                                    skill_file,
                                    source.as_str()
                                );
                                cache.insert(skill_meta.name.clone(), (skill_meta, skill_file, source));
                            }
                            Err(e) => {
                                warn!("Failed to parse skill {:?}: {}", skill_file, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse only metadata from YAML frontmatter (fast)
    async fn parse_metadata(&self, path: &Path) -> Result<SkillMetadata, MiddlewareError> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| MiddlewareError::ToolExecution(format!("Failed to read skill: {}", e)))?;

        parse_frontmatter(&content)
    }

    /// List all available skills (metadata only)
    ///
    /// Skills are sorted alphabetically by name for deterministic ordering
    /// in system prompts and reproducible behavior.
    pub async fn list_skills(&self) -> Vec<(SkillMetadata, SkillSource)> {
        let cache = self.metadata_cache.read().await;
        let mut skills: Vec<_> = cache
            .values()
            .map(|(meta, _, source)| (meta.clone(), *source))
            .collect();

        // Sort by skill name for deterministic ordering
        skills.sort_by(|a, b| a.0.name.cmp(&b.0.name));
        skills
    }

    /// Get skill metadata by name
    pub async fn get_metadata(&self, name: &str) -> Option<SkillMetadata> {
        let cache = self.metadata_cache.read().await;
        cache.get(name).map(|(meta, _, _)| meta.clone())
    }

    /// Load full skill content (lazy, cached)
    pub async fn load_skill(&self, name: &str) -> Result<SkillContent, MiddlewareError> {
        // Check content cache first
        {
            let cache = self.content_cache.read().await;
            if let Some(content) = cache.get(name) {
                return Ok(content.clone());
            }
        }

        // Get path from metadata cache
        let (metadata, path) = {
            let cache = self.metadata_cache.read().await;
            match cache.get(name) {
                Some((meta, path, _)) => (meta.clone(), path.clone()),
                None => {
                    return Err(MiddlewareError::ToolExecution(format!(
                        "Skill not found: {}",
                        name
                    )))
                }
            }
        };

        // Load and parse full content
        let raw_content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| MiddlewareError::ToolExecution(format!("Failed to read skill: {}", e)))?;

        let body = parse_body(&raw_content);
        let content = SkillContent::new(metadata, body, path.to_string_lossy().to_string());

        // Cache the content
        {
            let mut cache = self.content_cache.write().await;
            cache.insert(name.to_string(), content.clone());
        }

        Ok(content)
    }

    /// Refresh skill cache (re-scan directories)
    pub async fn refresh(&self) -> Result<(), MiddlewareError> {
        // Clear content cache
        {
            let mut cache = self.content_cache.write().await;
            cache.clear();
        }

        // Re-initialize metadata
        self.initialize().await
    }
}

/// Parse YAML frontmatter from markdown content
///
/// Expected format:
/// ```markdown
/// ---
/// name: skill-name
/// description: Skill description
/// ---
/// Body content here...
/// ```
///
/// The closing `---` must be on its own line (possibly with trailing whitespace).
fn parse_frontmatter(content: &str) -> Result<SkillMetadata, MiddlewareError> {
    let content = content.trim();

    if !content.starts_with("---") {
        return Err(MiddlewareError::ToolExecution(
            "Skill file must start with YAML frontmatter (---)".to_string(),
        ));
    }

    // Skip the opening --- and any trailing content on that line
    let rest = &content[3..];
    let rest = rest.strip_prefix('\n').unwrap_or(rest);

    // Find the closing --- on its own line
    // Look for \n--- followed by newline, whitespace+newline, or end of string
    let end_idx = find_closing_frontmatter(rest)
        .ok_or_else(|| MiddlewareError::ToolExecution(
            "Missing closing --- in frontmatter (must be on its own line)".to_string()
        ))?;

    let yaml_str = &rest[..end_idx];

    serde_yaml::from_str(yaml_str.trim())
        .map_err(|e| MiddlewareError::ToolExecution(format!("Invalid YAML frontmatter: {}", e)))
}

/// Find the position of the closing frontmatter delimiter
/// The closing `---` must be on its own line (with optional trailing whitespace)
fn find_closing_frontmatter(content: &str) -> Option<usize> {
    let mut pos = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "---" {
            return Some(pos);
        }
        pos += line.len() + 1; // +1 for newline
    }
    None
}

/// Extract body content (after frontmatter)
fn parse_body(content: &str) -> String {
    let content = content.trim();

    if !content.starts_with("---") {
        return content.to_string();
    }

    // Skip the opening ---
    let rest = &content[3..];
    let rest = rest.strip_prefix('\n').unwrap_or(rest);

    // Find closing frontmatter
    if let Some(end_idx) = find_closing_frontmatter(rest) {
        // Skip past the closing --- line
        let after_yaml = &rest[end_idx..];
        // Find the end of the --- line
        if let Some(newline_pos) = after_yaml.find('\n') {
            after_yaml[newline_pos + 1..].trim().to_string()
        } else {
            // No content after frontmatter
            String::new()
        }
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter_valid() {
        let content = r#"---
name: test-skill
description: A test skill
tags:
  - testing
---
# Body Content

This is the skill body.
"#;

        let metadata = parse_frontmatter(content).unwrap();
        assert_eq!(metadata.name, "test-skill");
        assert_eq!(metadata.description, "A test skill");
        assert_eq!(metadata.tags, vec!["testing"]);
    }

    #[test]
    fn test_parse_frontmatter_minimal() {
        let content = r#"---
name: minimal
description: Minimal skill
---
Body here
"#;

        let metadata = parse_frontmatter(content).unwrap();
        assert_eq!(metadata.name, "minimal");
        assert!(metadata.tags.is_empty());
    }

    #[test]
    fn test_parse_frontmatter_missing_start() {
        let content = "No frontmatter here";
        let result = parse_frontmatter(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_frontmatter_missing_end() {
        let content = r#"---
name: incomplete
description: Missing closing
"#;
        let result = parse_frontmatter(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_body() {
        let content = r#"---
name: test
description: Test
---
# Heading

Body content here.

More content.
"#;

        let body = parse_body(content);
        assert!(body.starts_with("# Heading"));
        assert!(body.contains("Body content here."));
        assert!(body.contains("More content."));
    }

    #[test]
    fn test_parse_body_no_frontmatter() {
        let content = "Just plain content";
        let body = parse_body(content);
        assert_eq!(body, "Just plain content");
    }

    #[test]
    fn test_parse_frontmatter_with_dashes_in_yaml() {
        // Edge case: YAML content containing --- (but not on its own line)
        let content = r#"---
name: complex-skill
description: A skill with dashes---in the description
---
# Body

Content here with --- in text.
"#;

        let metadata = parse_frontmatter(content).unwrap();
        assert_eq!(metadata.name, "complex-skill");
        assert_eq!(metadata.description, "A skill with dashes---in the description");

        let body = parse_body(content);
        assert!(body.contains("Content here with --- in text"));
    }

    #[test]
    fn test_parse_frontmatter_with_trailing_whitespace() {
        // Edge case: closing --- with trailing whitespace
        let content = "---\nname: test\ndescription: Test\n---   \nBody";

        let metadata = parse_frontmatter(content).unwrap();
        assert_eq!(metadata.name, "test");

        let body = parse_body(content);
        assert_eq!(body, "Body");
    }

    #[tokio::test]
    async fn test_skill_loader_empty_dirs() {
        let loader = SkillLoader::new(None, None);
        loader.initialize().await.unwrap();

        let skills = loader.list_skills().await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_skill_loader_with_temp_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let skill_dir = temp_dir.path().join("test-skill");
        std::fs::create_dir_all(&skill_dir).unwrap();

        let skill_file = skill_dir.join("SKILL.md");
        std::fs::write(
            &skill_file,
            r#"---
name: test-skill
description: A test skill for testing
---
# Test Skill

This is the test skill body.
"#,
        )
        .unwrap();

        let loader = SkillLoader::new(None, Some(temp_dir.path().to_path_buf()));
        loader.initialize().await.unwrap();

        // Test list_skills
        let skills = loader.list_skills().await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].0.name, "test-skill");
        assert_eq!(skills[0].1, SkillSource::Project);

        // Test get_metadata
        let metadata = loader.get_metadata("test-skill").await.unwrap();
        assert_eq!(metadata.description, "A test skill for testing");

        // Test load_skill (lazy load)
        let content = loader.load_skill("test-skill").await.unwrap();
        assert!(content.body.contains("This is the test skill body"));

        // Test cache hit (load again)
        let content2 = loader.load_skill("test-skill").await.unwrap();
        assert_eq!(content.body, content2.body);

        // Test skill not found
        let result = loader.load_skill("nonexistent").await;
        assert!(result.is_err());
    }
}
