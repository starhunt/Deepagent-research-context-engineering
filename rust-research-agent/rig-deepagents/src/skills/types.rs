//! Skill type definitions
//!
//! Python Reference: research_agent/skills/middleware.py
//!
//! Skills are markdown files with YAML frontmatter that define
//! reusable agent capabilities with progressive disclosure.

use serde::{Deserialize, Serialize};

/// Skill metadata from YAML frontmatter
///
/// # Example SKILL.md:
/// ```markdown
/// ---
/// name: academic-search
/// description: Search arXiv papers with structured output
/// ---
/// [Full skill instructions...]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SkillMetadata {
    /// Unique skill name (kebab-case, e.g., "academic-search")
    pub name: String,

    /// Human-readable description for agent context
    pub description: String,

    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Optional version string
    #[serde(default)]
    pub version: Option<String>,

    /// Optional author information
    #[serde(default)]
    pub author: Option<String>,
}

/// Complete skill content including metadata and body
#[derive(Debug, Clone)]
pub struct SkillContent {
    /// Parsed metadata from frontmatter
    pub metadata: SkillMetadata,

    /// Full markdown body (after frontmatter)
    pub body: String,

    /// Source file path (for error reporting)
    pub source_path: String,
}

impl SkillContent {
    /// Create a new SkillContent
    pub fn new(metadata: SkillMetadata, body: String, source_path: String) -> Self {
        Self {
            metadata,
            body,
            source_path,
        }
    }

    /// Get the skill name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get the skill description
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Format skill summary for system prompt injection
    /// (Progressive disclosure: only metadata shown initially)
    pub fn summary(&self) -> String {
        format!("- **{}**: {}", self.metadata.name, self.metadata.description)
    }

    /// Get full content for when skill is invoked
    pub fn full_content(&self) -> String {
        format!(
            "# Skill: {}\n\n{}\n\n{}",
            self.metadata.name, self.metadata.description, self.body
        )
    }
}

/// Skill source location
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillSource {
    /// User-level skills (~/.claude/skills/)
    User,
    /// Project-level skills (PROJECT_ROOT/skills/)
    Project,
}

impl SkillSource {
    /// Get display name for source
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_metadata_deserialize() {
        let yaml = r#"
name: test-skill
description: A test skill for unit testing
tags:
  - testing
  - example
version: "1.0"
"#;
        let metadata: SkillMetadata = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(metadata.name, "test-skill");
        assert_eq!(metadata.description, "A test skill for unit testing");
        assert_eq!(metadata.tags, vec!["testing", "example"]);
        assert_eq!(metadata.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_skill_metadata_minimal() {
        let yaml = r#"
name: minimal
description: Minimal skill
"#;
        let metadata: SkillMetadata = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(metadata.name, "minimal");
        assert!(metadata.tags.is_empty());
        assert!(metadata.version.is_none());
    }

    #[test]
    fn test_skill_content_summary() {
        let metadata = SkillMetadata {
            name: "report-writing".to_string(),
            description: "Generate structured reports".to_string(),
            tags: vec![],
            version: None,
            author: None,
        };
        let content = SkillContent::new(
            metadata,
            "# Instructions\n\nWrite reports...".to_string(),
            "skills/report-writing/SKILL.md".to_string(),
        );

        assert_eq!(
            content.summary(),
            "- **report-writing**: Generate structured reports"
        );
    }

    #[test]
    fn test_skill_content_full() {
        let metadata = SkillMetadata {
            name: "test".to_string(),
            description: "Test description".to_string(),
            tags: vec![],
            version: None,
            author: None,
        };
        let content = SkillContent::new(
            metadata,
            "Body content here".to_string(),
            "test.md".to_string(),
        );

        let full = content.full_content();
        assert!(full.contains("# Skill: test"));
        assert!(full.contains("Test description"));
        assert!(full.contains("Body content here"));
    }

    #[test]
    fn test_skill_source() {
        assert_eq!(SkillSource::User.as_str(), "user");
        assert_eq!(SkillSource::Project.as_str(), "project");
    }
}
