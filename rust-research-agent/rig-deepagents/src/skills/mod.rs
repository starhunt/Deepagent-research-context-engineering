//! Skills system with progressive disclosure
//!
//! Python Reference: research_agent/skills/middleware.py
//!
//! Skills are reusable agent capabilities defined in markdown files
//! with YAML frontmatter. The system implements progressive disclosure:
//!
//! 1. At session start: Only skill summaries (name + description) are
//!    injected into the system prompt
//! 2. When skill is invoked: Full skill content is loaded on-demand
//!
//! # Directory Structure
//!
//! ```text
//! ~/.claude/skills/           # User-level skills
//! ├── academic-search/
//! │   └── SKILL.md
//! └── data-synthesis/
//!     └── SKILL.md
//!
//! {PROJECT_ROOT}/skills/      # Project-level skills (higher priority)
//! ├── report-writing/
//! │   └── SKILL.md
//! └── skill-creator/
//!     └── SKILL.md
//! ```
//!
//! # SKILL.md Format
//!
//! ```markdown
//! ---
//! name: skill-name
//! description: Brief description for system prompt
//! tags: [optional, categorization]
//! ---
//! # Full Skill Instructions
//!
//! Detailed instructions loaded on-demand...
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use rig_deepagents::skills::{SkillLoader, SkillsMiddleware};
//!
//! // Create and initialize loader
//! let loader = SkillLoader::from_env();
//! loader.initialize().await?;
//!
//! // Create middleware
//! let middleware = SkillsMiddleware::new(loader);
//!
//! // List available skills (metadata only)
//! for (meta, source) in loader.list_skills().await {
//!     println!("{}: {} ({})", meta.name, meta.description, source.as_str());
//! }
//!
//! // Load full skill content on demand
//! let skill = loader.load_skill("academic-search").await?;
//! println!("{}", skill.full_content());
//! ```

pub mod types;
pub mod loader;
pub mod middleware;

pub use types::{SkillMetadata, SkillContent, SkillSource};
pub use loader::SkillLoader;
pub use middleware::SkillsMiddleware;
