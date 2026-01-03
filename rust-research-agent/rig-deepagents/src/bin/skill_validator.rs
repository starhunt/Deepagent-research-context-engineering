//! Skill Validator CLI Tool
//!
//! Validates skill files (SKILL.md) for correct YAML frontmatter format.
//!
//! # Usage
//!
//! ```bash
//! # Validate a single skill file
//! skill-validator path/to/SKILL.md
//!
//! # Validate all skills in a directory
//! skill-validator --dir path/to/skills/
//!
//! # Strict mode (warnings become errors)
//! skill-validator --strict path/to/SKILL.md
//!
//! # JSON output for programmatic use
//! skill-validator --json path/to/SKILL.md
//! ```
//!
//! # Exit Codes
//!
//! - 0: All validations passed
//! - 1: Validation errors found
//! - 2: CLI usage error

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Skill file validator for YAML frontmatter
#[derive(Parser, Debug)]
#[command(name = "skill-validator")]
#[command(about = "Validates skill SKILL.md files for correct YAML frontmatter format")]
#[command(version)]
struct Args {
    /// Path to a SKILL.md file or directory containing skills
    #[arg(required_unless_present = "dir")]
    path: Option<PathBuf>,

    /// Validate all skills in a directory (recursively)
    #[arg(short, long)]
    dir: Option<PathBuf>,

    /// Strict mode: treat warnings as errors
    #[arg(short, long)]
    strict: bool,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Quiet mode: only output on errors
    #[arg(short, long)]
    quiet: bool,
}

/// Validation result for a single skill file
#[derive(Debug, Serialize)]
struct ValidationResult {
    path: String,
    valid: bool,
    errors: Vec<ValidationIssue>,
    warnings: Vec<ValidationIssue>,
}

/// A validation issue (error or warning)
#[derive(Debug, Serialize)]
struct ValidationIssue {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<usize>,
}

/// Skill metadata from YAML frontmatter (matches types.rs)
/// Fields are parsed for validation completeness even if not all are used
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SkillMetadata {
    name: Option<String>,
    description: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    version: Option<String>,
    author: Option<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    // Collect files to validate
    let files = match collect_files(&args) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("Error: {}", e);
            return ExitCode::from(2);
        }
    };

    if files.is_empty() {
        if !args.quiet {
            eprintln!("No SKILL.md files found");
        }
        return ExitCode::from(2);
    }

    // Validate all files
    let results: Vec<ValidationResult> = files
        .iter()
        .map(|path| validate_skill_file(path))
        .collect();

    // Count issues
    let total_errors: usize = results.iter().map(|r| r.errors.len()).sum();
    let total_warnings: usize = results.iter().map(|r| r.warnings.len()).sum();

    // Output results
    if args.json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    } else {
        for result in &results {
            print_result(result, args.quiet);
        }

        if !args.quiet {
            println!();
            println!(
                "Validated {} file(s): {} error(s), {} warning(s)",
                results.len(),
                total_errors,
                total_warnings
            );
        }
    }

    // Determine exit code
    if total_errors > 0 || (args.strict && total_warnings > 0) {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}

/// Collect SKILL.md files to validate
fn collect_files(args: &Args) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    if let Some(dir) = &args.dir {
        collect_skills_in_dir(dir, &mut files)?;
    }

    if let Some(path) = &args.path {
        if path.is_dir() {
            collect_skills_in_dir(path, &mut files)?;
        } else if path.exists() {
            files.push(path.clone());
        } else {
            return Err(format!("Path does not exist: {}", path.display()));
        }
    }

    Ok(files)
}

/// Recursively find SKILL.md files in a directory
/// Each skill is expected to be in its own directory: skills/{skill-name}/SKILL.md
fn collect_skills_in_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("Not a directory: {}", dir.display()));
    }

    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Look for SKILL.md in subdirectory
            let skill_file = path.join("SKILL.md");
            if skill_file.exists() {
                files.push(skill_file);
                // Don't recurse further once we find a skill directory
            } else {
                // Only recurse into subdirectories that don't have SKILL.md
                // (e.g., for nested skill organization)
                collect_skills_in_dir(&path, files)?;
            }
        } else if path.file_name().map(|n| n == "SKILL.md").unwrap_or(false) {
            // Handle case where SKILL.md is directly in the scanned directory
            files.push(path);
        }
    }

    Ok(())
}

/// Validate a single SKILL.md file
fn validate_skill_file(path: &Path) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Read file content
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            errors.push(ValidationIssue {
                code: "E001".to_string(),
                message: format!("Failed to read file: {}", e),
                line: None,
            });
            return ValidationResult {
                path: path.display().to_string(),
                valid: false,
                errors,
                warnings,
            };
        }
    };

    // Check for frontmatter markers
    let content = content.trim();
    if !content.starts_with("---") {
        errors.push(ValidationIssue {
            code: "E002".to_string(),
            message: "File must start with YAML frontmatter (---)".to_string(),
            line: Some(1),
        });
        return ValidationResult {
            path: path.display().to_string(),
            valid: false,
            errors,
            warnings,
        };
    }

    // Find closing frontmatter
    let rest = &content[3..];
    let rest = rest.strip_prefix('\n').unwrap_or(rest);

    let end_idx = find_closing_frontmatter(rest);
    if end_idx.is_none() {
        errors.push(ValidationIssue {
            code: "E003".to_string(),
            message: "Missing closing frontmatter delimiter (---)".to_string(),
            line: None,
        });
        return ValidationResult {
            path: path.display().to_string(),
            valid: false,
            errors,
            warnings,
        };
    }

    let yaml_str = &rest[..end_idx.unwrap()];

    // Parse YAML
    let metadata: SkillMetadata = match serde_yaml::from_str(yaml_str) {
        Ok(m) => m,
        Err(e) => {
            errors.push(ValidationIssue {
                code: "E004".to_string(),
                message: format!("Invalid YAML: {}", e),
                line: e.location().map(|l| l.line() + 1), // +1 for opening ---
            });
            return ValidationResult {
                path: path.display().to_string(),
                valid: false,
                errors,
                warnings,
            };
        }
    };

    // Validate required fields
    if metadata.name.is_none() {
        errors.push(ValidationIssue {
            code: "E005".to_string(),
            message: "Missing required field: name".to_string(),
            line: None,
        });
    }

    if metadata.description.is_none() {
        errors.push(ValidationIssue {
            code: "E006".to_string(),
            message: "Missing required field: description".to_string(),
            line: None,
        });
    }

    // Validate name format (kebab-case)
    if let Some(name) = &metadata.name {
        if !is_valid_kebab_case(name) {
            errors.push(ValidationIssue {
                code: "E007".to_string(),
                message: format!(
                    "Skill name '{}' must be kebab-case (lowercase, hyphens only)",
                    name
                ),
                line: None,
            });
        }

        // Check for reserved names
        if is_reserved_name(name) {
            errors.push(ValidationIssue {
                code: "E008".to_string(),
                message: format!("Skill name '{}' is reserved", name),
                line: None,
            });
        }
    }

    // Warnings
    if metadata.description.as_ref().map(|d| d.len()).unwrap_or(0) > 200 {
        warnings.push(ValidationIssue {
            code: "W001".to_string(),
            message: "Description exceeds recommended 200 characters".to_string(),
            line: None,
        });
    }

    if metadata.description.as_ref().map(|d| d.len()).unwrap_or(0) < 10 {
        warnings.push(ValidationIssue {
            code: "W002".to_string(),
            message: "Description is too short (< 10 characters)".to_string(),
            line: None,
        });
    }

    if metadata.version.is_none() {
        warnings.push(ValidationIssue {
            code: "W003".to_string(),
            message: "Consider adding a version field".to_string(),
            line: None,
        });
    }

    // Check body content
    let body = parse_body(content);
    if body.trim().is_empty() {
        warnings.push(ValidationIssue {
            code: "W004".to_string(),
            message: "Skill body is empty".to_string(),
            line: None,
        });
    }

    ValidationResult {
        path: path.display().to_string(),
        valid: errors.is_empty(),
        errors,
        warnings,
    }
}

/// Find the position of the closing frontmatter delimiter
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

    let rest = &content[3..];
    let rest = rest.strip_prefix('\n').unwrap_or(rest);

    if let Some(end_idx) = find_closing_frontmatter(rest) {
        let after_yaml = &rest[end_idx..];
        if let Some(newline_pos) = after_yaml.find('\n') {
            after_yaml[newline_pos + 1..].trim().to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    }
}

/// Check if a string is valid kebab-case
fn is_valid_kebab_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    // Must start and end with alphanumeric
    let chars: Vec<char> = s.chars().collect();
    if !chars.first().map(|c| c.is_ascii_lowercase()).unwrap_or(false) {
        return false;
    }
    if !chars.last().map(|c| c.is_ascii_alphanumeric()).unwrap_or(false) {
        return false;
    }

    // Only lowercase letters, digits, and hyphens
    // No consecutive hyphens
    let mut prev_hyphen = false;
    for c in chars {
        if c == '-' {
            if prev_hyphen {
                return false; // consecutive hyphens
            }
            prev_hyphen = true;
        } else if c.is_ascii_lowercase() || c.is_ascii_digit() {
            prev_hyphen = false;
        } else {
            return false; // invalid character
        }
    }

    true
}

/// Check if name is reserved
fn is_reserved_name(name: &str) -> bool {
    matches!(
        name,
        "help" | "version" | "list" | "all" | "default" | "none" | "test"
    )
}

/// Print validation result to console
fn print_result(result: &ValidationResult, quiet: bool) {
    if quiet && result.valid && result.warnings.is_empty() {
        return;
    }

    if result.valid {
        println!("\x1b[32m✓\x1b[0m {}", result.path);
    } else {
        println!("\x1b[31m✗\x1b[0m {}", result.path);
    }

    for error in &result.errors {
        let line_info = error
            .line
            .map(|l| format!(" (line {})", l))
            .unwrap_or_default();
        println!(
            "  \x1b[31merror[{}]\x1b[0m: {}{}",
            error.code, error.message, line_info
        );
    }

    for warning in &result.warnings {
        let line_info = warning
            .line
            .map(|l| format!(" (line {})", l))
            .unwrap_or_default();
        println!(
            "  \x1b[33mwarn[{}]\x1b[0m: {}{}",
            warning.code, warning.message, line_info
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_skill_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let skill_dir = dir.path().join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let skill_file = skill_dir.join("SKILL.md");
        let mut f = std::fs::File::create(&skill_file).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        skill_file
    }

    #[test]
    fn test_valid_skill() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "valid-skill",
            r#"---
name: valid-skill
description: A valid test skill for testing purposes
version: "1.0"
---
# Valid Skill

This skill does something useful.
"#,
        );

        let result = validate_skill_file(&path);
        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_missing_name() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "missing-name",
            r#"---
description: Has description but no name
---
Body content.
"#,
        );

        let result = validate_skill_file(&path);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "E005"));
    }

    #[test]
    fn test_missing_description() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "missing-desc",
            r#"---
name: missing-desc
---
Body content.
"#,
        );

        let result = validate_skill_file(&path);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "E006"));
    }

    #[test]
    fn test_invalid_kebab_case() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "invalid-case",
            r#"---
name: InvalidCase
description: Has invalid case name
---
Body content.
"#,
        );

        let result = validate_skill_file(&path);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "E007"));
    }

    #[test]
    fn test_missing_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(&dir, "no-frontmatter", "Just plain content without frontmatter");

        let result = validate_skill_file(&path);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "E002"));
    }

    #[test]
    fn test_unclosed_frontmatter() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "unclosed",
            r#"---
name: unclosed
description: Missing closing delimiter
"#,
        );

        let result = validate_skill_file(&path);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "E003"));
    }

    #[test]
    fn test_warning_empty_body() {
        let dir = TempDir::new().unwrap();
        let path = create_skill_file(
            &dir,
            "empty-body",
            r#"---
name: empty-body
description: A skill with empty body content
version: "1.0"
---
"#,
        );

        let result = validate_skill_file(&path);
        assert!(result.valid); // Still valid, just warning
        assert!(result.warnings.iter().any(|e| e.code == "W004"));
    }

    #[test]
    fn test_is_valid_kebab_case() {
        assert!(is_valid_kebab_case("valid-name"));
        assert!(is_valid_kebab_case("simple"));
        assert!(is_valid_kebab_case("with-123-numbers"));
        assert!(is_valid_kebab_case("a"));

        assert!(!is_valid_kebab_case("Invalid")); // uppercase
        assert!(!is_valid_kebab_case("with_underscore")); // underscore
        assert!(!is_valid_kebab_case("-starts-with-hyphen")); // starts with hyphen
        assert!(!is_valid_kebab_case("ends-with-hyphen-")); // ends with hyphen
        assert!(!is_valid_kebab_case("double--hyphen")); // consecutive hyphens
        assert!(!is_valid_kebab_case("")); // empty
    }

    #[test]
    fn test_reserved_names() {
        assert!(is_reserved_name("help"));
        assert!(is_reserved_name("version"));
        assert!(!is_reserved_name("my-skill"));
    }

    #[test]
    fn test_collect_skills_in_dir() {
        let dir = TempDir::new().unwrap();

        // Create nested skill structure
        create_skill_file(
            &dir,
            "skill-a",
            "---\nname: skill-a\ndescription: A\n---\nBody",
        );
        create_skill_file(
            &dir,
            "skill-b",
            "---\nname: skill-b\ndescription: B\n---\nBody",
        );

        let mut files = Vec::new();
        collect_skills_in_dir(dir.path(), &mut files).unwrap();

        assert_eq!(files.len(), 2);
    }
}
