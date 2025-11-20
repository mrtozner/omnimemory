use super::{CommandContext, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub id: String,
    pub r#type: String,
    pub content: serde_json::Value,
    pub timestamp: SystemTime,
    pub relevance_score: f32,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBundle {
    pub items: Vec<ContextItem>,
    pub total_size_bytes: usize,
    pub context_included: Vec<String>,
    pub generation_time_ms: u64,
    pub filtered_by_size: bool,
}

pub fn handle_context(
    ctx: &CommandContext,
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    max_size: String,
    since: Option<String>,
) -> Result<()> {
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Assembling context..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Gathering context sources...");
        pb.set_position(30);
    }

    // Parse max size
    let max_size_bytes = parse_size(&max_size)?;
    if max_size_bytes == 0 {
        return Err(anyhow::anyhow!("Invalid max size: {}", max_size));
    }

    if let Some(ref pb) = pb {
        pb.set_message("Filtering and assembling...");
        pb.set_position(70);
    }

    // Mock implementation - would gather real context
    let context = assemble_context(&include, &exclude, max_size_bytes, &since)?;

    if let Some(ref pb) = pb {
        pb.set_message("Bundle assembled");
        pb.finish_with_message("✓ Context ready");
    }

    // Handle output
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&context)?);
        }
        super::OutputFormat::Human => {
            print_human_context(&context);
        }
        _ => {
            println!("{}", serde_json::to_string(&context)?);
        }
    }

    Ok(())
}

fn parse_size(size_str: &str) -> Result<usize> {
    let size_str = size_str.to_lowercase();
    let (number_str, unit) = size_str.split_at(
        size_str
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(size_str.len())
    );

    let number: f64 = number_str.parse()?;
    let bytes = match unit {
        "" | "b" => number as usize,
        "kb" | "k" => (number * 1024.0) as usize,
        "mb" | "m" => (number * 1024.0 * 1024.0) as usize,
        "gb" | "g" => (number * 1024.0 * 1024.0 * 1024.0) as usize,
        _ => return Err(anyhow::anyhow!("Invalid size unit: {}", unit)),
    };

    Ok(bytes)
}

fn assemble_context(
    include: &Option<Vec<String>>,
    exclude: &Option<Vec<String>>,
    max_size: usize,
    since: &Option<String>,
) -> Result<ContextBundle> {
    // Default context types if none specified
    let include_types = include.clone().unwrap_or_else(|| {
        vec![
            "recent_commands".to_string(),
            "working_directory".to_string(),
            "git_status".to_string(),
            "environment".to_string(),
            "file_operations".to_string(),
        ]
    });

    let exclude_types = exclude.clone().unwrap_or_default();
    let mut items = Vec::new();

    for context_type in include_types {
        if exclude_types.contains(&context_type) {
            continue;
        }

        match context_type.as_str() {
            "recent_commands" => {
                items.push(ContextItem {
                    id: "cmd-001".to_string(),
                    r#type: "recent_commands".to_string(),
                    content: serde_json::to_value(vec![
                        "ls -la",
                        "git status", 
                        "cargo test",
                        "docker build .",
                    ])?,
                    timestamp: SystemTime::now(),
                    relevance_score: 0.95,
                    metadata: HashMap::from([
                        ("command_count".to_string(), serde_json::Value::Number(4.into())),
                        ("time_range".to_string(), serde_json::Value::String("last_hour".to_string())),
                    ]),
                });
            }
            "working_directory" => {
                items.push(ContextItem {
                    id: "dir-001".to_string(),
                    r#type: "working_directory".to_string(),
                    content: serde_json::to_value("/workspace/omnimemory-cli")?,
                    timestamp: SystemTime::now(),
                    relevance_score: 1.0,
                    metadata: HashMap::from([
                        ("path".to_string(), serde_json::Value::String("/workspace/omnimemory-cli".to_string())),
                        ("git_repo".to_string(), serde_json::Value::Bool(true)),
                        ("project_type".to_string(), serde_json::Value::String("rust_cli".to_string())),
                    ]),
                });
            }
            "git_status" => {
                items.push(ContextItem {
                    id: "git-001".to_string(),
                    r#type: "git_status".to_string(),
                    content: serde_json::to_value(serde_json::json!({
                        "branch": "main",
                        "has_changes": true,
                        "staged_files": ["src/main.rs", "Cargo.toml"],
                        "untracked_files": ["README.md"],
                        "last_commit": "feat: initial CLI implementation"
                    }))?,
                    timestamp: SystemTime::now(),
                    relevance_score: 0.88,
                    metadata: HashMap::from([
                        ("branch".to_string(), serde_json::Value::String("main".to_string())),
                        ("commit_count".to_string(), serde_json::Value::Number(42.into())),
                    ]),
                });
            }
            "environment" => {
                items.push(ContextItem {
                    id: "env-001".to_string(),
                    r#type: "environment".to_string(),
                    content: serde_json::to_value(serde_json::json!({
                        "shell": "bash",
                        "user": "developer",
                        "rust_version": "1.70.0",
                        "cargo_version": "1.70.0",
                        "docker_available": true,
                        "git_version": "2.40.0"
                    }))?,
                    timestamp: SystemTime::now(),
                    relevance_score: 0.75,
                    metadata: HashMap::from([
                        ("tools_count".to_string(), serde_json::Value::Number(6.into())),
                        ("platform".to_string(), serde_json::Value::String("linux".to_string())),
                    ]),
                });
            }
            "file_operations" => {
                items.push(ContextItem {
                    id: "file-001".to_string(),
                    r#type: "file_operations".to_string(),
                    content: serde_json::to_value(vec![
                        "Modified: src/main.rs",
                        "Created: src/commands/suggest.rs", 
                        "Modified: Cargo.toml",
                    ])?,
                    timestamp: SystemTime::now(),
                    relevance_score: 0.82,
                    metadata: HashMap::from([
                        ("operation_count".to_string(), serde_json::Value::Number(3.into())),
                        ("session_start".to_string(), serde_json::Value::String("2h ago".to_string())),
                    ]),
                });
            }
            _ => {
                // Unknown context type - skip
                continue;
            }
        }
    }

    // Calculate total size and filter if necessary
    let mut total_size = items.iter()
        .map(|item| serde_json::to_string(&item.content).unwrap_or_default().len())
        .sum();

    let mut filtered = false;
    if total_size > max_size {
        // Sort by relevance and remove lowest scoring items until under limit
        items.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        while total_size > max_size && !items.is_empty() {
            items.pop();
            total_size = items.iter()
                .map(|item| serde_json::to_string(&item.content).unwrap_or_default().len())
                .sum();
        }
        filtered = true;
    }

    Ok(ContextBundle {
        items,
        total_size_bytes: total_size,
        context_included: include.clone().unwrap_or_default(),
        generation_time_ms: 89,
        filtered_by_size: filtered,
    })
}

fn print_human_context(bundle: &ContextBundle) {
    println!("\n{}", "📋 Context Bundle".bold().blue());
    println!("Total size: {:.2} KB", bundle.total_size_bytes as f64 / 1024.0);
    println!("Items included: {}\n", bundle.items.len());

    for (i, item) in bundle.items.iter().enumerate() {
        println!("{}. {} ({:.1}% relevance)",
            i + 1,
            item.r#type.replace('_', " ").replace('-', " ").split_whitespace()
                .map(|s| s.chars().next().map_or_else(|| s.to_string(), |c| c.to_uppercase().collect::<String>() + &s[1..]))
                .collect::<Vec<_>>()
                .join(" "),
            item.relevance_score * 100.0
        );

        match item.r#type.as_str() {
            "recent_commands" => {
                if let Some(commands) = item.content.as_array() {
                    println!("   Commands: {}", commands.len());
                    for cmd in commands.iter().take(3) {
                        if let Some(cmd_str) = cmd.as_str() {
                            println!("   • {}", cmd_str.cyan());
                        }
                    }
                    if commands.len() > 3 {
                        println!("   ... and {} more", commands.len() - 3);
                    }
                }
            }
            "working_directory" => {
                if let Some(dir) = item.content.as_str() {
                    println!("   Directory: {}", dir.cyan());
                }
            }
            "git_status" => {
                if let Some(obj) = item.content.as_object() {
                    if let Some(branch) = obj.get("branch").and_then(|v| v.as_str()) {
                        println!("   Branch: {}", branch.cyan());
                    }
                    if let Some(has_changes) = obj.get("has_changes").and_then(|v| v.as_bool()) {
                        if has_changes {
                            println!("   Status: {} Changes detected", "⚠".yellow());
                        } else {
                            println!("   Status: {} Clean", "✓".green());
                        }
                    }
                }
            }
            "environment" => {
                if let Some(obj) = item.content.as_object() {
                    println!("   Environment: {} tools detected", 
                        obj.keys().count()
                    );
                    if let Some(shell) = obj.get("shell").and_then(|v| v.as_str()) {
                        println!("   Shell: {}", shell.cyan());
                    }
                }
            }
            _ => {
                println!("   Data: {}", 
                    serde_json::to_string(&item.content)
                        .unwrap_or_default()
                        .chars()
                        .take(50)
                        .collect::<String>()
                );
            }
        }
        println!();
    }

    if bundle.filtered_by_size {
        println!("{} Context was truncated due to size limit", "⚠".yellow());
    }

    println!("Context types: {}", bundle.context_included.join(", "));
}