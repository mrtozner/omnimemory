use super::{CommandContext, Result};
use crate::integration::snapshots::models::*;
use clap::{Arg, ArgMatches, Command};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Snapshot CLI command implementations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotQueryResponse {
    pub results: Vec<SnapshotQueryResult>,
    pub query: String,
    pub total_results: usize,
    pub search_time_ms: u64,
    pub filters_applied: HashMap<String, serde_json::Value>,
}

/// Handle snapshot command
pub async fn handle_snapshot(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    match matches.subcommand() {
        Some(("create", create_matches)) => {
            handle_snapshot_create(ctx, create_matches).await
        }
        Some(("query", query_matches)) => {
            handle_snapshot_query(ctx, query_matches).await
        }
        Some(("list", list_matches)) => {
            handle_snapshot_list(ctx, list_matches).await
        }
        Some(("show", show_matches)) => {
            handle_snapshot_show(ctx, show_matches).await
        }
        Some(("delete", delete_matches)) => {
            handle_snapshot_delete(ctx, delete_matches).await
        }
        Some(("stats", stats_matches)) => {
            handle_snapshot_stats(ctx, stats_matches).await
        }
        Some(("cleanup", cleanup_matches)) => {
            handle_snapshot_cleanup(ctx, cleanup_matches).await
        }
        _ => {
            ctx.print_error("Invalid snapshot command. Use 'omni snapshot --help' for usage.");
            Ok(())
        }
    }
}

/// Create snapshot subcommand
pub async fn handle_snapshot_create(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let command = matches.get_one::<String>("command")
        .ok_or_else(|| anyhow::anyhow!("Command is required"))?.clone();
    
    let output = matches.get_one::<String>("output")
        .cloned()
        .unwrap_or_default();
    
    let exit_code = matches.get_one::<i32>("exit-code").cloned();
    
    let working_dir = std::env::current_dir()?
        .to_string_lossy()
        .to_string();

    let execution_time = matches.get_one::<u64>("execution-time")
        .cloned()
        .unwrap_or(0);

    let title = matches.get_one::<String>("title").cloned();
    let force = matches.get_flag("force");

    // Create semantic context
    let semantic_context = create_semantic_context(&command, &working_dir).await?;

    let request = CreateSnapshotRequest {
        command,
        output,
        exit_code,
        working_directory: working_dir,
        execution_time_ms: execution_time,
        semantic_context,
        force_create: force,
        custom_title: title,
        additional_tags: matches.get_many::<String>("tag")
            .map(|tags| tags.map(|t| t.clone()).collect())
            .unwrap_or_default(),
    };

    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Creating snapshot..."))
    } else {
        None
    };

    // This would integrate with the actual snapshot manager
    // For now, we'll simulate the creation
    if let Some(ref pb) = pb {
        pb.set_message("Generating summary...");
        pb.set_position(50);
    }

    let snapshot = simulate_snapshot_creation(&request).await?;

    if let Some(ref pb) = pb {
        pb.set_message("Saving snapshot...");
        pb.set_position(100);
        pb.finish_with_message("✓ Snapshot created");
    }

    // Print result
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
        _ => {
            println!("{} Snapshot created: {}", "✓".green(), snapshot.title);
            println!("   ID: {}", snapshot.id);
            println!("   Summary: {}", snapshot.summary);
            println!("   Importance: {:?}", snapshot.importance);
        }
    }

    Ok(())
}

/// Query snapshots subcommand
pub async fn handle_snapshot_query(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let query = matches.get_one::<String>("query")
        .ok_or_else(|| anyhow::anyhow!("Query is required"))?.clone();

    let limit = matches.get_one::<usize>("limit").cloned().unwrap_or(10);
    let min_similarity = matches.get_one::<f32>("min-similarity").cloned().unwrap_or(0.7);
    
    let min_importance = matches.get_one::<String>("min-importance")
        .and_then(|s| s.parse::<SnapshotImportance>().ok());

    let time_range = matches.get_one::<String>("since").map(|since_str| {
        // Parse time range (simple implementation)
        TimeRangeFilter {
            start: Some(chrono::Utc::now() - chrono::Duration::hours(1)), // Placeholder
            end: Some(chrono::Utc::now()),
        }
    });

    let working_directory = matches.get_one::<String>("directory").cloned();
    let required_tags = matches.get_many::<String>("tag")
        .map(|tags| tags.map(|t| t.clone()).collect())
        .or(Some(vec![]));

    let request = QuerySnapshotsRequest {
        query,
        min_importance,
        time_range,
        working_directory,
        required_tags,
        limit,
        min_similarity,
    };

    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Searching snapshots..."))
    } else {
        None
    };

    // Simulate search
    if let Some(ref pb) = pb {
        pb.set_message("Performing semantic search...");
        pb.set_position(70);
    }

    let response = simulate_snapshot_search(&request).await?;

    if let Some(ref pb) = pb {
        pb.finish_with_message("✓ Search completed");
    }

    // Print results
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        _ => {
            println!("{} Search results for '{}'", "🔍".blue(), response.query);
            println!("Found {} snapshots in {}ms\n", 
                response.total_results, response.search_time_ms);

            if response.results.is_empty() {
                println!("{}", "No snapshots found matching your criteria.".yellow());
            } else {
                for (i, result) in response.results.iter().enumerate() {
                    println!("{}. {}", i + 1, result.snapshot.title.bold());
                    println!("   ID: {} | Score: {:.2} | Importance: {:?}", 
                        result.snapshot.id, result.similarity_score, result.snapshot.importance);
                    println!("   Summary: {}", result.snapshot.summary);
                    println!("   Match: {}", result.match_reason);
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// List snapshots subcommand
pub async fn handle_snapshot_list(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let limit = matches.get_one::<usize>("limit").cloned().unwrap_or(20);
    let working_directory = matches.get_one::<String>("directory").cloned();
    
    let importance = matches.get_one::<String>("importance")
        .and_then(|s| s.parse::<SnapshotImportance>().ok());

    let required_tags = matches.get_many::<String>("tag")
        .map(|tags| tags.map(|t| t.clone()).collect())
        .or(Some(vec![]));

    // Simulate snapshot listing
    let snapshots = simulate_snapshot_list(limit, working_directory, importance.clone()).await?;

    // Print results
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&snapshots)?);
        }
        _ => {
            println!("{} Snapshots (showing {} most recent)", "📋".blue(), snapshots.len());
            
            if snapshots.is_empty() {
                println!("{}", "No snapshots found.".yellow());
            } else {
                println!("ID | Title | Importance | Created At");
                println!("---|-------|------------|----------");
                
                for snapshot in snapshots {
                    println!("{} | {} | {:?} | {}", 
                        truncate_id(&snapshot.id),
                        truncate_text(&snapshot.title, 30),
                        snapshot.importance,
                        snapshot.created_at.format("%Y-%m-%d %H:%M")
                    );
                }
            }
        }
    }

    Ok(())
}

/// Show snapshot subcommand
pub async fn handle_snapshot_show(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let id = matches.get_one::<String>("id")
        .ok_or_else(|| anyhow::anyhow!("Snapshot ID is required"))?.clone();

    // Simulate getting snapshot
    let snapshot = simulate_get_snapshot(&id).await?
        .ok_or_else(|| anyhow::anyhow!("Snapshot not found: {}", id))?;

    // Print snapshot details
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&snapshot)?);
        }
        _ => {
            println!("{} Snapshot Details", "📖".blue());
            println!("ID: {}", snapshot.id);
            println!("Title: {}", snapshot.title);
            println!("Summary: {}", snapshot.summary);
            println!("Command: {}", snapshot.command);
            
            if let Some(exit_code) = snapshot.exit_code {
                println!("Exit Code: {}", exit_code);
            }
            
            println!("Importance: {:?}", snapshot.importance);
            println!("Created: {}", snapshot.created_at);
            println!("Execution Time: {}ms", snapshot.execution_time_ms);
            println!("Working Directory: {}", snapshot.semantic_context.working_directory);
            
            if !snapshot.semantic_context.tags.is_empty() {
                println!("Tags: {}", snapshot.semantic_context.tags.join(", "));
            }
            
            if !snapshot.output.is_empty() {
                println!("\nOutput:");
                println!("{}", snapshot.output);
            }
        }
    }

    Ok(())
}

/// Delete snapshot subcommand
pub async fn handle_snapshot_delete(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let id = matches.get_one::<String>("id")
        .ok_or_else(|| anyhow::anyhow!("Snapshot ID is required"))?.clone();
    
    let force = matches.get_flag("force");

    if !force {
        let confirmed = ctx.is_interactive() && dialoguer::Confirm::new()
            .with_prompt(&format!("Delete snapshot '{}'?", id))
            .default(false)
            .interact()?;
        
        if !confirmed {
            ctx.print_warning("Deletion cancelled.");
            return Ok(());
        }
    }

    // Simulate deletion
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Deleting snapshot..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Removing from storage...");
        pb.set_position(50);
    }

    simulate_delete_snapshot(&id).await?;

    if let Some(ref pb) = pb {
        pb.set_message("Cleaning up vector index...");
        pb.set_position(100);
        pb.finish_with_message("✓ Snapshot deleted");
    }

    ctx.print_success(&format!("Snapshot {} deleted", id));

    Ok(())
}

/// Stats snapshot subcommand
pub async fn handle_snapshot_stats(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    // Simulate stats
    let stats = simulate_snapshot_stats().await?;

    // Print stats
    match ctx.output_format {
        super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        _ => {
            println!("{} Snapshot Statistics", "📊".blue());
            println!("Total Snapshots: {}", stats.total_snapshots);
            println!("Storage Size: {} MB", stats.storage_size_bytes / (1024 * 1024));
            println!("Average Execution Time: {:.1}ms", stats.average_execution_time_ms);
            
            if let Some(oldest) = stats.oldest_snapshot {
                println!("Oldest Snapshot: {}", oldest.format("%Y-%m-%d %H:%M"));
            }
            
            if let Some(newest) = stats.newest_snapshot {
                println!("Newest Snapshot: {}", newest.format("%Y-%m-%d %H:%M"));
            }
            
            println!("\nBy Importance:");
            for (importance, count) in &stats.by_importance {
                println!("  {}: {}", importance, count);
            }
            
            println!("\nBy Command Type:");
            for (cmd_type, count) in &stats.by_command_type {
                println!("  {}: {}", cmd_type, count);
            }
        }
    }

    Ok(())
}

/// Cleanup snapshot subcommand
pub async fn handle_snapshot_cleanup(ctx: &CommandContext, matches: &ArgMatches) -> Result<()> {
    let dry_run = matches.get_flag("dry-run");
    let force = matches.get_flag("force");

    if !force && !dry_run {
        let confirmed = ctx.is_interactive() && dialoguer::Confirm::new()
            .with_prompt("Run cleanup? This will delete old snapshots based on your configuration.")
            .default(false)
            .interact()?;
        
        if !confirmed {
            ctx.print_warning("Cleanup cancelled.");
            return Ok(());
        }
    }

    let pb = if ctx.cli.verbose > 0 && !dry_run {
        Some(ctx.create_progress("Cleaning up old snapshots..."))
    } else {
        None
    };

    // Simulate cleanup
    let deleted_count = simulate_snapshot_cleanup(dry_run).await?;

    if let Some(ref pb) = pb {
        pb.finish_with_message("✓ Cleanup completed");
    }

    if dry_run {
        ctx.print_warning(&format!("Dry run: {} snapshots would be deleted", deleted_count));
    } else {
        ctx.print_success(&format!("Cleanup completed: {} snapshots deleted", deleted_count));
    }

    Ok(())
}

// Helper functions

async fn create_semantic_context(command: &str, working_dir: &str) -> Result<SemanticContext> {
    let mut tags = Vec::new();
    let mut file_paths = Vec::new();
    let mut command_type = "unknown".to_string();
    let mut primary_tool = None;

    // Analyze command
    let parts: Vec<&str> = command.split_whitespace().collect();
    let flags: Vec<String> = if parts.len() > 1 {
        parts[1..].iter().map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    };

    if !parts.is_empty() {
        command_type = parts[0].to_string();
        primary_tool = Some(parts[0].to_string());

        // Add tool-based tags
        match parts[0] {
            "git" => tags.push("git".to_string()),
            "docker" => tags.push("docker".to_string()),
            "cargo" => tags.push("rust".to_string()),
            "npm" | "yarn" => tags.push("javascript".to_string()),
            "pip" | "conda" => tags.push("python".to_string()),
            _ => {}
        }

        // Extract file paths from command
        for part in &parts {
            if part.starts_with('.') || part.contains('/') {
                file_paths.push(part.to_string());
            }
        }

        // Add command type tags
        if command.starts_with("git commit") {
            tags.push("git-commit".to_string());
        } else if command.starts_with("docker build") {
            tags.push("docker-build".to_string());
        } else if command.starts_with("cargo test") {
            tags.push("testing".to_string());
        }
    }

    Ok(SemanticContext {
        working_directory: working_dir.to_string(),
        environment: std::env::vars().collect(),
        recent_commands: Vec::new(), // Would be populated from session history
        git_info: None, // Would be populated from git info
        docker_context: None, // Would be populated from docker info
        system_info: SystemContext {
            platform: std::env::consts::OS.to_string(),
            shell: std::env::var("SHELL").unwrap_or_default(),
            user: std::env::var("USER").unwrap_or_default(),
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| std::env::var("COMPUTERNAME").unwrap_or_default()),
            cpu_info: "unknown".to_string(),
            memory_gb: 0.0,
        },
        tags,
        file_paths,
        command_structure: CommandStructure {
            command_type,
            primary_tool,
            parameters: HashMap::new(),
            flags,
            is_batch_command: command.contains("&&") || command.contains("||"),
        },
    })
}

fn truncate_id(id: &str) -> String {
    if id.len() > 12 {
        format!("{}...", &id[..12])
    } else {
        id.to_string()
    }
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() > max_len {
        format!("{}...", &text[..max_len-3])
    } else {
        text.to_string()
    }
}

// Simulation functions (would be replaced with actual snapshot manager calls)

async fn simulate_snapshot_creation(request: &CreateSnapshotRequest) -> Result<Snapshot> {
    let importance = if request.force_create {
        SnapshotImportance::Medium
    } else {
        SnapshotImportance::from_command_and_context(
            &request.command,
            request.exit_code,
            &request.working_directory,
            &HashMap::new(),
        )
    };

    let summary = format!("Command: {} - {} | Tags: {}", 
        request.command,
        if request.exit_code == Some(0) { "Success" } else { "Failed" },
        request.semantic_context.tags.join(", ")
    );

    Ok(Snapshot {
        id: format!("snap-{}", uuid::Uuid::new_v4()),
        title: request.custom_title.clone().unwrap_or_else(|| 
            format!("Snapshot of {}", request.command.split_whitespace().next().unwrap_or("Unknown"))
        ),
        summary: summary.chars().take(500).collect(),
        command: request.command.clone(),
        output: request.output.clone(),
        exit_code: request.exit_code,
        created_at: chrono::Utc::now(),
        execution_time_ms: request.execution_time_ms,
        semantic_context: request.semantic_context.clone(),
        importance,
        embedding_id: Some(format!("embed-{}", uuid::Uuid::new_v4())),
        related_snapshot_ids: vec![],
        custom_metadata: HashMap::new(),
        storage_info: SnapshotStorageInfo {
            file_path: format!("/tmp/snapshots/{}.json", uuid::Uuid::new_v4()),
            size_bytes: 1024,
            vector_index: None,
            db_record_id: None,
        },
    })
}

async fn simulate_snapshot_search(request: &QuerySnapshotsRequest) -> Result<SnapshotQueryResponse> {
    // Simulate search delay
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let mock_results = vec![SnapshotQueryResult {
        snapshot: Snapshot {
            id: "snap-example".to_string(),
            title: "Example Snapshot".to_string(),
            summary: "Example summary".to_string(),
            command: "example command".to_string(),
            output: "example output".to_string(),
            exit_code: Some(0),
            created_at: chrono::Utc::now(),
            execution_time_ms: 1000,
            semantic_context: create_semantic_context("example command", "/tmp").await?,
            importance: SnapshotImportance::Medium,
            embedding_id: Some("embed-example".to_string()),
            related_snapshot_ids: vec![],
            custom_metadata: HashMap::new(),
            storage_info: SnapshotStorageInfo {
                file_path: "/tmp/example.json".to_string(),
                size_bytes: 1024,
                vector_index: None,
                db_record_id: None,
            },
        },
        similarity_score: 0.85,
        match_reason: "Query match".to_string(),
        highlighted_content: vec!["Example summary".to_string()],
    }];

    Ok(SnapshotQueryResponse {
        results: mock_results,
        query: request.query.clone(),
        total_results: 1,
        search_time_ms: 100,
        filters_applied: HashMap::new(),
    })
}

async fn simulate_snapshot_list(limit: usize, _working_dir: Option<String>, _importance: Option<SnapshotImportance>) -> Result<Vec<Snapshot>> {
    let mut snapshots = Vec::new();
    
    for i in 0..limit.min(5) {
        snapshots.push(Snapshot {
            id: format!("snap-{:03}", i),
            title: format!("Snapshot {}", i),
            summary: format!("Summary for snapshot {}", i),
            command: format!("command {}", i),
            output: "output".to_string(),
            exit_code: Some(0),
            created_at: chrono::Utc::now(),
            execution_time_ms: 1000,
            semantic_context: create_semantic_context(&format!("command {}", i), "/tmp").await?,
            importance: SnapshotImportance::Low,
            embedding_id: Some(format!("embed-{:03}", i)),
            related_snapshot_ids: vec![],
            custom_metadata: HashMap::new(),
            storage_info: SnapshotStorageInfo {
                file_path: format!("/tmp/snapshots/{:03}.json", i),
                size_bytes: 1024,
                vector_index: None,
                db_record_id: None,
            },
        });
    }

    Ok(snapshots)
}

async fn simulate_get_snapshot(id: &str) -> Result<Option<Snapshot>> {
    Ok(Some(Snapshot {
        id: id.to_string(),
        title: "Example Snapshot".to_string(),
        summary: "Example summary".to_string(),
        command: "example command".to_string(),
        output: "example output".to_string(),
        exit_code: Some(0),
        created_at: chrono::Utc::now(),
        execution_time_ms: 1000,
        semantic_context: create_semantic_context("example command", "/tmp").await?,
        importance: SnapshotImportance::Medium,
        embedding_id: Some("embed-example".to_string()),
        related_snapshot_ids: vec![],
        custom_metadata: HashMap::new(),
        storage_info: SnapshotStorageInfo {
            file_path: "/tmp/example.json".to_string(),
            size_bytes: 1024,
            vector_index: None,
            db_record_id: None,
        },
    }))
}

async fn simulate_delete_snapshot(id: &str) -> Result<()> {
    // Simulate deletion delay
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    println!("Would delete snapshot: {}", id);
    Ok(())
}

async fn simulate_snapshot_stats() -> Result<SnapshotStats> {
    Ok(SnapshotStats {
        total_snapshots: 42,
        by_importance: HashMap::from([
            ("Low".to_string(), 25),
            ("Medium".to_string(), 12),
            ("High".to_string(), 4),
            ("Critical".to_string(), 1),
        ]),
        by_working_directory: HashMap::from([
            ("/home/user/project".to_string(), 20),
            ("/home/user/other".to_string(), 15),
            ("/tmp".to_string(), 7),
        ]),
        by_command_type: HashMap::from([
            ("git".to_string(), 10),
            ("docker".to_string(), 8),
            ("cargo".to_string(), 5),
            ("other".to_string(), 19),
        ]),
        average_execution_time_ms: 1250.5,
        storage_size_bytes: 1024 * 1024, // 1MB
        oldest_snapshot: Some(chrono::Utc::now() - chrono::Duration::days(30)),
        newest_snapshot: Some(chrono::Utc::now()),
    })
}

async fn simulate_snapshot_cleanup(dry_run: bool) -> Result<u32> {
    // Simulate cleanup delay
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    if dry_run {
        Ok(3) // Would delete 3 snapshots
    } else {
        Ok(3) // Actually deleted 3 snapshots
    }
}

/// Add snapshot subcommand to Clap command builder
pub fn add_snapshot_subcommand(base_command: Command) -> Command {
    base_command.subcommand(
        Command::new("snapshot")
            .about("Manage command execution snapshots")
            .subcommand_required(true)
            .subcommand(
                Command::new("create")
                    .about("Create a new snapshot")
                    .arg(Arg::new("command")
                        .short('c')
                        .long("command")
                        .value_name("COMMAND")
                        .help("The command that was executed")
                        .required(true))
                    .arg(Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("OUTPUT")
                        .help("The command output"))
                    .arg(Arg::new("exit-code")
                        .short('e')
                        .long("exit-code")
                        .value_name("CODE")
                        .help("The exit code of the command"))
                    .arg(Arg::new("execution-time")
                        .short('t')
                        .long("execution-time")
                        .value_name("MS")
                        .help("Execution time in milliseconds"))
                    .arg(Arg::new("title")
                        .short('T')
                        .long("title")
                        .value_name("TITLE")
                        .help("Custom title for the snapshot"))
                    .arg(Arg::new("tag")
                        .short('g')
                        .long("tag")
                        .value_name("TAG")
                        .help("Add a tag to the snapshot")
                        .action(clap::ArgAction::Append))
                    .arg(Arg::new("force")
                        .short('f')
                        .long("force")
                        .help("Force creation even for low-importance commands"))
            )
            .subcommand(
                Command::new("query")
                    .about("Search for snapshots using semantic queries")
                    .arg(Arg::new("query")
                        .short('q')
                        .long("query")
                        .value_name("QUERY")
                        .help("Natural language search query")
                        .required(true))
                    .arg(Arg::new("limit")
                        .short('l')
                        .long("limit")
                        .value_name("N")
                        .help("Maximum number of results")
                        .default_value("10"))
                    .arg(Arg::new("min-similarity")
                        .short('s')
                        .long("min-similarity")
                        .value_name("THRESHOLD")
                        .help("Minimum similarity score (0.0-1.0)")
                        .default_value("0.7"))
                    .arg(Arg::new("min-importance")
                        .short('i')
                        .long("min-importance")
                        .value_name("LEVEL")
                        .help("Minimum importance level")
                        .value_parser(["low", "medium", "high", "critical"]))
                    .arg(Arg::new("since")
                        .long("since")
                        .value_name("TIME")
                        .help("Time range filter (e.g., '1h', '1d')"))
                    .arg(Arg::new("directory")
                        .short('d')
                        .long("directory")
                        .value_name("PATH")
                        .help("Working directory filter"))
                    .arg(Arg::new("tag")
                        .short('g')
                        .long("tag")
                        .value_name("TAG")
                        .help("Required tag filter")
                        .action(clap::ArgAction::Append))
            )
            .subcommand(
                Command::new("list")
                    .about("List snapshots with optional filtering")
                    .arg(Arg::new("limit")
                        .short('l')
                        .long("limit")
                        .value_name("N")
                        .help("Maximum number of results")
                        .default_value("20"))
                    .arg(Arg::new("directory")
                        .short('d')
                        .long("directory")
                        .value_name("PATH")
                        .help("Working directory filter"))
                    .arg(Arg::new("importance")
                        .short('i')
                        .long("importance")
                        .value_name("LEVEL")
                        .help("Importance level filter")
                        .value_parser(["low", "medium", "high", "critical"]))
                    .arg(Arg::new("tag")
                        .short('g')
                        .long("tag")
                        .value_name("TAG")
                        .help("Required tag filter")
                        .action(clap::ArgAction::Append))
            )
            .subcommand(
                Command::new("show")
                    .about("Show detailed information about a snapshot")
                    .arg(Arg::new("id")
                        .short('i')
                        .long("id")
                        .value_name("ID")
                        .help("Snapshot ID")
                        .required(true))
            )
            .subcommand(
                Command::new("delete")
                    .about("Delete a snapshot")
                    .arg(Arg::new("id")
                        .short('i')
                        .long("id")
                        .value_name("ID")
                        .help("Snapshot ID")
                        .required(true))
                    .arg(Arg::new("force")
                        .short('f')
                        .long("force")
                        .help("Delete without confirmation"))
            )
            .subcommand(
                Command::new("stats")
                    .about("Show snapshot statistics")
            )
            .subcommand(
                Command::new("cleanup")
                    .about("Clean up old snapshots based on configuration")
                    .arg(Arg::new("dry-run")
                        .short('n')
                        .long("dry-run")
                        .help("Show what would be deleted without actually deleting"))
                    .arg(Arg::new("force")
                        .short('f')
                        .long("force")
                        .help("Run cleanup without confirmation"))
            )
    )
}