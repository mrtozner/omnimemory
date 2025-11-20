use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

mod commands;
mod integration;

// Gateway configuration
const GATEWAY_BASE: &str = "http://localhost:8009/api/v1";
const GATEWAY_API_KEY: &str = "omni_sk_test_key"; // Development key

#[derive(Parser, Debug, Clone)]
#[command(name = "omni")]
#[command(about = "OmniMemory CLI - AI-powered command suggestion and failure analysis")]
#[command(version = "0.1.0")]
#[command(author = "OmniMemory Team")]
#[command(long_about = None)]
pub struct Cli {
    /// Optional config file path. If not provided, uses default locations.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Output format for commands that support it
    #[arg(short, long, default_value = "human")]
    output: OutputFormat,

    /// Increase verbosity (can be used multiple times)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Quiet mode (suppress warnings and non-essential output)
    #[arg(short, long)]
    quiet: bool,

    /// Disable interactive prompts
    #[arg(short, long)]
    no_input: bool,

    /// Timeout for operations (e.g., "30s", "1m")
    #[arg(short, long, default_value = "30s")]
    timeout: humantime::Duration,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Human,
    Json,
    Plain,
    Table,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Human => write!(f, "human"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Plain => write!(f, "plain"),
            OutputFormat::Table => write!(f, "table"),
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
pub enum UnifiedAction {
    /// Check unified intelligence health status
    Status,

    /// Get predictions from unified engine
    Predict {
        /// Context as JSON string
        #[arg(short, long)]
        context: Option<String>,

        /// User profile as JSON string
        #[arg(short, long)]
        profile: Option<String>,
    },

    /// Get proactive suggestions
    Suggest {
        /// Show only suggestions that should be shown now
        #[arg(long)]
        show_now: bool,

        /// Context as JSON string
        #[arg(short, long)]
        context: Option<String>,
    },

    /// Submit feedback on a suggestion
    Feedback {
        /// Suggestion ID
        suggestion_id: String,

        /// Whether the suggestion was accepted
        #[arg(long)]
        accepted: bool,

        /// Response time in milliseconds
        #[arg(long)]
        response_time: f64,

        /// Optional text feedback
        #[arg(long)]
        comment: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum ServiceAction {
    /// Start all OmniMemory services
    Start,

    /// Stop all OmniMemory services
    Stop,

    /// Restart all services
    Restart,

    /// Show status of all services
    Status,

    /// Show service logs
    Logs {
        /// Service name to show logs for
        service: Option<String>,

        /// Follow log output
        #[arg(short, long)]
        follow: bool,

        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,
    },

    /// Run health checks on all services
    Health,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Propose next actions or tools based on recent context and history
    Suggest {
        /// Include current context (working directory, recent commands, etc.)
        #[arg(long)]
        context: bool,

        /// Generate suggestions without execution
        #[arg(long)]
        dry_run: bool,

        /// Maximum number of suggestions to generate
        #[arg(long, default_value = "5")]
        max_suggestions: usize,

        /// Focus on specific domains (e.g., "git", "docker", "system")
        #[arg(long)]
        domain: Option<Vec<String>>,

        /// Output suggestions as structured JSON
        #[arg(long)]
        json: bool,
    },

    /// Analyze prior tool invocation results and generate failure analysis
    WhyFailed {
        /// Specify operation ID to analyze
        #[arg(short, long)]
        id: Option<String>,

        /// Analyze the last N operations
        #[arg(short, long, default_value = "1")]
        last: usize,

        /// Focus on specific error types
        #[arg(long)]
        error_type: Option<String>,

        /// Include debugging information
        #[arg(short, long)]
        debug: bool,

        /// Generate remediation suggestions
        #[arg(long)]
        suggest_fix: bool,
    },

    /// Assemble current context for prompt injection
    Context {
        /// Include specific context types
        #[arg(short, long)]
        include: Option<Vec<String>>,

        /// Exclude specific context types
        #[arg(short, long)]
        exclude: Option<Vec<String>>,

        /// Maximum context size
        #[arg(long, default_value = "10KB")]
        max_size: String,

        /// Filter by time range (e.g., "1h", "1d")
        #[arg(long)]
        since: Option<String>,
    },

    /// Manage user profiles and preferences
    Profile {
        /// Set current profile
        #[arg(short, long)]
        set: Option<String>,

        /// Show current profile
        #[arg(short, long)]
        show: bool,

        /// Create new profile
        #[arg(short, long)]
        create: Option<String>,

        /// Delete profile
        #[arg(short, long)]
        delete: Option<String>,

        /// List all profiles
        #[arg(short, long)]
        list: bool,
    },

    /// Query fact database for historical information
    Facts {
        /// Search for specific facts
        query: Option<String>,

        /// Search by entity (command, tool, directory, etc.)
        #[arg(short, long)]
        entity: Option<String>,

        /// Time range for results
        #[arg(long)]
        since: Option<String>,

        /// Maximum results to return
        #[arg(long, default_value = "10")]
        limit: usize,

        /// Output format (facts, timeline, statistics)
        #[arg(short, long, default_value = "facts")]
        format: FactFormat,
    },

    /// Manage preferences (configuration settings)
    Pref {
        #[command(subcommand)]
        action: PrefCommand,
    },

    /// Integration and diagnostic commands
    Daemon {
        #[command(subcommand)]
        subcommand: DaemonCommand,
    },

    /// Display system information and diagnostics
    Doctor {
        /// Run comprehensive health check
        #[arg(short, long)]
        comprehensive: bool,

        /// Check specific components
        #[arg(long)]
        check: Option<Vec<String>>,

        /// Output format for health report
        #[arg(long, default_value = "human")]
        format: String,
    },

    /// Initialize OmniMemory system (MCP server, configs, integration)
    Init {
        /// Show what would be configured without making changes
        #[arg(long)]
        dry_run: bool,

        /// Skip interactive prompts and use defaults
        #[arg(long)]
        silent: bool,

        /// Comma-separated tool names or 'all' (e.g., "Claude Code,Cursor")
        #[arg(long)]
        tools: Option<String>,

        /// Use enterprise mode (only with --silent)
        #[arg(long)]
        enterprise: bool,

        /// Force reinstallation even if already configured
        #[arg(short, long)]
        force: bool,

        /// Skip updating Claude.md (OmniMemory works automatically via MCP)
        #[arg(long)]
        skip_claude_md: bool,

        /// Custom path to MCP server directory
        #[arg(long)]
        mcp_path: Option<String>,
    },

    /// Remove OmniMemory configuration from AI coding tools
    Remove {
        /// Show what would be removed without making changes
        #[arg(long)]
        dry_run: bool,

        /// Remove from all configured tools
        #[arg(long)]
        all: bool,

        /// Comma-separated tool names (e.g., "Claude Code,Cursor")
        #[arg(long)]
        tools: Option<String>,
    },

    /// Display interactive TUI dashboard with system status and metrics
    Dashboard,

    // NOTE: Snapshot command temporarily disabled - needs refactoring to use proper subcommand enum instead of ArgMatches
    // /// Manage command execution snapshots with semantic context and vector storage
    // Snapshot {
    //     #[command(subcommand)]
    //     subcommand: clap::ArgMatches,
    // },

    /// Context profiling and analysis commands
    Profiler {
        #[command(subcommand)]
        subcommand: ProfileSubcommand,
    },

    /// Context breakdown and detailed inspection
    ContextBreakdown {
        /// Analysis depth (basic, detailed, comprehensive)
        #[arg(long, default_value = "detailed")]
        depth: String,

        /// Specific context types to include
        #[arg(long)]
        include_types: Option<Vec<String>>,

        /// Context types to exclude
        #[arg(long)]
        exclude_types: Option<Vec<String>>,

        /// Time range for analysis (e.g., "1h", "1d", "1w")
        #[arg(long)]
        time_range: Option<String>,

        /// Minimum relevance score threshold
        #[arg(long)]
        min_relevance: Option<f32>,

        /// Show detailed segment information
        #[arg(long)]
        show_details: bool,

        /// Group by field (type, source, timestamp, size)
        #[arg(long)]
        group_by: Option<String>,

        /// Generate summary statistics
        #[arg(long)]
        generate_stats: bool,
    },

    /// Token usage analysis and reporting
    TokensUsage {
        /// Analysis scope (current, daily, weekly, monthly, all_time)
        #[arg(long, default_value = "current")]
        scope: String,

        /// Time period for analysis (e.g., "1d", "7d", "30d")
        #[arg(long, default_value = "24h")]
        time_period: String,

        /// Specific token types to analyze
        #[arg(long)]
        token_types: Option<Vec<String>>,

        /// Group by field (date, hour, operation_type, model)
        #[arg(long)]
        group_by: Option<String>,

        /// Include cost analysis
        #[arg(long)]
        include_costs: bool,

        /// Budget limit for analysis
        #[arg(long)]
        budget_limit: Option<f32>,

        /// Show detailed breakdown
        #[arg(long)]
        detailed_breakdown: bool,

        /// Generate optimization suggestions
        #[arg(long)]
        generate_optimizations: bool,
    },

    /// Tools performance monitoring and impact scoring
    ToolsPerformance {
        /// Analysis depth (basic, detailed, comprehensive)
        #[arg(long, default_value = "detailed")]
        analysis_depth: String,

        /// Time window for analysis (1h, 6h, 24h, 7d, 30d)
        #[arg(long, default_value = "24h")]
        time_window: String,

        /// Specific tools to analyze
        #[arg(long)]
        target_tools: Option<Vec<String>>,

        /// Sort tools by metric
        #[arg(long)]
        sort_by: Option<String>,

        /// Minimum usage threshold for inclusion
        #[arg(long)]
        min_usage_threshold: Option<usize>,

        /// Show detailed breakdowns
        #[arg(long)]
        detailed_breakdown: bool,

        /// Generate optimization suggestions
        #[arg(long)]
        generate_optimizations: bool,
    },

    /// Export profile analysis reports
    ProfileExport {
        /// Output file path
        #[arg(short, long, default_value = "profile_export")]
        output_file: String,

        /// Export format (json, csv, html, pdf, xml, yaml)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Include specific data types in export
        #[arg(long)]
        include_data: Option<Vec<String>>,

        /// Export scope (summary, detailed, comprehensive)
        #[arg(long, default_value = "detailed")]
        scope: String,

        /// Include charts and visualizations
        #[arg(long)]
        include_charts: bool,

        /// Include metadata and technical details
        #[arg(long)]
        include_metadata: bool,

        /// Compress output files
        #[arg(long)]
        compress: bool,

        /// Generate multiple files
        #[arg(long)]
        split_files: bool,
    },

    /// Interact with Unified Intelligence System
    Unified {
        #[command(subcommand)]
        action: UnifiedAction,
    },

    /// Manage OmniMemory services
    Services {
        #[command(subcommand)]
        action: ServiceAction,
    },

    /// Open dashboard in browser
    DashboardWeb {
        /// Open specific dashboard page
        #[arg(short, long)]
        page: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone, Serialize, Deserialize)]
pub enum ProfileSubcommand {
    /// Comprehensive context analysis
    Analyze {
        /// Profile name or identifier
        #[arg(short, long)]
        name: Option<String>,

        /// Analysis depth (basic, detailed, comprehensive)
        #[arg(long, default_value = "detailed")]
        depth: String,

        /// Include specific analysis types
        #[arg(long)]
        include: Option<Vec<String>>,

        /// Exclude specific analysis types
        #[arg(long)]
        exclude: Option<Vec<String>>,

        /// Generate reports for export
        #[arg(long)]
        generate_reports: bool,

        /// Save analysis results to file
        #[arg(short, long)]
        output_file: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum PrefCommand {
    /// Set a preference value
    Set {
        /// Configuration key (supports dot notation, e.g., "suggestion.max_results")
        key: String,

        /// Configuration value
        value: String,

        /// Scope (user, project, system)
        #[arg(long, default_value = "user")]
        scope: PrefScope,
    },

    /// Get a preference value
    Get {
        /// Configuration key
        key: String,

        /// Scope to read from
        #[arg(short, long)]
        scope: Option<PrefScope>,

        /// Show source of the value
        #[arg(short, long)]
        show_source: bool,
    },

    /// List all preferences
    List {
        /// Scope to list from
        #[arg(short, long)]
        scope: Option<PrefScope>,

        /// Filter by prefix
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// Reset preferences to defaults
    Reset {
        /// Scope to reset
        #[arg(short, long)]
        scope: Option<PrefScope>,

        /// Reset specific keys only
        #[arg(long)]
        keys: Option<Vec<String>>,
    },
}

#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrefScope {
    User,
    Project,
    System,
}

impl std::fmt::Display for PrefScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefScope::User => write!(f, "user"),
            PrefScope::Project => write!(f, "project"),
            PrefScope::System => write!(f, "system"),
        }
    }
}

#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize)]
pub enum FactFormat {
    Facts,
    Timeline,
    Statistics,
}

#[derive(Subcommand, Debug, Clone)]
pub enum DaemonCommand {
    /// Start the OmniMemory daemon
    Start {
        /// Daemon configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Enable debug logging
        #[arg(short, long)]
        debug: bool,
    },

    /// Stop the running daemon
    Stop {
        /// Force stop if graceful shutdown fails
        #[arg(short, long)]
        force: bool,
    },

    /// Check daemon status
    Status {
        /// Include detailed status information
        #[arg(short, long)]
        verbose: bool,

        /// Show process information
        #[arg(short, long)]
        process: bool,
    },

    /// Restart the daemon
    Restart {
        /// Don't wait for graceful shutdown
        #[arg(short, long)]
        force: bool,
    },

    /// Show daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,

        /// Follow log output
        #[arg(short, long)]
        follow: bool,

        /// Filter log levels (error, warn, info, debug)
        #[arg(long)]
        level: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity level
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info", 
        2 => "debug",
        _ => "trace",
    };

    if cli.quiet {
        std::env::set_var("RUST_LOG", "");
    } else {
        std::env::set_var("RUST_LOG", log_level);
    }

    env_logger::init();

    // Load configuration
    let config = load_config(&cli.config)?;

    // Create command context
    let ctx = commands::CommandContext::new(cli.clone(), config);

    // Handle commands
    match cli.command {
        None => {
            commands::handle_default(&ctx)?;
        }
        Some(Commands::Suggest { context, dry_run, max_suggestions, domain, json }) => {
            commands::handle_suggest(&ctx, context, dry_run, max_suggestions, domain, json)?;
        }
        Some(Commands::WhyFailed { id, last, error_type, debug, suggest_fix }) => {
            commands::handle_why_failed(&ctx, id, last, error_type, debug, suggest_fix)?;
        }
        Some(Commands::Context { include, exclude, max_size, since }) => {
            commands::handle_context(&ctx, include, exclude, max_size, since)?;
        }
        Some(Commands::Profile { set, show, create, delete, list }) => {
            commands::handle_profile(&ctx, set, show, create, delete, list)?;
        }
        Some(Commands::Facts { query, entity, since, limit, format }) => {
            commands::handle_facts(&ctx, query, entity, since, limit, format)?;
        }
        Some(Commands::Pref { action }) => {
            commands::handle_pref(&ctx, action)?;
        }
        Some(Commands::Daemon { subcommand }) => {
            commands::handle_daemon(&ctx, subcommand)?;
        }
        Some(Commands::Doctor { comprehensive, check, format }) => {
            commands::handle_doctor(&ctx, comprehensive, check, format)?;
        }
        Some(Commands::Init { dry_run, silent, tools, enterprise, force, skip_claude_md, mcp_path }) => {
            commands::handle_init(&ctx, dry_run, silent, tools.as_deref(), enterprise, force, skip_claude_md, mcp_path)?;
        }
        Some(Commands::Remove { dry_run, all, tools }) => {
            commands::handle_remove(&ctx, dry_run, all, tools.as_deref())?;
        }
        Some(Commands::Dashboard) => {
            commands::handle_dashboard(&ctx)?;
        }
        // Some(Commands::Snapshot { subcommand }) => {
        //     let rt = tokio::runtime::Runtime::new()?;
        //     rt.block_on(commands::handle_snapshot(&ctx, &subcommand))?;
        // }
        Some(Commands::Profiler { subcommand }) => {
            commands::handle_profile_command(&ctx, subcommand)?;
        }
        Some(Commands::ContextBreakdown { depth, include_types, exclude_types, time_range, min_relevance, show_details, group_by, generate_stats }) => {
            commands::handle_context_breakdown_cmd(&ctx, depth, include_types, exclude_types, time_range, min_relevance, show_details, group_by, generate_stats)?;
        }
        Some(Commands::TokensUsage { scope, time_period, token_types, group_by, include_costs, budget_limit, detailed_breakdown, generate_optimizations }) => {
            commands::handle_tokens_usage_cmd(&ctx, scope, time_period, token_types, group_by, include_costs, budget_limit, detailed_breakdown, generate_optimizations)?;
        }
        Some(Commands::ToolsPerformance { analysis_depth, time_window, target_tools, sort_by, min_usage_threshold, detailed_breakdown, generate_optimizations }) => {
            commands::handle_tools_performance_cmd(&ctx, analysis_depth, time_window, target_tools, sort_by, min_usage_threshold, detailed_breakdown, generate_optimizations)?;
        }
        Some(Commands::ProfileExport { output_file, format, include_data, scope, include_charts, include_metadata, compress, split_files }) => {
            commands::handle_profile_export_cmd(&ctx, output_file, format, include_data, scope, include_charts, include_metadata, compress, split_files)?;
        }
        Some(Commands::Unified { action }) => match action {
            UnifiedAction::Status => {
                println!("🧠 Checking Unified Intelligence status...");

                let client = reqwest::blocking::Client::new();

                // Check gateway health endpoint for unified intelligence
                let response = client
                    .get(format!("{}/health/unified", GATEWAY_BASE))
                    .header("Authorization", format!("Bearer {}", GATEWAY_API_KEY))
                    .send();

                match response {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(health) = resp.json::<serde_json::Value>() {
                            let status = health["status"].as_str().unwrap_or("unknown");
                            let percentage = health["operational_percentage"].as_f64().unwrap_or(0.0);

                            if status == "healthy" {
                                println!("✓ Unified Intelligence is operational ({}% endpoints active)", percentage);
                            } else if status == "degraded" {
                                println!("⚠ Unified Intelligence is degraded ({}% endpoints active)", percentage);
                            } else {
                                println!("✗ Unified Intelligence is offline");
                            }

                            println!("\nEndpoints status:");
                            if let Some(endpoints) = health["endpoints"].as_object() {
                                for (name, status) in endpoints {
                                    let status_str = status.as_str().unwrap_or("unknown");
                                    let symbol = if status_str == "operational" { "✓" } else { "✗" };
                                    println!("  {} {}: {}", symbol, name, status_str);
                                }
                            }

                            println!("\nGateway API: {}", GATEWAY_BASE);
                        }
                    }
                    _ => {
                        println!("✗ Unified Intelligence is not responding");
                        println!("  Check if gateway is running at {}", GATEWAY_BASE);
                        println!("  Run 'omni services start' to start all services");
                    }
                }
            }
            UnifiedAction::Predict { context: _, profile: _ } => {
                println!("🔮 Prediction service has moved to gateway API");
                println!("  Gateway: {}", GATEWAY_BASE);
                println!("  Note: Direct predictions are now available through MCP tools only");
            }
            UnifiedAction::Suggest { show_now: _, context: _ } => {
                println!("💡 Suggestion service has moved to gateway API");
                println!("  Gateway: {}", GATEWAY_BASE);
                println!("  Note: Suggestions are now available through MCP tools only");
            }
            UnifiedAction::Feedback { suggestion_id, accepted, response_time, comment } => {
                println!("📝 Recording feedback...");

                let client = reqwest::blocking::Client::new();
                let body = serde_json::json!({
                    "suggestion_id": suggestion_id,
                    "accepted": accepted,
                    "response_time_ms": response_time,
                    "feedback": comment
                });

                match client.post("http://localhost:8003/unified/feedback").json(&body).send() {
                    Ok(resp) if resp.status().is_success() => {
                        println!("✓ Feedback recorded successfully");
                    }
                    _ => println!("✗ Failed to record feedback"),
                }
            }
        },

        Some(Commands::Services { action }) => match action {
            ServiceAction::Start => {
                println!("🚀 Starting OmniMemory services...");
                let output = std::process::Command::new("bash")
                    .arg("-c")
                    .arg(format!("{}/omnimemory.sh start", env!("CARGO_MANIFEST_DIR").replace("/omnimemory-cli", "")))
                    .output();

                match output {
                    Ok(o) if o.status.success() => {
                        println!("{}", String::from_utf8_lossy(&o.stdout));
                        println!("✓ Services started successfully");
                    }
                    _ => println!("✗ Failed to start services"),
                }
            }
            ServiceAction::Stop => {
                println!("🛑 Stopping OmniMemory services...");
                let output = std::process::Command::new("bash")
                    .arg("-c")
                    .arg(format!("{}/omnimemory.sh stop", env!("CARGO_MANIFEST_DIR").replace("/omnimemory-cli", "")))
                    .output();

                match output {
                    Ok(o) if o.status.success() => {
                        println!("✓ Services stopped successfully");
                    }
                    _ => println!("✗ Failed to stop services"),
                }
            }
            ServiceAction::Status => {
                println!("📊 Checking service status...");

                let client = reqwest::blocking::Client::new();
                let response = client
                    .get(format!("{}/health/system", GATEWAY_BASE))
                    .header("Authorization", format!("Bearer {}", GATEWAY_API_KEY))
                    .send();

                match response {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(health) = resp.json::<serde_json::Value>() {
                            println!("\n🔧 OmniMemory System Status");
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                            if let Some(services) = health["services"].as_object() {
                                println!("\n📡 Services:");
                                for (name, info) in services {
                                    let status = info["status"].as_str().unwrap_or("unknown");
                                    let port = info["port"].as_i64().unwrap_or(0);

                                    let symbol = match status {
                                        "healthy" | "running" => "✓",
                                        "offline" => "✗",
                                        _ => "⚠"
                                    };

                                    if port > 0 {
                                        println!("  {} {} (port {}): {}", symbol, name, port, status);
                                    } else {
                                        println!("  {} {}: {}", symbol, name, status);
                                    }
                                }
                            }

                            if let Some(overall) = health["overall_health"].as_object() {
                                let healthy = overall["healthy"].as_i64().unwrap_or(0);
                                let total = overall["total"].as_i64().unwrap_or(0);
                                let percentage = overall["percentage"].as_f64().unwrap_or(0.0);

                                println!("\n📊 Overall Health: {}/{} services healthy ({}%)",
                                        healthy, total, percentage);
                            }

                            println!("\nGateway API: {}", GATEWAY_BASE);
                        }
                    }
                    Ok(resp) => {
                        println!("⚠ Gateway responded with status: {}", resp.status());
                        println!("  Check API key configuration");
                    }
                    Err(e) => {
                        println!("✗ Cannot connect to gateway at {}", GATEWAY_BASE);
                        println!("  Error: {}", e);
                        println!("  Run 'omni services start' to start all services");
                    }
                }
            }
            ServiceAction::Restart => {
                println!("🔄 Restarting services...");
                let output = std::process::Command::new("bash")
                    .arg("-c")
                    .arg(format!("{}/omnimemory.sh restart", env!("CARGO_MANIFEST_DIR").replace("/omnimemory-cli", "")))
                    .output();

                match output {
                    Ok(o) if o.status.success() => {
                        println!("✓ Services restarted successfully");
                    }
                    _ => println!("✗ Failed to restart services"),
                }
            }
            ServiceAction::Logs { service, follow, lines } => {
                let service_name = service.unwrap_or_else(|| "all".to_string());
                println!("📋 Showing logs for: {}", service_name);

                let cmd = if follow {
                    format!("tail -f -n {} /tmp/omnimemory-{}.log", lines, service_name)
                } else {
                    format!("tail -n {} /tmp/omnimemory-{}.log", lines, service_name)
                };

                let _ = std::process::Command::new("bash")
                    .arg("-c")
                    .arg(cmd)
                    .status();
            }
            ServiceAction::Health => {
                println!("🏥 Running health checks...");

                let client = reqwest::blocking::Client::new();

                // Check gateway health
                println!("\nChecking gateway health...");
                match client.get(format!("{}/health", GATEWAY_BASE.replace("/api/v1", ""))).send() {
                    Ok(resp) if resp.status().is_success() => {
                        println!("✓ Gateway is healthy");
                    }
                    _ => {
                        println!("✗ Gateway is not responding");
                    }
                }

                // Check system health via gateway
                println!("\nChecking system health via gateway...");
                match client
                    .get(format!("{}/health/system", GATEWAY_BASE))
                    .header("Authorization", format!("Bearer {}", GATEWAY_API_KEY))
                    .send()
                {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(health) = resp.json::<serde_json::Value>() {
                            if let Some(percentage) = health["overall_health"]["percentage"].as_f64() {
                                if percentage >= 80.0 {
                                    println!("✓ System health: {}% - All good!", percentage);
                                } else if percentage >= 50.0 {
                                    println!("⚠ System health: {}% - Some services degraded", percentage);
                                } else {
                                    println!("✗ System health: {}% - Critical issues", percentage);
                                }
                            }
                        }
                    }
                    _ => {
                        println!("✗ Cannot retrieve system health from gateway");
                    }
                }

                // Check unified intelligence health
                println!("\nChecking unified intelligence health...");
                match client
                    .get(format!("{}/health/unified", GATEWAY_BASE))
                    .header("Authorization", format!("Bearer {}", GATEWAY_API_KEY))
                    .send()
                {
                    Ok(resp) if resp.status().is_success() => {
                        if let Ok(health) = resp.json::<serde_json::Value>() {
                            let status = health["status"].as_str().unwrap_or("unknown");
                            match status {
                                "healthy" => println!("✓ Unified Intelligence: Fully operational"),
                                "degraded" => println!("⚠ Unified Intelligence: Partially operational"),
                                "offline" => println!("✗ Unified Intelligence: Offline"),
                                _ => println!("? Unified Intelligence: Unknown status"),
                            }
                        }
                    }
                    _ => {
                        println!("✗ Cannot retrieve unified intelligence health");
                    }
                }
            }
        },

        Some(Commands::DashboardWeb { page }) => {
            let url = match page.as_deref() {
                Some("unified") => "http://localhost:8004/unified-intelligence",
                Some("metrics") => "http://localhost:8004/metrics",
                Some("settings") => "http://localhost:8004/settings",
                _ => "http://localhost:8004",
            };

            println!("🌐 Opening dashboard: {}", url);

            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(url).spawn();

            #[cfg(target_os = "linux")]
            let _ = std::process::Command::new("xdg-open").arg(url).spawn();

            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("cmd").args(["/c", "start", url]).spawn();
        }
    }

    Ok(())
}

fn load_config(config_path: &Option<PathBuf>) -> anyhow::Result<Config> {
    // Configuration loading logic will be implemented here
    Ok(Config::default())
}

#[derive(Debug, Clone, Default)]
pub struct Config {
    // Configuration fields will be added here
}