use super::Config;
use crate::{Cli, PrefScope};
use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

pub mod suggest;
pub mod why_failed;
pub mod context;
pub mod profile;
pub mod facts;
pub mod pref;
pub mod daemon;
pub mod doctor;
pub mod init;
pub mod snapshot;
pub mod default;
pub mod dashboard;
pub mod remove;

pub use suggest::handle_suggest;
pub use why_failed::handle_why_failed;
pub use context::handle_context;
pub use facts::handle_facts;
pub use pref::handle_pref;
pub use daemon::handle_daemon;
pub use doctor::handle_doctor;
pub use init::handle_init;
pub use default::handle_default;
pub use dashboard::handle_dashboard;
pub use remove::handle_remove;

// Profile module imports
pub use profile::handle_profile_command;
pub use profile::{handle_tokens_usage, handle_tools_performance, handle_profile_export};

// Convenience handlers for CLI integration

use super::OutputFormat;

pub struct CommandContext {
    pub cli: Cli,
    pub config: Config,
    pub output_format: OutputFormat,
}

impl CommandContext {
    pub fn new(cli: Cli, config: Config) -> Self {
        Self {
            output_format: cli.output.clone(),
            cli,
            config,
        }
    }

    /// Print output in the appropriate format
    pub fn print_output<T: serde::Serialize>(&self, output: &T) -> Result<()> {
        match self.output_format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(output)?;
                println!("{}", json);
            }
            OutputFormat::Human => self.print_human(output)?,
            OutputFormat::Plain | OutputFormat::Table => {
                let json = serde_json::to_string(output)?;
                println!("{}", json);
            }
        }
        Ok(())
    }

    fn print_human<T: serde::Serialize>(&self, output: &T) -> Result<()> {
        // This is a simple implementation - in a real implementation,
        // you'd have specific formatting for different output types
        let json = serde_json::to_string_pretty(output)?;
        println!("{}", json);
        Ok(())
    }

    /// Create a progress bar for long-running operations
    pub fn create_progress(&self, message: &str) -> ProgressBar {
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{wide_bar}] {pos}/{len}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Print a success message
    pub fn print_success(&self, message: &str) {
        if !self.cli.quiet {
            println!("{} {}", "✓".green(), message);
        }
    }

    /// Print a warning message
    pub fn print_warning(&self, message: &str) {
        if !self.cli.quiet {
            eprintln!("{} {}", "⚠".yellow(), message);
        }
    }

    /// Print an error message
    pub fn print_error(&self, message: &str) {
        eprintln!("{} {}", "✗".red(), message);
    }

    /// Check if we're in interactive mode
    pub fn is_interactive(&self) -> bool {
        !self.cli.no_input && atty::is(atty::Stream::Stdout)
    }
}

/// Wrapper function for handling context breakdown CLI arguments
pub fn handle_context_breakdown_cmd(
    ctx: &CommandContext,
    depth: String,
    include_types: Option<Vec<String>>,
    exclude_types: Option<Vec<String>>,
    time_range: Option<String>,
    min_relevance: Option<f32>,
    show_details: bool,
    group_by: Option<String>,
    generate_stats: bool,
) -> Result<()> {
    use profile::{ContextBreakdown, BreakdownDepth, GroupByField};
    
    let breakdown_config = ContextBreakdown {
        depth: match depth.as_str() {
            "basic" => BreakdownDepth::Basic,
            "comprehensive" => BreakdownDepth::Comprehensive,
            _ => BreakdownDepth::Detailed,
        },
        include_types: include_types.unwrap_or_default(),
        exclude_types: exclude_types.unwrap_or_default(),
        time_range,
        min_relevance,
        group_by: group_by.as_ref().map(|s| match s.as_str() {
            "type" => GroupByField::Type,
            "source" => GroupByField::Source,
            "timestamp" => GroupByField::Timestamp,
            "size" => GroupByField::Size,
            "relevance" => GroupByField::Relevance,
            _ => GroupByField::Type,
        }),
        show_details,
        generate_stats,
    };

    profile::handle_context_breakdown(ctx, Some(breakdown_config))
}

/// Wrapper function for handling tokens usage CLI arguments
pub fn handle_tokens_usage_cmd(
    ctx: &CommandContext,
    scope: String,
    time_period: String,
    token_types: Option<Vec<String>>,
    group_by: Option<String>,
    include_costs: bool,
    budget_limit: Option<f32>,
    detailed_breakdown: bool,
    generate_optimizations: bool,
) -> Result<()> {
    use profile::{TokenUsage, GroupByOption, ModelCostConfig};
    
    let usage_config = TokenUsage {
        scope,
        time_period,
        token_types: token_types.unwrap_or_default(),
        group_by: group_by.as_ref().map(|s| match s.as_str() {
            "date" => GroupByOption::Date,
            "hour" => GroupByOption::Hour,
            "operation_type" => GroupByOption::OperationType,
            "model" => GroupByOption::Model,
            "user" => GroupByOption::User,
            "session" => GroupByOption::Session,
            _ => GroupByOption::Date,
        }),
        include_costs,
        budget_limit,
        detailed_breakdown,
        generate_optimizations,
        model_config: ModelCostConfig::default(),
    };

    handle_tokens_usage(ctx, Some(usage_config))
}

/// Wrapper function for handling tools performance CLI arguments
pub fn handle_tools_performance_cmd(
    ctx: &CommandContext,
    analysis_depth: String,
    time_window: String,
    target_tools: Option<Vec<String>>,
    sort_by: Option<String>,
    min_usage_threshold: Option<usize>,
    detailed_breakdown: bool,
    generate_optimizations: bool,
) -> Result<()> {
    use profile::{ToolsPerformance, SortByMetric, MetricType};

    let performance_config = ToolsPerformance {
        analysis_depth,
        time_window,
        target_tools: target_tools.unwrap_or_default(),
        include_metrics: vec![
            MetricType::UsageCount,
            MetricType::ResponseTime,
            MetricType::SuccessRate,
            MetricType::ErrorRate,
            MetricType::ImpactScore,
        ],
        sort_by: sort_by.as_ref().map(|s| match s.as_str() {
            "usage_count" => SortByMetric::UsageCount,
            "response_time" => SortByMetric::ResponseTime,
            "success_rate" => SortByMetric::SuccessRate,
            "impact_score" => SortByMetric::ImpactScore,
            "alphabetical" => SortByMetric::Alphabetical,
            _ => SortByMetric::ImpactScore,
        }),
        min_usage_threshold,
        detailed_breakdown,
        generate_optimizations,
        focus_categories: None,
    };

    handle_tools_performance(ctx, Some(performance_config))
}

/// Wrapper function for handling profile export CLI arguments
pub fn handle_profile_export_cmd(
    ctx: &CommandContext,
    output_file: String,
    format: String,
    include_data: Option<Vec<String>>,
    scope: String,
    include_charts: bool,
    include_metadata: bool,
    compress: bool,
    split_files: bool,
) -> Result<()> {
    use profile::{ExportConfig, ExportFormat, ExportScope, DataType, StyleOptions};
    
    let export_config = ExportConfig {
        output_file,
        format: match format.as_str() {
            "csv" => ExportFormat::Csv,
            "html" => ExportFormat::Html,
            "pdf" => ExportFormat::Pdf,
            "xml" => ExportFormat::Xml,
            "yaml" => ExportFormat::Yaml,
            _ => ExportFormat::Json,
        },
        include_data: include_data.unwrap_or_default().into_iter().map(|s| match s.as_str() {
            "profile_report" => DataType::ProfileReport,
            "token_usage" => DataType::TokenUsage,
            "tools_performance" => DataType::ToolsPerformance,
            "context_breakdown" => DataType::ContextBreakdown,
            "system_metrics" => DataType::SystemMetrics,
            "recommendations" => DataType::Recommendations,
            "export_summary" => DataType::ExportSummary,
            _ => DataType::ProfileReport,
        }).collect(),
        scope: match scope.as_str() {
            "summary" => ExportScope::Summary,
            "comprehensive" => ExportScope::Comprehensive,
            _ => ExportScope::Detailed,
        },
        template: None,
        include_charts,
        include_metadata,
        compress,
        split_files,
        style_options: StyleOptions::default(),
    };

    handle_profile_export(ctx, Some(export_config))
}

/// Stub handler for user profile management (not yet implemented)
pub fn handle_profile(
    _ctx: &CommandContext,
    set: Option<String>,
    show: bool,
    create: Option<String>,
    delete: Option<String>,
    list: bool,
) -> Result<()> {
    println!("User profile management not yet implemented.");
    println!("This command is for managing user profiles (set/show/create/delete/list).");
    println!("For context profiling analysis, use 'omni profiler' instead.");
    Ok(())
}