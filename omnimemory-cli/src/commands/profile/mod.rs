//! Context Profiling Commands for OmniMemory CLI
//! 
//! This module provides comprehensive context analysis capabilities including:
//! - Context profiling and analysis
//! - Token usage reporting  
//! - Tools performance monitoring
//! - Context breakdown and inspection
//! - Profile export functionality

pub mod profiler;
pub mod context_breakdown;
pub mod tokens_usage;
pub mod tools_performance;
pub mod profile_export;

pub use profiler::{handle_profile_main, ContextSegment};
pub use context_breakdown::{handle_context_breakdown, ContextBreakdown, ContextBreakdownReport, BreakdownDepth, GroupByField};
pub use tokens_usage::{handle_tokens_usage, TokenUsage, TokenUsageReport, GroupByOption, ModelCostConfig};
pub use tools_performance::{handle_tools_performance, ToolsPerformance, ToolsPerformanceReport, SortByMetric, MetricType};
pub use profile_export::{handle_profile_export, ExportConfig, ExportFormat, ExportScope, DataType, StyleOptions};

use crate::commands::{CommandContext, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main profiling command entry point that routes to appropriate subcommands
pub fn handle_profile_command<'de>(
    ctx: &CommandContext,
    subcommand: impl serde::Serialize + serde::Deserialize<'de>,
) -> Result<()> {
    let subcommand_str = serde_json::to_string(&subcommand)?;
    let subcommand: crate::ProfileSubcommand = serde_json::from_str(&subcommand_str)?;
    match subcommand {
        crate::ProfileSubcommand::Analyze { name, depth, include, exclude, generate_reports, output_file } => {
            let config = ProfileConfig {
                name,
                depth: match depth.as_str() {
                    "basic" => ProfileDepth::Basic,
                    "comprehensive" => ProfileDepth::Comprehensive,
                    _ => ProfileDepth::Detailed,
                },
                include,
                exclude,
                format: None,
                generate_reports,
                output_file,
            };
            handle_profile_main(ctx, Some(config))
        }
    }
}

/// Profile configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Profile name or identifier
    pub name: Option<String>,
    
    /// Analysis depth (basic, detailed, comprehensive)
    pub depth: ProfileDepth,
    
    /// Include specific analysis types
    pub include: Option<Vec<String>>,
    
    /// Exclude specific analysis types
    pub exclude: Option<Vec<String>>,
    
    /// Output format (json, human, table)
    pub format: Option<String>,
    
    /// Generate reports for export
    pub generate_reports: bool,
    
    /// Save analysis results to file
    pub output_file: Option<String>,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            name: None,
            depth: ProfileDepth::Detailed,
            include: None,
            exclude: None,
            format: None,
            generate_reports: true,
            output_file: None,
        }
    }
}

/// Profile analysis depth levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProfileDepth {
    Basic,
    Detailed,
    Comprehensive,
}

/// Main profile report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Profile identifier
    pub profile_id: String,
    
    /// Profile name
    pub profile_name: String,
    
    /// Analysis timestamp
    pub analyzed_at: chrono::DateTime<chrono::Utc>,
    
    /// Analysis depth performed
    pub depth: ProfileDepth,
    
    /// Context summary
    pub context_summary: ContextSummary,
    
    /// Token usage analysis
    pub token_usage: Option<tokens_usage::TokenUsageReport>,
    
    /// Tools performance metrics
    pub tools_performance: Option<tools_performance::ToolsPerformanceReport>,
    
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
    
    /// Overall health score (0-100)
    pub health_score: f32,
    
    /// Analysis metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Context summary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSummary {
    /// Total context segments
    pub total_segments: usize,
    
    /// Total tokens used
    pub total_tokens: usize,
    
    /// Context size in bytes
    pub total_size_bytes: usize,
    
    /// Average relevance score
    pub avg_relevance_score: f32,
    
    /// Context categories breakdown
    pub categories: HashMap<String, usize>,
    
    /// Time range covered
    pub time_range: String,
    
    /// Most active periods
    pub active_periods: Vec<String>,
}

/// Helper function to format duration in human-readable format
pub fn format_duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs();
    
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Helper function to calculate human-readable file size
pub fn format_file_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Helper function to calculate percentage with formatting
pub fn format_percentage(value: f32) -> String {
    format!("{:.1}%", value * 100.0)
}

/// Generate a unique profile ID
pub fn generate_profile_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let now = SystemTime::now();
    let since_epoch = now.duration_since(UNIX_EPOCH).unwrap();
    let nanos = since_epoch.subsec_nanos();
    
    format!("profile_{}_{}", since_epoch.as_secs(), nanos)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_secs(45)), "45s");
        assert_eq!(format_duration(std::time::Duration::from_secs(125)), "2m 5s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3725)), "1h 2m");
    }
    
    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
    }
    
    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.25), "25.0%");
        assert_eq!(format_percentage(0.5), "50.0%");
        assert_eq!(format_percentage(1.0), "100.0%");
    }
    
    #[test]
    fn test_generate_profile_id() {
        let id1 = generate_profile_id();
        let id2 = generate_profile_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("profile_"));
        assert!(id2.starts_with("profile_"));
    }
}