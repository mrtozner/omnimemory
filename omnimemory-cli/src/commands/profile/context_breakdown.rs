//! Context Breakdown and Inspection Module
//!
//! This module provides detailed analysis and breakdown of context usage,
//! segment inspection, and hierarchical context analysis.

use crate::commands::{CommandContext, Result};
use super::{ContextSegment, format_file_size};
use chrono::Timelike;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Handle context breakdown command
pub fn handle_context_breakdown(
    ctx: &CommandContext,
    breakdown: Option<ContextBreakdown>,
) -> Result<()> {
    let config = breakdown.unwrap_or_default();
    
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Analyzing context breakdown..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Collecting context segments...");
        pb.set_position(30);
    }

    // Gather context segments from storage and system
    let segments = gather_context_segments(&config)?;

    if let Some(ref pb) = pb {
        pb.set_message("Categorizing segments...");
        pb.set_position(60);
    }

    // Categorize and analyze segments
    let categorized = categorize_segments(&segments);

    if let Some(ref pb) = pb {
        pb.set_message("Generating breakdown report...");
        pb.set_position(90);
    }

    // Create breakdown report
    // Analyze before moving segments
    let total_segments = segments.len();
    let size_analysis = analyze_size_distribution(&segments);
    let time_analysis = analyze_temporal_patterns(&segments);
    let recommendation_summary = generate_breakdown_recommendations(&segments);

    let report = ContextBreakdownReport {
        analysis_timestamp: chrono::Utc::now(),
        total_segments,
        categories: categorized,
        segment_details: segments,
        size_analysis,
        time_analysis,
        recommendation_summary,
        metadata: HashMap::from([
            ("analysis_depth".to_string(), serde_json::Value::String(format!("{:?}", config.depth))),
            ("include_types".to_string(), serde_json::to_value(&config.include_types)?),
        ]),
    };

    if let Some(ref pb) = pb {
        pb.set_message("Breakdown analysis complete");
        pb.finish_with_message("✓ Breakdown ready");
    }

    // Handle output
    match ctx.output_format {
        super::super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        super::super::OutputFormat::Human => {
            print_human_breakdown(&report, &config);
        }
        _ => {
            println!("{}", serde_json::to_string(&report)?);
        }
    }

    Ok(())
}

/// Context breakdown configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBreakdown {
    /// Analysis depth (basic, detailed, comprehensive)
    pub depth: BreakdownDepth,
    
    /// Specific context types to include
    pub include_types: Vec<String>,
    
    /// Context types to exclude
    pub exclude_types: Vec<String>,
    
    /// Time range for analysis (e.g., "1h", "1d", "1w")
    pub time_range: Option<String>,
    
    /// Minimum relevance score threshold
    pub min_relevance: Option<f32>,
    
    /// Group by field (type, source, timestamp, size)
    pub group_by: Option<GroupByField>,
    
    /// Show detailed segment information
    pub show_details: bool,
    
    /// Generate summary statistics
    pub generate_stats: bool,
}

impl Default for ContextBreakdown {
    fn default() -> Self {
        Self {
            depth: BreakdownDepth::Detailed,
            include_types: vec![],
            exclude_types: vec![],
            time_range: None,
            min_relevance: Some(0.5),
            group_by: None,
            show_details: true,
            generate_stats: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BreakdownDepth {
    Basic,
    Detailed,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupByField {
    Type,
    Source,
    Timestamp,
    Size,
    Relevance,
}

/// Main breakdown report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBreakdownReport {
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
    pub total_segments: usize,
    pub categories: HashMap<String, Vec<ContextSegment>>,
    pub segment_details: Vec<ContextSegment>,
    pub size_analysis: SizeAnalysis,
    pub time_analysis: TemporalAnalysis,
    pub recommendation_summary: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Size distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeAnalysis {
    pub total_size_bytes: usize,
    pub average_size_bytes: f32,
    pub min_size_bytes: usize,
    pub max_size_bytes: usize,
    pub size_distribution: HashMap<String, usize>, // small, medium, large, huge
    pub size_percentiles: HashMap<String, f32>, // p50, p75, p90, p99
}

/// Temporal pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub time_range: String,
    pub segments_per_minute: f32,
    pub peak_activity_periods: Vec<String>,
    pub activity_distribution: HashMap<String, usize>, // hour -> count
    pub oldest_segment: chrono::DateTime<chrono::Utc>,
    pub newest_segment: chrono::DateTime<chrono::Utc>,
}

/// Gather context segments from various sources
fn gather_context_segments(config: &ContextBreakdown) -> Result<Vec<ContextSegment>> {
    // Mock context segments with different types and characteristics
    let segments = vec![
        ContextSegment {
            id: "seg_001".to_string(),
            r#type: "command_history".to_string(),
            content: "git add . && git commit -m 'feat: add context profiling' && git push origin main".to_string(),
            size_bytes: 89,
            relevance_score: 0.95,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(5),
            source: "shell_integration".to_string(),
            metadata: HashMap::from([
                ("success".to_string(), serde_json::Value::Bool(true)),
                ("execution_time_ms".to_string(), serde_json::Value::Number(1250.into())),
                ("tools_used".to_string(), serde_json::to_value(vec!["git"])?),
            ]),
        },
        ContextSegment {
            id: "seg_002".to_string(),
            r#type: "file_edit".to_string(),
            content: "src/commands/profile.rs: updated handle_profile_main function with async processing".to_string(),
            size_bytes: 112,
            relevance_score: 0.92,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(8),
            source: "editor_integration".to_string(),
            metadata: HashMap::from([
                ("file_type".to_string(), serde_json::Value::String("rust".to_string())),
                ("lines_modified".to_string(), serde_json::Value::Number(45.into())),
                ("language".to_string(), serde_json::Value::String("rust".to_string())),
            ]),
        },
        ContextSegment {
            id: "seg_003".to_string(),
            r#type: "git_status".to_string(),
            content: "On branch main\\nYour branch is up to date with 'origin/main'.\\n\\nnothing to commit, working tree clean".to_string(),
            size_bytes: 95,
            relevance_score: 0.78,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(12),
            source: "git_daemon".to_string(),
            metadata: HashMap::from([
                ("branch".to_string(), serde_json::Value::String("main".to_string())),
                ("clean_working_tree".to_string(), serde_json::Value::Bool(true)),
                ("uncommitted_changes".to_string(), serde_json::Value::Number(0.into())),
            ]),
        },
        ContextSegment {
            id: "seg_004".to_string(),
            r#type: "environment_scan".to_string(),
            content: "Rust 1.70.0, Cargo 1.70.0, Git 2.40.0, Docker 24.0.0".to_string(),
            size_bytes: 65,
            relevance_score: 0.65,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(15),
            source: "environment_monitor".to_string(),
            metadata: HashMap::from([
                ("rust_version".to_string(), serde_json::Value::String("1.70.0".to_string())),
                ("cargo_version".to_string(), serde_json::Value::String("1.70.0".to_string())),
                ("tools_count".to_string(), serde_json::Value::Number(4.into())),
            ]),
        },
        ContextSegment {
            id: "seg_005".to_string(),
            r#type: "terminal_output".to_string(),
            content: "Compiling omnimemory-cli v0.1.0\\nFinished dev [unoptimized + debuginfo] target(s) in 2.5s".to_string(),
            size_bytes: 78,
            relevance_score: 0.71,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(18),
            source: "build_system".to_string(),
            metadata: HashMap::from([
                ("build_success".to_string(), serde_json::Value::Bool(true)),
                ("build_time_ms".to_string(), serde_json::Value::Number(2500.into())),
                ("optimization_level".to_string(), serde_json::Value::String("dev".to_string())),
            ]),
        },
        ContextSegment {
            id: "seg_006".to_string(),
            r#type: "error_log".to_string(),
            content: "Connection timeout to storage service after 5s".to_string(),
            size_bytes: 48,
            relevance_score: 0.35,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(25),
            source: "error_monitor".to_string(),
            metadata: HashMap::from([
                ("error_type".to_string(), serde_json::Value::String("timeout".to_string())),
                ("severity".to_string(), serde_json::Value::String("warning".to_string())),
                ("resolved".to_string(), serde_json::Value::Bool(true)),
            ]),
        },
        ContextSegment {
            id: "seg_007".to_string(),
            r#type: "search_result".to_string(),
            content: "Found 12 relevant files matching 'profile' query in 245ms".to_string(),
            size_bytes: 67,
            relevance_score: 0.82,
            timestamp: chrono::Utc::now() - chrono::Duration::minutes(22),
            source: "search_engine".to_string(),
            metadata: HashMap::from([
                ("query".to_string(), serde_json::Value::String("profile".to_string())),
                ("result_count".to_string(), serde_json::Value::Number(12.into())),
                ("search_time_ms".to_string(), serde_json::Value::Number(245.into())),
            ]),
        },
    ];

    // Apply filtering based on config
    let mut filtered_segments = segments;
    
    if !config.include_types.is_empty() {
        filtered_segments.retain(|seg| config.include_types.contains(&seg.r#type));
    }
    
    if !config.exclude_types.is_empty() {
        filtered_segments.retain(|seg| !config.exclude_types.contains(&seg.r#type));
    }
    
    if let Some(min_relevance) = config.min_relevance {
        filtered_segments.retain(|seg| seg.relevance_score >= min_relevance);
    }

    Ok(filtered_segments)
}

/// Categorize segments by type
fn categorize_segments(segments: &Vec<ContextSegment>) -> HashMap<String, Vec<ContextSegment>> {
    let mut categories = HashMap::new();
    
    for segment in segments {
        categories
            .entry(segment.r#type.clone())
            .or_insert_with(Vec::new)
            .push(segment.clone());
    }
    
    categories
}

/// Analyze size distribution
fn analyze_size_distribution(segments: &Vec<ContextSegment>) -> SizeAnalysis {
    if segments.is_empty() {
        return SizeAnalysis {
            total_size_bytes: 0,
            average_size_bytes: 0.0,
            min_size_bytes: 0,
            max_size_bytes: 0,
            size_distribution: HashMap::new(),
            size_percentiles: HashMap::new(),
        };
    }

    let sizes: Vec<usize> = segments.iter().map(|s| s.size_bytes).collect();
    let total_size = sizes.iter().sum();
    let avg_size = total_size as f32 / sizes.len() as f32;
    let min_size = sizes.iter().min().copied().unwrap_or(0);
    let max_size = sizes.iter().max().copied().unwrap_or(0);

    // Sort for percentile calculation
    let mut sorted_sizes = sizes.clone();
    sorted_sizes.sort();
    
    let p50 = if !sorted_sizes.is_empty() {
        sorted_sizes[(sorted_sizes.len() * 50) / 100]
    } else { 0 };
    
    let p75 = if !sorted_sizes.is_empty() {
        sorted_sizes[(sorted_sizes.len() * 75) / 100]
    } else { 0 };
    
    let p90 = if !sorted_sizes.is_empty() {
        sorted_sizes[(sorted_sizes.len() * 90) / 100]
    } else { 0 };
    
    let p99 = if !sorted_sizes.is_empty() {
        sorted_sizes[(sorted_sizes.len() * 99) / 100]
    } else { 0 };

    let mut distribution = HashMap::new();
    distribution.insert("tiny".to_string(), sizes.iter().filter(|&&s| s < 50).count());
    distribution.insert("small".to_string(), sizes.iter().filter(|&&s| s >= 50 && s < 100).count());
    distribution.insert("medium".to_string(), sizes.iter().filter(|&&s| s >= 100 && s < 200).count());
    distribution.insert("large".to_string(), sizes.iter().filter(|&&s| s >= 200).count());

    let percentiles = HashMap::from([
        ("p50".to_string(), p50 as f32),
        ("p75".to_string(), p75 as f32),
        ("p90".to_string(), p90 as f32),
        ("p99".to_string(), p99 as f32),
    ]);

    SizeAnalysis {
        total_size_bytes: total_size,
        average_size_bytes: avg_size,
        min_size_bytes: min_size,
        max_size_bytes: max_size,
        size_distribution: distribution,
        size_percentiles: percentiles,
    }
}

/// Analyze temporal patterns
fn analyze_temporal_patterns(segments: &Vec<ContextSegment>) -> TemporalAnalysis {
    if segments.is_empty() {
        return TemporalAnalysis {
            time_range: "none".to_string(),
            segments_per_minute: 0.0,
            peak_activity_periods: vec![],
            activity_distribution: HashMap::new(),
            oldest_segment: chrono::Utc::now(),
            newest_segment: chrono::Utc::now(),
        };
    }

    let mut timestamps: Vec<chrono::DateTime<chrono::Utc>> = segments.iter()
        .map(|s| s.timestamp)
        .collect();
    
    timestamps.sort();
    
    let oldest = timestamps.first().copied().unwrap_or(chrono::Utc::now());
    let newest = timestamps.last().copied().unwrap_or(chrono::Utc::now());
    
    let duration = newest.signed_duration_since(oldest);
    let minutes = duration.num_minutes().max(1);
    
    let segments_per_minute = segments.len() as f32 / minutes as f32;
    
    // Group by hour for activity distribution
    let mut activity_distribution = HashMap::new();
    for segment in segments {
        let hour = segment.timestamp.hour();
        *activity_distribution.entry(hour.to_string()).or_insert(0) += 1;
    }
    
    // Find peak activity periods
    let mut peak_periods = Vec::new();
    let avg_activity = activity_distribution.values().sum::<usize>() as f32 / activity_distribution.len() as f32;
    
    for (hour, count) in &activity_distribution {
        if *count as f32 > avg_activity * 1.5 {
            peak_periods.push(format!("{}:00", hour));
        }
    }

    TemporalAnalysis {
        time_range: format!("{} minutes", minutes),
        segments_per_minute,
        peak_activity_periods: peak_periods,
        activity_distribution,
        oldest_segment: oldest,
        newest_segment: newest,
    }
}

/// Generate recommendations based on breakdown analysis
fn generate_breakdown_recommendations(segments: &Vec<ContextSegment>) -> Vec<String> {
    let mut recommendations = Vec::new();

    if segments.len() > 20 {
        recommendations.push("High number of context segments detected - consider pruning low-relevance items".to_string());
    }

    let avg_relevance = segments.iter()
        .map(|s| s.relevance_score)
        .sum::<f32>() / segments.len() as f32;

    if avg_relevance < 0.7 {
        recommendations.push("Improve context relevance filtering - average relevance is low".to_string());
    }

    // Check for error/issue segments
    let error_segments = segments.iter()
        .filter(|s| s.r#type == "error_log" || s.r#type == "failure")
        .count();

    if error_segments > 0 {
        recommendations.push(format!("{} error/failure segments detected - review system stability", error_segments));
    }

    // Check for size outliers
    let sizes: Vec<usize> = segments.iter().map(|s| s.size_bytes).collect();
    if !sizes.is_empty() {
        let avg_size = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let oversized = segments.iter().filter(|s| s.size_bytes as f32 > avg_size * 3.0).count();
        
        if oversized > 0 {
            recommendations.push(format!("{} unusually large segments found - consider splitting", oversized));
        }
    }

    if recommendations.is_empty() {
        recommendations.push("Context breakdown analysis shows healthy patterns".to_string());
    }

    recommendations
}

/// Print human-readable breakdown report
fn print_human_breakdown(report: &ContextBreakdownReport, config: &ContextBreakdown) {
    println!("\n{}", "🔍 Context Breakdown Analysis".bold().blue());
    println!("Total Segments: {}", report.total_segments);
    println!("Time Range: {}", report.time_analysis.time_range);
    println!("Activity Rate: {:.1} segments/minute\n", report.time_analysis.segments_per_minute);

    println!("{}", "Size Analysis".bold().magenta());
    println!("  Total Size: {}", format_file_size(report.size_analysis.total_size_bytes));
    println!("  Average Size: {} bytes", report.size_analysis.average_size_bytes.round() as usize);
    println!("  Size Range: {} - {}", 
        format_file_size(report.size_analysis.min_size_bytes),
        format_file_size(report.size_analysis.max_size_bytes)
    );
    
    println!("\n  Size Distribution:");
    for (size_cat, count) in &report.size_analysis.size_distribution {
        println!("    {}: {} segments", size_cat, count);
    }
    
    if !report.size_analysis.size_percentiles.is_empty() {
        println!("\n  Percentiles:");
        for (percentile, size) in &report.size_analysis.size_percentiles {
            println!("    {}: {} bytes", percentile, *size as usize);
        }
    }
    println!();

    println!("{}", "Categories".bold().magenta());
    let mut category_totals = Vec::new();
    for (category, segments) in &report.categories {
        let total_size: usize = segments.iter().map(|s| s.size_bytes).sum();
        let avg_relevance = segments.iter().map(|s| s.relevance_score).sum::<f32>() / segments.len() as f32;
        category_totals.push((category, segments.len(), total_size, avg_relevance));
    }
    
    // Sort by segment count
    category_totals.sort_by(|a, b| b.1.cmp(&a.1));
    
    for (category, count, total_size, avg_relevance) in category_totals {
        println!("  {}: {} segments ({} bytes, {:.1}% relevance)", 
            category.cyan(), count, format_file_size(total_size), avg_relevance * 100.0);
    }
    println!();

    if !report.time_analysis.peak_activity_periods.is_empty() {
        println!("{}", "Peak Activity".bold().magenta());
        println!("  Active periods: {}", report.time_analysis.peak_activity_periods.join(", "));
        println!();
    }

    println!("{}", "Detailed Segments".bold().magenta());
    if config.show_details {
        for (i, segment) in report.segment_details.iter().enumerate() {
            println!("  {}. [{}] {} ({} bytes, {:.1}% relevance)",
                i + 1,
                segment.r#type.cyan(),
                segment.source,
                format_file_size(segment.size_bytes),
                segment.relevance_score * 100.0
            );
            
            if !segment.content.is_empty() && segment.content.len() < 100 {
                println!("     Content: {}", segment.content.dimmed());
            }
            
            println!("     Source: {}, Time: {}", 
                segment.source,
                segment.timestamp.format("%H:%M").to_string()
            );
            println!();
        }
    } else {
        println!("  (Details hidden - use --show-details to display)");
    }

    println!("{}", "Recommendations".bold().magenta());
    for (i, rec) in report.recommendation_summary.iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }

    println!("\nAnalyzed at: {}", report.analysis_timestamp.format("%Y-%m-%d %H:%M:%S"));
}