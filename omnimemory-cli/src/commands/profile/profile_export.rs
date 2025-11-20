//! Profile Export Module
//!
//! This module handles exporting context profiling analysis reports in various
//! formats including JSON, CSV, HTML, and PDF with comprehensive formatting.

use crate::commands::{CommandContext, Result};
use super::{ProfileReport, TokenUsageReport, ToolsPerformanceReport, ContextBreakdownReport};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Handle profile export command
pub fn handle_profile_export(
    ctx: &CommandContext,
    export: Option<ExportConfig>,
) -> Result<()> {
    let config = export.unwrap_or_default();
    
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Preparing export..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Gathering profile data...");
        pb.set_position(20);
    }

    // Gather profile data from various sources
    let profile_data = gather_profile_data(&config)?;

    if let Some(ref pb) = pb {
        pb.set_message("Formatting export data...");
        pb.set_position(50);
    }

    // Format data according to export configuration
    let export_data = format_export_data(&profile_data, &config)?;

    if let Some(ref pb) = pb {
        pb.set_message("Generating export file...");
        pb.set_position(80);
    }

    // Generate export files
    let export_result = generate_export_files(&export_data, &config)?;

    if let Some(ref pb) = pb {
        pb.set_message("Export complete");
        pb.finish_with_message("✓ Export ready");
    }

    // Handle output
    match ctx.output_format {
        super::super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&export_result)?);
        }
        super::super::OutputFormat::Human => {
            print_human_export_info(&export_result, &config);
        }
        _ => {
            println!("{}", serde_json::to_string(&export_result)?);
        }
    }

    Ok(())
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output file path
    pub output_file: String,
    
    /// Export format (json, csv, html, pdf, xml)
    pub format: ExportFormat,
    
    /// Include specific data types in export
    pub include_data: Vec<DataType>,
    
    /// Export scope (summary, detailed, comprehensive)
    pub scope: ExportScope,
    
    /// Custom template for export formatting
    pub template: Option<String>,
    
    /// Include charts and visualizations (for HTML/PDF)
    pub include_charts: bool,
    
    /// Include metadata and technical details
    pub include_metadata: bool,
    
    /// Compress output files
    pub compress: bool,
    
    /// Generate multiple files (one per data type)
    pub split_files: bool,
    
    /// Custom styling options
    pub style_options: StyleOptions,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            output_file: "profile_export".to_string(),
            format: ExportFormat::Json,
            include_data: vec![
                DataType::ProfileReport,
                DataType::TokenUsage,
                DataType::ToolsPerformance,
                DataType::ContextBreakdown,
            ],
            scope: ExportScope::Detailed,
            template: None,
            include_charts: true,
            include_metadata: true,
            compress: false,
            split_files: false,
            style_options: StyleOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Pdf,
    Xml,
    Yaml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportScope {
    Summary,
    Detailed,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    ProfileReport,
    TokenUsage,
    ToolsPerformance,
    ContextBreakdown,
    SystemMetrics,
    Recommendations,
    ExportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleOptions {
    pub theme: String,
    pub color_scheme: String,
    pub font_family: String,
    pub include_logo: bool,
    pub custom_css: Option<String>,
    pub chart_colors: Vec<String>,
}

impl Default for StyleOptions {
    fn default() -> Self {
        Self {
            theme: "professional".to_string(),
            color_scheme: "blue".to_string(),
            font_family: "Arial".to_string(),
            include_logo: true,
            custom_css: None,
            chart_colors: vec![
                "#2563eb".to_string(), // Blue
                "#7c3aed".to_string(), // Purple
                "#059669".to_string(), // Green
                "#dc2626".to_string(), // Red
                "#d97706".to_string(), // Orange
            ],
        }
    }
}

/// Unified profile data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileExportData {
    pub metadata: ExportMetadata,
    pub profile_report: Option<ProfileReport>,
    pub token_usage_report: Option<TokenUsageReport>,
    pub tools_performance_report: Option<ToolsPerformanceReport>,
    pub context_breakdown_report: Option<ContextBreakdownReport>,
    pub system_metrics: SystemMetricsExport,
    pub recommendations: Vec<String>,
    pub export_config: ExportConfig,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub export_timestamp: chrono::DateTime<chrono::Utc>,
    pub export_format: ExportFormat,
    pub export_scope: ExportScope,
    pub data_sources: Vec<String>,
    pub total_reports_included: usize,
    pub file_size_estimate: usize,
    pub version: String,
    pub generator: String,
    pub checksum: Option<String>,
}

/// System metrics export structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsExport {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub disk_usage: f32,
    pub network_io: f32,
    pub process_count: usize,
    pub uptime_hours: f32,
    pub system_load: f32,
    pub environment: String,
    pub hardware_info: HashMap<String, String>,
}

/// Export result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub export_timestamp: chrono::DateTime<chrono::Utc>,
    pub files_created: Vec<ExportFile>,
    pub total_size_bytes: usize,
    pub export_duration_ms: u64,
    pub success: bool,
    pub summary: ExportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFile {
    pub file_path: String,
    pub file_type: ExportFormat,
    pub size_bytes: usize,
    pub data_types_included: Vec<DataType>,
    pub compression_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSummary {
    pub total_exports: usize,
    pub successful_exports: usize,
    pub failed_exports: usize,
    pub total_records_exported: usize,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Gather profile data from various sources
fn gather_profile_data(config: &ExportConfig) -> Result<ProfileExportData> {
    // Mock data gathering - in real implementation, this would collect from storage/adapters
    let profile_report = if config.include_data.contains(&DataType::ProfileReport) {
        Some(ProfileReport {
            profile_id: "profile_001".to_string(),
            profile_name: "default".to_string(),
            analyzed_at: chrono::Utc::now(),
            depth: super::ProfileDepth::Detailed,
            context_summary: super::ContextSummary {
                total_segments: 25,
                total_tokens: 1250,
                total_size_bytes: 5430,
                avg_relevance_score: 0.82,
                categories: HashMap::from([
                    ("commands".to_string(), 8),
                    ("git".to_string(), 6),
                    ("files".to_string(), 11),
                ]),
                time_range: "2h".to_string(),
                active_periods: vec!["last_30_min".to_string(), "last_2h".to_string()],
            },
            token_usage: None,
            tools_performance: None,
            recommendations: vec![
                "Consider context compression for efficiency".to_string(),
                "Optimize file operation caching".to_string(),
            ],
            health_score: 85.2,
            metadata: HashMap::new(),
        })
    } else {
        None
    };

    let token_usage_report = if config.include_data.contains(&DataType::TokenUsage) {
        Some(TokenUsageReport {
            report_timestamp: chrono::Utc::now(),
            usage_summary: super::tokens_usage::TokenUsageData {
                total_tokens: 5420,
                input_tokens: 3800,
                output_tokens: 1200,
                context_tokens: 420,
                system_tokens: 0,
                usage_by_type: HashMap::from([
                    ("analysis".to_string(), 2100),
                    ("generation".to_string(), 1800),
                    ("processing".to_string(), 1520),
                ]),
                usage_by_source: HashMap::from([
                    ("cli".to_string(), 4200),
                    ("api".to_string(), 1220),
                ]),
                usage_timeline: vec![],
                peak_usage: super::tokens_usage::UsagePeak {
                    timestamp: chrono::Utc::now() - chrono::Duration::hours(2),
                    token_count: 890,
                    operation_type: "analysis".to_string(),
                },
                average_usage_per_hour: 135.5,
                efficiency_score: 0.89,
                metadata: HashMap::new(),
            },
            efficiency_metrics: super::tokens_usage::EfficiencyMetrics {
                token_efficiency: 0.78,
                context_utilization: 0.65,
                compression_ratio: 0.82,
                waste_percentage: 0.15,
                optimization_potential: 0.25,
                cost_efficiency: 0.85,
                performance_score: 78.5,
            },
            cost_analysis: super::tokens_usage::CostAnalysis {
                total_cost: 2.45,
                cost_breakdown: HashMap::from([
                    ("analysis".to_string(), 1.2),
                    ("generation".to_string(), 0.9),
                    ("processing".to_string(), 0.35),
                ]),
                cost_per_hour: 0.41,
                cost_per_operation: 0.0005,
                projected_monthly_cost: 24.50,
                cost_trends: vec![],
                budget_utilization: 0.35,
                cost_optimization_opportunities: vec![
                    "Implement context compression".to_string(),
                    "Use batch processing for similar operations".to_string(),
                ],
            },
            optimization_recommendations: vec![
                "Implement token budgeting controls".to_string(),
                "Optimize context summarization".to_string(),
            ],
            budget_status: Some(super::tokens_usage::BudgetStatus {
                budget_limit: 50.0,
                current_usage: 17.5,
                remaining_budget: 32.5,
                utilization_percentage: 0.35,
                projected_usage: 22.8,
                over_budget_risk: false,
                estimated_days_remaining: 25.0,
            }),
            historical_trends: HashMap::new(),
            metadata: HashMap::new(),
        })
    } else {
        None
    };

    let tools_performance_report = if config.include_data.contains(&DataType::ToolsPerformance) {
        Some(ToolsPerformanceReport {
            report_timestamp: chrono::Utc::now(),
            performance_summary: super::tools_performance::ToolsPerformanceData {
                tools: HashMap::new(),
                overall_stats: super::tools_performance::OverallPerformanceStats {
                    total_tools: 6,
                    active_tools: 5,
                    total_operations: 1250,
                    total_response_time_ms: 89500.0,
                    average_response_time_ms: 716.0,
                    overall_success_rate: 0.94,
                    system_health_score: 8.2,
                    peak_usage_time: chrono::Utc::now() - chrono::Duration::hours(1),
                    busiest_tool: "git".to_string(),
                },
                usage_timeline: vec![],
                performance_trends: HashMap::new(),
                error_analysis: super::tools_performance::ErrorAnalysis {
                    total_errors: 75,
                    error_rate: 0.06,
                    common_errors: HashMap::from([
                        ("timeout".to_string(), 28),
                        ("permission_denied".to_string(), 22),
                        ("resource_exhausted".to_string(), 15),
                        ("network_error".to_string(), 10),
                    ]),
                    error_by_tool: HashMap::from([
                        ("docker".to_string(), 35),
                        ("cargo".to_string(), 25),
                        ("git".to_string(), 10),
                        ("ls".to_string(), 5),
                    ]),
                    error_trends: vec![],
                    reliability_concerns: vec![
                        "Docker builds showing degraded performance".to_string(),
                    ],
                },
                resource_utilization: super::tools_performance::ResourceUtilization {
                    cpu_utilization_avg: 68.0,
                    memory_utilization_avg: 72.0,
                    disk_io_utilization_avg: 55.0,
                    network_io_utilization_avg: 35.0,
                    resource_efficiency: 0.75,
                },
                metadata: HashMap::new(),
            },
            impact_scores: super::tools_performance::ImpactScores {
                tools_scores: HashMap::new(),
                category_scores: HashMap::from([
                    ("version_control".to_string(), 8.5),
                    ("build_system".to_string(), 7.2),
                    ("file_system".to_string(), 6.8),
                ]),
                overall_system_impact: 7.5,
                top_performers: vec!["git".to_string(), "ls".to_string()],
                underperformers: vec!["docker".to_string()],
                optimization_priorities: vec![
                    "Optimize Docker build performance".to_string(),
                    "Improve cargo compilation efficiency".to_string(),
                ],
            },
            performance_patterns: super::tools_performance::PerformancePatterns {
                usage_patterns: HashMap::new(),
                performance_patterns: HashMap::new(),
                bottleneck_analysis: super::tools_performance::BottleneckAnalysis {
                    identified_bottlenecks: vec![],
                    resource_constraints: HashMap::new(),
                    performance_limiting_factors: vec![],
                },
                scaling_analysis: super::tools_performance::ScalingAnalysis {
                    current_capacity_utilization: 0.68,
                    scaling_limitations: vec![],
                    performance_at_capacity: 0.72,
                    scaling_recommendations: vec![],
                },
                optimization_potential: HashMap::new(),
            },
            recommendations: vec![
                "Monitor resource utilization trends".to_string(),
                "Implement performance alerting".to_string(),
            ],
            system_health: super::tools_performance::SystemHealth {
                overall_health_score: 82.0,
                health_indicators: HashMap::new(),
                identified_issues: vec![],
                recommendations: vec![],
            },
            optimization_opportunities: vec![],
            metadata: HashMap::new(),
        })
    } else {
        None
    };

    let context_breakdown_report = if config.include_data.contains(&DataType::ContextBreakdown) {
        Some(ContextBreakdownReport {
            analysis_timestamp: chrono::Utc::now(),
            total_segments: 47,
            categories: HashMap::new(),
            segment_details: vec![],
            size_analysis: super::context_breakdown::SizeAnalysis {
                total_size_bytes: 15840,
                average_size_bytes: 337.0,
                min_size_bytes: 45,
                max_size_bytes: 1200,
                size_distribution: HashMap::from([
                    ("tiny".to_string(), 12),
                    ("small".to_string(), 18),
                    ("medium".to_string(), 15),
                    ("large".to_string(), 2),
                ]),
                size_percentiles: HashMap::from([
                    ("p50".to_string(), 280.0),
                    ("p75".to_string(), 450.0),
                    ("p90".to_string(), 680.0),
                    ("p99".to_string(), 950.0),
                ]),
            },
            time_analysis: super::context_breakdown::TemporalAnalysis {
                time_range: "6 hours".to_string(),
                segments_per_minute: 0.13,
                peak_activity_periods: vec!["14:00".to_string(), "15:00".to_string()],
                activity_distribution: HashMap::from([
                    ("9".to_string(), 5),
                    ("14".to_string(), 15),
                    ("15".to_string(), 18),
                    ("16".to_string(), 9),
                ]),
                oldest_segment: chrono::Utc::now() - chrono::Duration::hours(6),
                newest_segment: chrono::Utc::now(),
            },
            recommendation_summary: vec![
                "Consider implementing context pruning".to_string(),
                "Optimize segment categorization".to_string(),
            ],
            metadata: HashMap::new(),
        })
    } else {
        None
    };

    let system_metrics = SystemMetricsExport {
        cpu_usage: 45.2,
        memory_usage: 68.7,
        disk_usage: 34.1,
        network_io: 12.5,
        process_count: 125,
        uptime_hours: 48.5,
        system_load: 2.1,
        environment: "development".to_string(),
        hardware_info: HashMap::from([
            ("cpu_model".to_string(), "Intel i7-9700K".to_string()),
            ("memory_total".to_string(), "16GB".to_string()),
            ("storage_type".to_string(), "SSD".to_string()),
        ]),
    };

    let metadata = ExportMetadata {
        export_timestamp: chrono::Utc::now(),
        export_format: config.format.clone(),
        export_scope: config.scope.clone(),
        data_sources: vec![
            "context_storage".to_string(),
            "performance_monitor".to_string(),
            "token_tracker".to_string(),
            "system_metrics".to_string(),
        ],
        total_reports_included: config.include_data.len(),
        file_size_estimate: 15420, // Mock estimate
        version: "1.0.0".to_string(),
        generator: "OmniMemory CLI".to_string(),
        checksum: None, // Would be calculated
    };

    Ok(ProfileExportData {
        metadata,
        profile_report,
        token_usage_report,
        tools_performance_report,
        context_breakdown_report,
        system_metrics,
        recommendations: vec![
            "Review performance optimization opportunities".to_string(),
            "Monitor resource utilization trends".to_string(),
        ],
        export_config: config.clone(),
    })
}

/// Format export data according to configuration
fn format_export_data(
    data: &ProfileExportData,
    config: &ExportConfig,
) -> Result<serde_json::Value> {
    let mut export_data = serde_json::Map::new();

    // Add metadata
    export_data.insert("metadata".to_string(), serde_json::to_value(&data.metadata)?);
    
    // Add system metrics if included
    if config.include_data.contains(&DataType::SystemMetrics) {
        export_data.insert("system_metrics".to_string(), serde_json::to_value(&data.system_metrics)?);
    }

    // Add reports based on configuration
    if let Some(ref profile_report) = data.profile_report {
        export_data.insert("profile_report".to_string(), serde_json::to_value(profile_report)?);
    }

    if let Some(ref token_report) = data.token_usage_report {
        export_data.insert("token_usage_report".to_string(), serde_json::to_value(token_report)?);
    }

    if let Some(ref tools_report) = data.tools_performance_report {
        export_data.insert("tools_performance_report".to_string(), serde_json::to_value(tools_report)?);
    }

    if let Some(ref context_report) = data.context_breakdown_report {
        export_data.insert("context_breakdown_report".to_string(), serde_json::to_value(context_report)?);
    }

    // Add recommendations
    export_data.insert("recommendations".to_string(), serde_json::to_value(&data.recommendations)?);

    // Add export configuration
    export_data.insert("export_config".to_string(), serde_json::to_value(&config)?);

    // Add metadata if requested
    if config.include_metadata {
        let metadata = serde_json::json!({
            "generator_info": {
                "tool": "OmniMemory CLI",
                "version": env!("CARGO_PKG_VERSION"),
                "export_time": chrono::Utc::now().to_rfc3339(),
            },
            "data_summary": {
                "total_reports": data.metadata.total_reports_included,
                "export_scope": data.metadata.export_scope,
                "format": data.metadata.export_format,
            }
        });
        export_data.insert("export_metadata".to_string(), metadata);
    }

    Ok(serde_json::Value::Object(export_data))
}

/// Generate export files based on format and configuration
fn generate_export_files(
    data: &serde_json::Value,
    config: &ExportConfig,
) -> Result<ExportResult> {
    let start_time = std::time::Instant::now();
    let mut files_created = Vec::new();
    let mut total_size = 0;
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();

    // Generate output path
    let mut output_path = PathBuf::from(&config.output_file);
    
    match config.format {
        ExportFormat::Json => {
            if output_path.extension().is_none() {
                output_path.set_extension("json");
            }
            
            let json_content = serde_json::to_string_pretty(data)?;
            std::fs::write(&output_path, json_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Json,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
        
        ExportFormat::Csv => {
            // CSV export would require data transformation
            let csv_content = generate_csv_content(data, config)?;
            
            if output_path.extension().is_none() {
                output_path.set_extension("csv");
            }
            
            std::fs::write(&output_path, csv_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Csv,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
        
        ExportFormat::Html => {
            let html_content = generate_html_content(data, config)?;
            
            if output_path.extension().is_none() {
                output_path.set_extension("html");
            }
            
            std::fs::write(&output_path, html_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Html,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
        
        ExportFormat::Xml => {
            let xml_content = generate_xml_content(data, config)?;
            
            if output_path.extension().is_none() {
                output_path.set_extension("xml");
            }
            
            std::fs::write(&output_path, xml_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Xml,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
        
        ExportFormat::Yaml => {
            let yaml_content = serde_yaml::to_string(data)?;
            
            if output_path.extension().is_none() {
                output_path.set_extension("yaml");
            }
            
            std::fs::write(&output_path, yaml_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Yaml,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
        
        ExportFormat::Pdf => {
            // PDF generation would require external library
            warnings.push("PDF generation requires external dependency - using HTML fallback".to_string());
            
            let html_content = generate_html_content(data, config)?;
            output_path.set_extension("html");
            std::fs::write(&output_path, html_content)?;
            
            let file_size = std::fs::metadata(&output_path)?.len() as usize;
            total_size += file_size;
            
            files_created.push(ExportFile {
                file_path: output_path.to_string_lossy().to_string(),
                file_type: ExportFormat::Html,
                size_bytes: file_size,
                data_types_included: config.include_data.clone(),
                compression_ratio: None,
            });
        }
    }

    if files_created.is_empty() {
        warnings.push("No files were created - check configuration".to_string());
    } else {
        recommendations.push("Review exported data for completeness".to_string());
    }

    let export_duration = start_time.elapsed().as_millis() as u64;

    let success = !files_created.is_empty();

    let summary = ExportSummary {
        total_exports: 1,
        successful_exports: if success { 1 } else { 0 },
        failed_exports: if success { 0 } else { 1 },
        total_records_exported: config.include_data.len(),
        warnings,
        recommendations,
    };

    Ok(ExportResult {
        export_timestamp: chrono::Utc::now(),
        files_created,
        total_size_bytes: total_size,
        export_duration_ms: export_duration,
        success,
        summary,
    })
}

/// Generate CSV content from JSON data
fn generate_csv_content(data: &serde_json::Value, config: &ExportConfig) -> Result<String> {
    let mut csv_rows = Vec::new();
    
    // CSV header
    csv_rows.push("Type,Name,Value,Unit,Timestamp".to_string());
    
    // Extract data and convert to CSV format
    if let Some(profile_report) = data.get("profile_report") {
        if let Some(health_score) = profile_report.get("health_score") {
            csv_rows.push(format!(
                "Profile,Health Score,{},,{}",
                health_score.as_f64().unwrap_or(0.0),
                chrono::Utc::now().to_rfc3339()
            ));
        }
        
        if let Some(context_summary) = profile_report.get("context_summary") {
            if let Some(total_segments) = context_summary.get("total_segments") {
                csv_rows.push(format!(
                    "Context,Total Segments,{},,{}",
                    total_segments.as_u64().unwrap_or(0),
                    chrono::Utc::now().to_rfc3339()
                ));
            }
        }
    }

    if let Some(token_report) = data.get("token_usage_report") {
        if let Some(usage_summary) = token_report.get("usage_summary") {
            if let Some(total_tokens) = usage_summary.get("total_tokens") {
                csv_rows.push(format!(
                    "Tokens,Total Usage,{},tokens,{}",
                    total_tokens.as_u64().unwrap_or(0),
                    chrono::Utc::now().to_rfc3339()
                ));
            }
            
            if let Some(efficiency_score) = usage_summary.get("efficiency_score") {
                csv_rows.push(format!(
                    "Tokens,Efficiency Score,{},%,{}",
                    efficiency_score.as_f64().unwrap_or(0.0) * 100.0,
                    chrono::Utc::now().to_rfc3339()
                ));
            }
        }
    }

    Ok(csv_rows.join("\n"))
}

/// Generate HTML content from JSON data
fn generate_html_content(data: &serde_json::Value, config: &ExportConfig) -> Result<String> {
    let style_options = &config.style_options;
    
    let html_template = format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniMemory Profile Export</title>
    <style>
        body {{
            font-family: {font_family};
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid {primary_color};
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: {primary_color};
            margin: 0;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid {secondary_color};
            background-color: #f9f9f9;
        }}
        .section h2 {{
            color: {primary_color};
            margin-top: 0;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .value {{
            font-weight: bold;
            color: {accent_color};
        }}
        .chart {{
            height: 200px;
            background-color: #f0f0f0;
            border-radius: 4px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }}
        {custom_css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OmniMemory Profile Export</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        {content}
        
        <div class="footer">
            <p>Generated by OmniMemory CLI v{version}</p>
            <p>Export Format: {format} | Scope: {scope}</p>
        </div>
    </div>
</body>
</html>
    "#,
        font_family = style_options.font_family,
        primary_color = match style_options.color_scheme.as_str() {
            "blue" => "#2563eb",
            "purple" => "#7c3aed",
            "green" => "#059669",
            "red" => "#dc2626",
            _ => "#2563eb",
        },
        secondary_color = match style_options.color_scheme.as_str() {
            "blue" => "#3b82f6",
            "purple" => "#8b5cf6",
            "green" => "#10b981",
            "red" => "#ef4444",
            _ => "#3b82f6",
        },
        accent_color = match style_options.color_scheme.as_str() {
            "blue" => "#1d4ed8",
            "purple" => "#6d28d9",
            "green" => "#047857",
            "red" => "#b91c1c",
            _ => "#1d4ed8",
        },
        custom_css = style_options.custom_css.as_ref().map(|s| s.as_str()).unwrap_or(""),
        timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
        version = env!("CARGO_PKG_VERSION"),
        format = format!("{:?}", config.format),
        scope = format!("{:?}", config.scope),
        content = generate_html_content_sections(data, config)?
    );

    Ok(html_template)
}

/// Generate HTML content sections
fn generate_html_content_sections(data: &serde_json::Value, config: &ExportConfig) -> Result<String> {
    let mut sections = Vec::new();

    // Profile Report Section
    if let Some(profile_report) = data.get("profile_report") {
        let mut content = String::new();
        
        if let Some(health_score) = profile_report.get("health_score") {
            content.push_str(&format!(
                "<div class='metric'><span>Health Score</span><span class='value'>{:.1}/100</span></div>",
                health_score.as_f64().unwrap_or(0.0)
            ));
        }
        
        if let Some(context_summary) = profile_report.get("context_summary") {
            if let Some(total_segments) = context_summary.get("total_segments") {
                content.push_str(&format!(
                    "<div class='metric'><span>Context Segments</span><span class='value'>{}</span></div>",
                    total_segments.as_u64().unwrap_or(0)
                ));
            }
            
            if let Some(total_tokens) = context_summary.get("total_tokens") {
                content.push_str(&format!(
                    "<div class='metric'><span>Total Tokens</span><span class='value'>{}</span></div>",
                    total_tokens.as_u64().unwrap_or(0)
                ));
            }
        }
        
        sections.push(format!(
            "<div class='section'><h2>Profile Summary</h2>{}</div>",
            content
        ));
    }

    // Token Usage Section
    if let Some(token_report) = data.get("token_usage_report") {
        let mut content = String::new();
        
        if let Some(usage_summary) = token_report.get("usage_summary") {
            if let Some(total_tokens) = usage_summary.get("total_tokens") {
                content.push_str(&format!(
                    "<div class='metric'><span>Total Token Usage</span><span class='value'>{}</span></div>",
                    total_tokens.as_u64().unwrap_or(0)
                ));
            }
            
            if let Some(efficiency_score) = usage_summary.get("efficiency_score") {
                content.push_str(&format!(
                    "<div class='metric'><span>Efficiency Score</span><span class='value'>{:.1}%</span></div>",
                    efficiency_score.as_f64().unwrap_or(0.0) * 100.0
                ));
            }
        }
        
        if config.include_charts {
            content.push_str("<div class='chart'>Token Usage Chart Placeholder</div>");
        }
        
        sections.push(format!(
            "<div class='section'><h2>Token Usage Analysis</h2>{}</div>",
            content
        ));
    }

    Ok(sections.join("\n"))
}

/// Generate XML content from JSON data
fn generate_xml_content(data: &serde_json::Value, config: &ExportConfig) -> Result<String> {
    let mut xml_content = String::new();
    xml_content.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml_content.push_str("<profile_export>\n");
    
    // Add metadata
    xml_content.push_str("  <metadata>\n");
    xml_content.push_str(&format!("    <timestamp>{}</timestamp>\n", chrono::Utc::now().to_rfc3339()));
    xml_content.push_str(&format!("    <format>{}</format>\n", format!("{:?}", config.format)));
    xml_content.push_str("  </metadata>\n");
    
    // Add data sections
    if let Some(profile_report) = data.get("profile_report") {
        xml_content.push_str("  <profile_report>\n");
        if let Some(health_score) = profile_report.get("health_score") {
            xml_content.push_str(&format!("    <health_score>{:.1}</health_score>\n", 
                health_score.as_f64().unwrap_or(0.0)));
        }
        xml_content.push_str("  </profile_report>\n");
    }
    
    xml_content.push_str("</profile_export>");
    
    Ok(xml_content)
}

/// Print human-readable export information
fn print_human_export_info(result: &ExportResult, config: &ExportConfig) {
    println!("\n{}", "📤 Profile Export Complete".bold().blue());
    println!("Export Duration: {}ms", result.export_duration_ms);
    println!("Success: {}\n", if result.success { "✓" } else { "✗" });

    if !result.files_created.is_empty() {
        println!("{}", "Files Created".bold().magenta());
        for file in &result.files_created {
            println!("  {} ({}) - {} bytes", 
                file.file_path.cyan(),
                format!("{:?}", file.file_type),
                file.size_bytes
            );
        }
        println!();
    }

    println!("Export Summary:");
    println!("  Total Exports: {}", result.summary.total_exports);
    println!("  Successful: {} | Failed: {}", 
        result.summary.successful_exports, 
        result.summary.failed_exports
    );
    println!("  Total Size: {} bytes\n", result.total_size_bytes);

    if !result.summary.warnings.is_empty() {
        println!("{}", "Warnings".bold().yellow());
        for warning in &result.summary.warnings {
            println!("  ⚠ {}", warning);
        }
        println!();
    }

    if !result.summary.recommendations.is_empty() {
        println!("{}", "Recommendations".bold().green());
        for recommendation in &result.summary.recommendations {
            println!("  💡 {}", recommendation);
        }
        println!();
    }

    println!("Exported at: {}", result.export_timestamp.format("%Y-%m-%d %H:%M:%S"));
}