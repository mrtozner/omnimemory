//! Tools Performance Analysis Module
//!
//! This module provides comprehensive tools performance monitoring including
//! impact scoring, usage analytics, response time analysis, and optimization insights.

use crate::commands::{CommandContext, Result};
use super::{format_percentage};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Handle tools performance command
pub fn handle_tools_performance(
    ctx: &CommandContext,
    performance: Option<ToolsPerformance>,
) -> Result<()> {
    let config = performance.unwrap_or_default();
    
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Analyzing tools performance..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Collecting tool usage metrics...");
        pb.set_position(20);
    }

    // Gather tools performance data
    let tools_data = collect_tools_performance(&config)?;

    if let Some(ref pb) = pb {
        pb.set_message("Calculating impact scores...");
        pb.set_position(40);
    }

    // Calculate impact scores for each tool
    let impact_scores = calculate_impact_scores(&tools_data);

    if let Some(ref pb) = pb {
        pb.set_message("Analyzing performance patterns...");
        pb.set_position(60);
    }

    // Analyze performance patterns
    let performance_patterns = analyze_performance_patterns(&tools_data);

    if let Some(ref pb) = pb {
        pb.set_message("Generating recommendations...");
        pb.set_position(80);
    }

    // Generate optimization recommendations
    let recommendations = generate_performance_recommendations(&tools_data, &impact_scores);

    // Analyze system health before moving tools_data
    let system_health = analyze_system_health(&tools_data);
    let optimization_opportunities = identify_optimization_opportunities(&tools_data);

    // Create comprehensive performance report
    let report = ToolsPerformanceReport {
        report_timestamp: chrono::Utc::now(),
        performance_summary: tools_data,
        impact_scores,
        performance_patterns,
        recommendations,
        system_health,
        optimization_opportunities,
        metadata: HashMap::from([
            ("analysis_depth".to_string(), serde_json::Value::String(config.analysis_depth.clone())),
            ("time_window".to_string(), serde_json::Value::String(config.time_window.clone())),
            ("include_metrics".to_string(), serde_json::to_value(&config.include_metrics)?),
        ]),
    };

    if let Some(ref pb) = pb {
        pb.set_message("Performance analysis complete");
        pb.finish_with_message("✓ Analysis ready");
    }

    // Handle output
    match ctx.output_format {
        super::super::OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        super::super::OutputFormat::Human => {
            print_human_performance_report(&report, &config);
        }
        _ => {
            println!("{}", serde_json::to_string(&report)?);
        }
    }

    Ok(())
}

/// Tools performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsPerformance {
    /// Analysis depth (basic, detailed, comprehensive)
    pub analysis_depth: String,
    
    /// Time window for analysis (1h, 6h, 24h, 7d, 30d)
    pub time_window: String,
    
    /// Specific tools to analyze
    pub target_tools: Vec<String>,
    
    /// Performance metrics to include
    pub include_metrics: Vec<MetricType>,
    
    /// Sort tools by metric
    pub sort_by: Option<SortByMetric>,
    
    /// Minimum usage threshold for inclusion
    pub min_usage_threshold: Option<usize>,
    
    /// Show detailed breakdowns
    pub detailed_breakdown: bool,
    
    /// Generate optimization suggestions
    pub generate_optimizations: bool,
    
    /// Focus on specific metric categories
    pub focus_categories: Option<Vec<String>>,
}

impl Default for ToolsPerformance {
    fn default() -> Self {
        Self {
            analysis_depth: "detailed".to_string(),
            time_window: "24h".to_string(),
            target_tools: vec![],
            include_metrics: vec![
                MetricType::UsageCount,
                MetricType::ResponseTime,
                MetricType::SuccessRate,
                MetricType::ErrorRate,
            ],
            sort_by: Some(SortByMetric::ImpactScore),
            min_usage_threshold: Some(5),
            detailed_breakdown: true,
            generate_optimizations: true,
            focus_categories: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    UsageCount,
    ResponseTime,
    SuccessRate,
    ErrorRate,
    ImpactScore,
    Efficiency,
    Reliability,
    UserSatisfaction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortByMetric {
    UsageCount,
    ResponseTime,
    SuccessRate,
    ImpactScore,
    Alphabetical,
}

/// Tools performance data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsPerformanceData {
    pub tools: HashMap<String, ToolPerformance>,
    pub overall_stats: OverallPerformanceStats,
    pub usage_timeline: Vec<UsageEvent>,
    pub performance_trends: HashMap<String, Vec<PerformancePoint>>,
    pub error_analysis: ErrorAnalysis,
    pub resource_utilization: ResourceUtilization,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPerformance {
    pub tool_name: String,
    pub category: String,
    pub usage_count: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub average_response_time_ms: f32,
    pub median_response_time_ms: f32,
    pub p95_response_time_ms: f32,
    pub success_rate: f32,
    pub error_rate: f32,
    pub impact_score: f32,
    pub efficiency_rating: f32,
    pub reliability_rating: f32,
    pub user_satisfaction: f32,
    pub recent_performance: Vec<PerformancePoint>,
    pub resource_usage: ResourceUsage,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tool_name: String,
    pub success: bool,
    pub response_time_ms: f32,
    pub user_id: Option<String>,
    pub operation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub response_time_ms: f32,
    pub success: bool,
    pub resource_usage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: f32,
    pub disk_io_mb: f32,
    pub network_io_mb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallPerformanceStats {
    pub total_tools: usize,
    pub active_tools: usize,
    pub total_operations: usize,
    pub total_response_time_ms: f32,
    pub average_response_time_ms: f32,
    pub overall_success_rate: f32,
    pub system_health_score: f32,
    pub peak_usage_time: chrono::DateTime<chrono::Utc>,
    pub busiest_tool: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub total_errors: usize,
    pub error_rate: f32,
    pub common_errors: HashMap<String, usize>,
    pub error_by_tool: HashMap<String, usize>,
    pub error_trends: Vec<ErrorTrend>,
    pub reliability_concerns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrend {
    pub date: chrono::DateTime<chrono::Utc>,
    pub error_count: usize,
    pub error_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_utilization_avg: f32,
    pub memory_utilization_avg: f32,
    pub disk_io_utilization_avg: f32,
    pub network_io_utilization_avg: f32,
    pub resource_efficiency: f32,
}

/// Impact scores calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactScores {
    pub tools_scores: HashMap<String, ImpactScore>,
    pub category_scores: HashMap<String, f32>,
    pub overall_system_impact: f32,
    pub top_performers: Vec<String>,
    pub underperformers: Vec<String>,
    pub optimization_priorities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactScore {
    pub tool_name: String,
    pub impact_score: f32,
    pub usage_weight: f32,
    pub performance_weight: f32,
    pub reliability_weight: f32,
    pub efficiency_weight: f32,
    pub score_breakdown: HashMap<String, f32>,
    pub recommendations: Vec<String>,
}

/// Performance pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePatterns {
    pub usage_patterns: HashMap<String, UsagePattern>,
    pub performance_patterns: HashMap<String, PerformancePattern>,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub scaling_analysis: ScalingAnalysis,
    pub optimization_potential: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePattern {
    pub pattern_type: String,
    pub peak_hours: Vec<u8>,
    pub average_usage_per_hour: f32,
    pub usage_distribution: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    pub pattern_type: String,
    pub correlation_with_load: f32,
    pub performance_degradation_threshold: f32,
    pub optimization_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub identified_bottlenecks: Vec<Bottleneck>,
    pub resource_constraints: HashMap<String, f32>,
    pub performance_limiting_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub severity: String,
    pub impact_on_performance: f32,
    pub suggested_resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub current_capacity_utilization: f32,
    pub scaling_limitations: Vec<String>,
    pub performance_at_capacity: f32,
    pub scaling_recommendations: Vec<String>,
}

/// Collect tools performance data (mock implementation)
fn collect_tools_performance(config: &ToolsPerformance) -> Result<ToolsPerformanceData> {
    let tools = HashMap::from([
        (
            "git".to_string(),
            ToolPerformance {
                tool_name: "git".to_string(),
                category: "version_control".to_string(),
                usage_count: 125,
                success_count: 118,
                error_count: 7,
                average_response_time_ms: 245.5,
                median_response_time_ms: 180.0,
                p95_response_time_ms: 520.0,
                success_rate: 0.944,
                error_rate: 0.056,
                impact_score: 8.5,
                efficiency_rating: 0.89,
                reliability_rating: 0.94,
                user_satisfaction: 4.2,
                recent_performance: vec![
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(15),
                        response_time_ms: 210.0,
                        success: true,
                        resource_usage: 0.15,
                    },
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(30),
                        response_time_ms: 185.0,
                        success: true,
                        resource_usage: 0.12,
                    },
                ],
                resource_usage: ResourceUsage {
                    cpu_usage_percent: 15.0,
                    memory_usage_mb: 45.0,
                    disk_io_mb: 120.0,
                    network_io_mb: 15.0,
                },
                metadata: HashMap::from([
                    ("last_error".to_string(), serde_json::Value::String("merge conflict resolution timeout".to_string())),
                    ("error_resolved".to_string(), serde_json::Value::Bool(true)),
                ]),
            }
        ),
        (
            "cargo".to_string(),
            ToolPerformance {
                tool_name: "cargo".to_string(),
                category: "build_system".to_string(),
                usage_count: 89,
                success_count: 85,
                error_count: 4,
                average_response_time_ms: 1250.0,
                median_response_time_ms: 1100.0,
                p95_response_time_ms: 2800.0,
                success_rate: 0.955,
                error_rate: 0.045,
                impact_score: 9.2,
                efficiency_rating: 0.75,
                reliability_rating: 0.96,
                user_satisfaction: 4.5,
                recent_performance: vec![
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(20),
                        response_time_ms: 1450.0,
                        success: true,
                        resource_usage: 0.45,
                    },
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(45),
                        response_time_ms: 1100.0,
                        success: true,
                        resource_usage: 0.38,
                    },
                ],
                resource_usage: ResourceUsage {
                    cpu_usage_percent: 65.0,
                    memory_usage_mb: 320.0,
                    disk_io_mb: 850.0,
                    network_io_mb: 45.0,
                },
                metadata: HashMap::from([
                    ("build_time_avg".to_string(), serde_json::Value::Number(1250.into())),
                    ("optimization_enabled".to_string(), serde_json::Value::Bool(true)),
                ]),
            }
        ),
        (
            "docker".to_string(),
            ToolPerformance {
                tool_name: "docker".to_string(),
                category: "containerization".to_string(),
                usage_count: 45,
                success_count: 42,
                error_count: 3,
                average_response_time_ms: 3200.0,
                median_response_time_ms: 2800.0,
                p95_response_time_ms: 6500.0,
                success_rate: 0.933,
                error_rate: 0.067,
                impact_score: 7.8,
                efficiency_rating: 0.68,
                reliability_rating: 0.93,
                user_satisfaction: 3.9,
                recent_performance: vec![
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(10),
                        response_time_ms: 3500.0,
                        success: false,
                        resource_usage: 0.85,
                    },
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(25),
                        response_time_ms: 2900.0,
                        success: true,
                        resource_usage: 0.72,
                    },
                ],
                resource_usage: ResourceUsage {
                    cpu_usage_percent: 85.0,
                    memory_usage_mb: 1200.0,
                    disk_io_mb: 2100.0,
                    network_io_mb: 125.0,
                },
                metadata: HashMap::from([
                    ("container_count".to_string(), serde_json::Value::Number(8.into())),
                    ("images_size_gb".to_string(), serde_json::Value::Number(15.into())),
                ]),
            }
        ),
        (
            "ls".to_string(),
            ToolPerformance {
                tool_name: "ls".to_string(),
                category: "file_system".to_string(),
                usage_count: 340,
                success_count: 339,
                error_count: 1,
                average_response_time_ms: 25.0,
                median_response_time_ms: 20.0,
                p95_response_time_ms: 85.0,
                success_rate: 0.997,
                error_rate: 0.003,
                impact_score: 6.5,
                efficiency_rating: 0.98,
                reliability_rating: 0.997,
                user_satisfaction: 4.8,
                recent_performance: vec![
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(5),
                        response_time_ms: 18.0,
                        success: true,
                        resource_usage: 0.02,
                    },
                    PerformancePoint {
                        timestamp: chrono::Utc::now() - chrono::Duration::minutes(8),
                        response_time_ms: 22.0,
                        success: true,
                        resource_usage: 0.03,
                    },
                ],
                resource_usage: ResourceUsage {
                    cpu_usage_percent: 3.0,
                    memory_usage_mb: 8.0,
                    disk_io_mb: 15.0,
                    network_io_mb: 0.0,
                },
                metadata: HashMap::from([
                    ("average_directory_size".to_string(), serde_json::Value::Number(150.into())),
                    ("files_listed_count".to_string(), serde_json::Value::Number(12500.into())),
                ]),
            }
        ),
    ]);

    let overall_stats = OverallPerformanceStats {
        total_tools: 4,
        active_tools: 4,
        total_operations: 599,
        total_response_time_ms: 145650.0,
        average_response_time_ms: 243.1,
        overall_success_rate: 0.975,
        system_health_score: 8.2,
        peak_usage_time: chrono::Utc::now() - chrono::Duration::hours(2),
        busiest_tool: "ls".to_string(),
    };

    Ok(ToolsPerformanceData {
        tools,
        overall_stats,
        usage_timeline: vec![], // Would be populated with real data
        performance_trends: HashMap::new(), // Would be populated with real data
        error_analysis: ErrorAnalysis {
            total_errors: 15,
            error_rate: 0.025,
            common_errors: HashMap::from([
                ("timeout".to_string(), 6),
                ("permission_denied".to_string(), 4),
                ("resource_exhausted".to_string(), 3),
                ("network_error".to_string(), 2),
            ]),
            error_by_tool: HashMap::from([
                ("git".to_string(), 7),
                ("cargo".to_string(), 4),
                ("docker".to_string(), 3),
                ("ls".to_string(), 1),
            ]),
            error_trends: vec![
                ErrorTrend {
                    date: chrono::Utc::now() - chrono::Duration::days(1),
                    error_count: 12,
                    error_rate: 0.028,
                },
                ErrorTrend {
                    date: chrono::Utc::now() - chrono::Duration::days(2),
                    error_count: 15,
                    error_rate: 0.032,
                },
            ],
            reliability_concerns: vec![
                "Docker response times degrading under load".to_string(),
                "Cargo builds showing memory pressure".to_string(),
            ],
        },
        resource_utilization: ResourceUtilization {
            cpu_utilization_avg: 42.0,
            memory_utilization_avg: 58.0,
            disk_io_utilization_avg: 67.0,
            network_io_utilization_avg: 35.0,
            resource_efficiency: 0.78,
        },
        metadata: HashMap::from([
            ("collection_period".to_string(), serde_json::Value::String(config.time_window.clone())),
            ("measurement_interval".to_string(), serde_json::Value::String("1m".to_string())),
        ]),
    })
}

/// Calculate impact scores for tools
fn calculate_impact_scores(tools_data: &ToolsPerformanceData) -> ImpactScores {
    let mut tools_scores = HashMap::new();
    let mut category_scores = HashMap::new();
    let mut total_impact = 0.0;
    let mut total_categories = 0;

    for (tool_name, tool_perf) in &tools_data.tools {
        // Calculate weighted impact score
        let usage_weight = (tool_perf.usage_count as f32 / tools_data.overall_stats.total_operations as f32) * 0.3;
        let performance_weight = (1.0 - (tool_perf.average_response_time_ms / 5000.0)) * 0.25; // Normalize to 5s max
        let reliability_weight = tool_perf.success_rate * 0.25;
        let efficiency_weight = tool_perf.efficiency_rating * 0.2;

        let impact_score = (usage_weight + performance_weight + reliability_weight + efficiency_weight) * 10.0;

        let score_breakdown = HashMap::from([
            ("usage_impact".to_string(), usage_weight * 10.0),
            ("performance_impact".to_string(), performance_weight * 10.0),
            ("reliability_impact".to_string(), reliability_weight * 10.0),
            ("efficiency_impact".to_string(), efficiency_weight * 10.0),
        ]);

        let recommendations = generate_tool_recommendations(&tool_perf);

        tools_scores.insert(tool_name.clone(), ImpactScore {
            tool_name: tool_name.clone(),
            impact_score,
            usage_weight,
            performance_weight,
            reliability_weight,
            efficiency_weight,
            score_breakdown,
            recommendations,
        });

        // Update category scores
        let category = &tool_perf.category;
        let current_score = category_scores.entry(category.clone()).or_insert(0.0);
        *current_score += impact_score;

        total_impact += impact_score;
        if !category_scores.contains_key(category) {
            total_categories += 1;
        }
    }

    // Normalize category scores
    for score in category_scores.values_mut() {
        *score = (*score / (tools_data.overall_stats.total_tools as f32 / total_categories as f32)) / 10.0;
    }

    let overall_system_impact = total_impact / tools_data.overall_stats.total_tools as f32;

    // Sort tools by impact score
    let mut tool_scores_vec: Vec<_> = tools_scores.values().collect();
    tool_scores_vec.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap());

    let top_performers: Vec<String> = tool_scores_vec.iter()
        .take(2)
        .map(|s| s.tool_name.clone())
        .collect();

    let underperformers: Vec<String> = tool_scores_vec.iter()
        .rev()
        .take(2)
        .map(|s| s.tool_name.clone())
        .collect();

    let optimization_priorities = generate_optimization_priorities(&tools_scores);

    ImpactScores {
        tools_scores,
        category_scores,
        overall_system_impact,
        top_performers,
        underperformers,
        optimization_priorities,
    }
}

/// Analyze performance patterns
fn analyze_performance_patterns(tools_data: &ToolsPerformanceData) -> PerformancePatterns {
    let usage_patterns = HashMap::from([
        ("peak_usage".to_string(), UsagePattern {
            pattern_type: "diurnal".to_string(),
            peak_hours: vec![9, 10, 14, 15, 16],
            average_usage_per_hour: 24.9,
            usage_distribution: HashMap::from([
                ("morning".to_string(), 0.35),
                ("afternoon".to_string(), 0.42),
                ("evening".to_string(), 0.23),
            ]),
        }),
    ]);

    let performance_patterns = HashMap::from([
        ("response_time_degradation".to_string(), PerformancePattern {
            pattern_type: "load_dependent".to_string(),
            correlation_with_load: 0.78,
            performance_degradation_threshold: 0.75,
            optimization_impact: 0.65,
        }),
    ]);

    let bottleneck_analysis = BottleneckAnalysis {
        identified_bottlenecks: vec![
            Bottleneck {
                component: "docker_container_build".to_string(),
                severity: "medium".to_string(),
                impact_on_performance: 0.45,
                suggested_resolution: "Implement parallel builds and caching".to_string(),
            },
            Bottleneck {
                component: "cargo_compilation_memory".to_string(),
                severity: "low".to_string(),
                impact_on_performance: 0.25,
                suggested_resolution: "Optimize memory allocation patterns".to_string(),
            },
        ],
        resource_constraints: HashMap::from([
            ("memory".to_string(), 0.75),
            ("cpu".to_string(), 0.65),
            ("disk_io".to_string(), 0.55),
            ("network".to_string(), 0.35),
        ]),
        performance_limiting_factors: vec![
            "Large container image builds".to_string(),
            "Memory-intensive compilation".to_string(),
        ],
    };

    let scaling_analysis = ScalingAnalysis {
        current_capacity_utilization: 0.58,
        scaling_limitations: vec![
            "Memory constraint on cargo builds".to_string(),
            "Docker daemon responsiveness".to_string(),
        ],
        performance_at_capacity: 0.72,
        scaling_recommendations: vec![
            "Increase available memory for build operations".to_string(),
            "Consider distributed container builds".to_string(),
        ],
    };

    let optimization_potential = HashMap::from([
        ("docker".to_string(), 0.35),
        ("cargo".to_string(), 0.25),
        ("git".to_string(), 0.15),
        ("ls".to_string(), 0.05),
    ]);

    PerformancePatterns {
        usage_patterns,
        performance_patterns,
        bottleneck_analysis,
        scaling_analysis,
        optimization_potential,
    }
}

/// Generate performance recommendations for tools
fn generate_tool_recommendations(tool_perf: &ToolPerformance) -> Vec<String> {
    let mut recommendations = Vec::new();

    if tool_perf.average_response_time_ms > 2000.0 {
        recommendations.push("Consider performance optimization - high response time".to_string());
    }

    if tool_perf.success_rate < 0.95 {
        recommendations.push("Improve reliability - success rate below threshold".to_string());
    }

    if tool_perf.error_rate > 0.05 {
        recommendations.push("Reduce error frequency".to_string());
    }

    if tool_perf.efficiency_rating < 0.8 {
        recommendations.push("Optimize resource usage efficiency".to_string());
    }

    if tool_perf.usage_count > 100 {
        recommendations.push("High usage volume - consider caching strategies".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Performance metrics are within acceptable ranges".to_string());
    }

    recommendations
}

/// Generate optimization priorities
fn generate_optimization_priorities(impact_scores: &HashMap<String, ImpactScore>) -> Vec<String> {
    let mut priorities = Vec::new();

    // Find tools with lowest efficiency but high usage
    for score in impact_scores.values() {
        if score.usage_weight > 0.2 && score.efficiency_weight < 0.7 {
            priorities.push(format!("High-impact optimization: {}", score.tool_name));
        }
    }

    // Find tools with high impact scores
    let mut sorted_scores: Vec<_> = impact_scores.values().collect();
    sorted_scores.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap());

    for score in sorted_scores.iter().take(3) {
        if !priorities.contains(&format!("High-impact optimization: {}", score.tool_name)) {
            priorities.push(format!("Impact optimization: {}", score.tool_name));
        }
    }

    if priorities.is_empty() {
        priorities.push("Current performance levels are acceptable".to_string());
    }

    priorities
}

/// Analyze system health
fn analyze_system_health(tools_data: &ToolsPerformanceData) -> SystemHealth {
    let health_indicators = HashMap::from([
        ("overall_success_rate".to_string(), tools_data.overall_stats.overall_success_rate),
        ("response_time_health".to_string(), (1.0 - (tools_data.overall_stats.average_response_time_ms / 3000.0)).max(0.0)),
        ("resource_efficiency".to_string(), tools_data.resource_utilization.resource_efficiency),
        ("error_rate_health".to_string(), 1.0 - tools_data.error_analysis.error_rate),
    ]);

    let health_score = health_indicators.values()
        .sum::<f32>() / health_indicators.len() as f32 * 100.0;

    let health_issues = vec![
        "Docker performance degradation under load".to_string(),
        "Memory usage approaching limits".to_string(),
    ];

    SystemHealth {
        overall_health_score: health_score,
        health_indicators,
        identified_issues: health_issues,
        recommendations: vec![
            "Monitor resource utilization closely".to_string(),
            "Implement performance optimization strategies".to_string(),
        ],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_health_score: f32,
    pub health_indicators: HashMap<String, f32>,
    pub identified_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Identify optimization opportunities
fn identify_optimization_opportunities(tools_data: &ToolsPerformanceData) -> Vec<OptimizationOpportunity> {
    let mut opportunities = Vec::new();

    for (tool_name, tool_perf) in &tools_data.tools {
        let potential = match tool_perf.category.as_str() {
            "build_system" => 0.35,
            "containerization" => 0.40,
            "version_control" => 0.15,
            "file_system" => 0.05,
            _ => 0.20,
        };

        if potential > 0.2 {
            opportunities.push(OptimizationOpportunity {
                tool_name: tool_name.clone(),
                optimization_type: match tool_perf.category.as_str() {
                    "build_system" => "Build Optimization".to_string(),
                    "containerization" => "Container Performance".to_string(),
                    "version_control" => "Operation Efficiency".to_string(),
                    "file_system" => "File I/O Optimization".to_string(),
                    _ => "Performance Tuning".to_string(),
                },
                impact_potential: potential,
                effort_estimate: match potential {
                    x if x > 0.3 => "High",
                    x if x > 0.2 => "Medium",
                    _ => "Low",
                }.to_string(),
                description: generate_optimization_description(tool_perf),
            });
        }
    }

    opportunities.sort_by(|a, b| b.impact_potential.partial_cmp(&a.impact_potential).unwrap());
    opportunities
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub tool_name: String,
    pub optimization_type: String,
    pub impact_potential: f32,
    pub effort_estimate: String,
    pub description: String,
}

fn generate_optimization_description(tool_perf: &ToolPerformance) -> String {
    let mut description = String::new();
    
    if tool_perf.average_response_time_ms > 2000.0 {
        description.push_str("Response time optimization needed; ");
    }
    
    if tool_perf.success_rate < 0.95 {
        description.push_str("Reliability improvements required; ");
    }
    
    if tool_perf.resource_usage.cpu_usage_percent > 50.0 {
        description.push_str("CPU optimization opportunity; ");
    }

    if description.is_empty() {
        description = "Performance tuning for efficiency gains".to_string();
    } else {
        description.truncate(description.len() - 2); // Remove trailing "; "
    }

    description
}

/// Main tools performance report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsPerformanceReport {
    pub report_timestamp: chrono::DateTime<chrono::Utc>,
    pub performance_summary: ToolsPerformanceData,
    pub impact_scores: ImpactScores,
    pub performance_patterns: PerformancePatterns,
    pub recommendations: Vec<String>,
    pub system_health: SystemHealth,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Generate general performance recommendations
fn generate_performance_recommendations(
    tools_data: &ToolsPerformanceData,
    impact_scores: &ImpactScores,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Overall system recommendations
    if tools_data.overall_stats.system_health_score < 7.0 {
        recommendations.push("System health below threshold - investigate performance bottlenecks".to_string());
    }

    if tools_data.resource_utilization.cpu_utilization_avg > 70.0 {
        recommendations.push("High CPU utilization detected - consider scaling resources".to_string());
    }

    if tools_data.resource_utilization.memory_utilization_avg > 80.0 {
        recommendations.push("Memory usage approaching limits - implement memory optimization".to_string());
    }

    // Tool-specific recommendations
    for (tool_name, tool_perf) in &tools_data.tools {
        if tool_perf.impact_score > 8.0 {
            recommendations.push(format!("Prioritize optimization for high-impact tool: {}", tool_name));
        }
        
        if tool_perf.average_response_time_ms > 3000.0 {
            recommendations.push(format!("Critical: {} response time too high ({:.0}ms)", tool_name, tool_perf.average_response_time_ms));
        }
    }

    // Error-related recommendations
    if tools_data.error_analysis.error_rate > 0.05 {
        recommendations.push(format!("High error rate ({:.1}%) - improve error handling", 
            tools_data.error_analysis.error_rate * 100.0));
    }

    if recommendations.is_empty() {
        recommendations.push("Performance metrics are within acceptable ranges".to_string());
    }

    recommendations
}

/// Print human-readable performance report
fn print_human_performance_report(report: &ToolsPerformanceReport, config: &ToolsPerformance) {
    println!("\n{}", "⚡ Tools Performance Analysis".bold().blue());
    println!("Analysis Period: {}", config.time_window);
    println!("Tools Analyzed: {}", report.performance_summary.overall_stats.total_tools);
    println!("Total Operations: {}", report.performance_summary.overall_stats.total_operations);
    println!("Generated: {}\n", report.report_timestamp.format("%Y-%m-%d %H:%M:%S"));

    println!("{}", "System Health".bold().magenta());
    println!("  Overall Score: {:.1}/100", report.system_health.overall_health_score);
    println!("  Success Rate: {}", format_percentage(report.performance_summary.overall_stats.overall_success_rate));
    println!("  Avg Response Time: {:.0}ms", report.performance_summary.overall_stats.average_response_time_ms);
    println!("  Resource Efficiency: {}\n", format_percentage(report.performance_summary.resource_utilization.resource_efficiency));

    println!("{}", "Top Tools by Impact Score".bold().magenta());
    let mut tool_scores_vec: Vec<_> = report.impact_scores.tools_scores.values().collect();
    tool_scores_vec.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap());

    for (i, score) in tool_scores_vec.iter().take(5).enumerate() {
        println!("  {}. {}: {:.1}/10 ({} ops)", 
            i + 1,
            score.tool_name.cyan(),
            score.impact_score,
            report.performance_summary.tools.get(&score.tool_name).map(|t| t.usage_count).unwrap_or(0)
        );
    }
    println!();

    if config.detailed_breakdown {
        println!("{}", "Detailed Tool Performance".bold().magenta());
        for (tool_name, tool_perf) in &report.performance_summary.tools {
            println!("  {} ({}):", tool_name.cyan(), tool_perf.category);
            println!("    Usage: {} operations | Success Rate: {}", 
                tool_perf.usage_count,
                format_percentage(tool_perf.success_rate)
            );
            println!("    Response Time: {:.0}ms avg, {:.0}ms p95", 
                tool_perf.average_response_time_ms,
                tool_perf.p95_response_time_ms
            );
            println!("    Impact Score: {:.1} | Efficiency: {}", 
                tool_perf.impact_score,
                format_percentage(tool_perf.efficiency_rating)
            );
            
            if tool_perf.error_count > 0 {
                println!("    Errors: {} ({})", 
                    tool_perf.error_count,
                    format_percentage(tool_perf.error_rate)
                );
            }
            println!();
        }
    }

    if !report.optimization_opportunities.is_empty() {
        println!("{}", "Optimization Opportunities".bold().magenta());
        for (i, opp) in report.optimization_opportunities.iter().enumerate() {
            println!("  {}. {} - {} (Impact: {}, Effort: {})", 
                i + 1,
                opp.tool_name.cyan(),
                opp.optimization_type,
                format_percentage(opp.impact_potential),
                opp.effort_estimate
            );
            println!("    {}", opp.description.dimmed());
        }
        println!();
    }

    println!("{}", "Performance Recommendations".bold().magenta());
    for (i, rec) in report.recommendations.iter().enumerate() {
        println!("  {}. {}", i + 1, rec);
    }

    if !report.performance_summary.error_analysis.reliability_concerns.is_empty() {
        println!("\n{}", "Reliability Concerns".bold().red());
        for concern in &report.performance_summary.error_analysis.reliability_concerns {
            println!("  • {}", concern);
        }
    }

    println!("\nResource Utilization:");
    println!("  CPU: {} | Memory: {} | Disk I/O: {} | Network I/O: {}", 
        format_percentage(report.performance_summary.resource_utilization.cpu_utilization_avg / 100.0),
        format_percentage(report.performance_summary.resource_utilization.memory_utilization_avg / 100.0),
        format_percentage(report.performance_summary.resource_utilization.disk_io_utilization_avg / 100.0),
        format_percentage(report.performance_summary.resource_utilization.network_io_utilization_avg / 100.0)
    );
}