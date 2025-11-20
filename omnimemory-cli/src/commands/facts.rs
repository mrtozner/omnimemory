use super::{CommandContext, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub entity_type: String,
    pub entity_id: String,
    pub fact_type: String,
    pub content: serde_json::Value,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQueryResult {
    pub query: String,
    pub results: Vec<Fact>,
    pub total_found: usize,
    pub search_time_ms: u64,
    pub result_format: crate::FactFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactTimeline {
    pub entity: String,
    pub timeline: Vec<TimelineEvent>,
    pub date_range: (DateTime<Utc>, DateTime<Utc>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub description: String,
    pub details: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactStatistics {
    pub entity_type: String,
    pub total_facts: usize,
    pub facts_by_type: HashMap<String, usize>,
    pub facts_by_source: HashMap<String, usize>,
    pub recent_activity: Vec<String>,
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

pub fn handle_facts(
    ctx: &CommandContext,
    query: Option<String>,
    entity: Option<String>,
    since: Option<String>,
    limit: usize,
    format: crate::FactFormat,
) -> Result<()> {
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Querying facts database..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Searching fact database...");
        pb.set_position(50);
    }

    match format {
        crate::FactFormat::Facts => {
            let results = query_facts(query, entity, since, limit)?;
            
            match ctx.output_format {
                super::OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&results)?);
                }
                super::OutputFormat::Human => {
                    print_human_facts(&results);
                }
                _ => {
                    println!("{}", serde_json::to_string(&results)?);
                }
            }
        }
        crate::FactFormat::Timeline => {
            let timeline = generate_timeline(entity, since)?;
            
            match ctx.output_format {
                super::OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&timeline)?);
                }
                super::OutputFormat::Human => {
                    print_human_timeline(&timeline);
                }
                _ => {
                    println!("{}", serde_json::to_string(&timeline)?);
                }
            }
        }
        crate::FactFormat::Statistics => {
            let stats = generate_statistics(entity, since)?;
            
            match ctx.output_format {
                super::OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&stats)?);
                }
                super::OutputFormat::Human => {
                    print_human_statistics(&stats);
                }
                _ => {
                    println!("{}", serde_json::to_string(&stats)?);
                }
            }
        }
    }

    if let Some(ref pb) = pb {
        pb.finish_with_message("✓ Query completed");
    }

    Ok(())
}

fn query_facts(
    query: Option<String>,
    entity: Option<String>,
    since: Option<String>,
    limit: usize,
) -> Result<FactQueryResult> {
    let mock_facts = get_mock_facts();
    
    let filtered_facts: Vec<Fact> = mock_facts
        .into_iter()
        .filter(|fact| {
            // Filter by query text if provided
            if let Some(ref search_text) = query {
                let content_str = serde_json::to_string(&fact.content).unwrap_or_default();
                let search_text = search_text.to_lowercase();
                
                content_str.to_lowercase().contains(&search_text) ||
                fact.entity_id.to_lowercase().contains(&search_text) ||
                fact.fact_type.to_lowercase().contains(&search_text) ||
                fact.tags.iter().any(|tag| tag.to_lowercase().contains(&search_text))
            } else {
                true
            }
        })
        .filter(|fact| {
            // Filter by entity if provided
            if let Some(ref entity_filter) = entity {
                fact.entity_id.to_lowercase() == entity_filter.to_lowercase() ||
                fact.entity_type.to_lowercase() == entity_filter.to_lowercase()
            } else {
                true
            }
        })
        .take(limit)
        .collect();

    let total_found = filtered_facts.len();

    Ok(FactQueryResult {
        query: query.unwrap_or_else(|| "all facts".to_string()),
        results: filtered_facts,
        total_found,
        search_time_ms: 23,
        result_format: crate::FactFormat::Facts,
    })
}

fn generate_timeline(entity: Option<String>, since: Option<String>) -> Result<FactTimeline> {
    let timeline_events = vec![
        TimelineEvent {
            timestamp: Utc::now() - chrono::Duration::hours(2),
            event_type: "command_executed".to_string(),
            description: "Executed 'git status' command".to_string(),
            details: HashMap::from([
                ("command".to_string(), serde_json::Value::String("git status".to_string())),
                ("exit_code".to_string(), serde_json::Value::Number(0.into())),
            ]),
        },
        TimelineEvent {
            timestamp: Utc::now() - chrono::Duration::hours(1),
            event_type: "file_modified".to_string(),
            description: "Modified src/main.rs".to_string(),
            details: HashMap::from([
                ("file".to_string(), serde_json::Value::String("src/main.rs".to_string())),
                ("lines_added".to_string(), serde_json::Value::Number(42.into())),
                ("lines_removed".to_string(), serde_json::Value::Number(8.into())),
            ]),
        },
        TimelineEvent {
            timestamp: Utc::now() - chrono::Duration::minutes(30),
            event_type: "error_occurred".to_string(),
            description: "Compilation error in proc-macro2".to_string(),
            details: HashMap::from([
                ("error_type".to_string(), serde_json::Value::String("compilation".to_string())),
                ("severity".to_string(), serde_json::Value::String("medium".to_string())),
            ]),
        },
        TimelineEvent {
            timestamp: Utc::now() - chrono::Duration::minutes(15),
            event_type: "suggestion_received".to_string(),
            description: "AI suggested running 'cargo test'".to_string(),
            details: HashMap::from([
                ("suggestion_id".to_string(), serde_json::Value::String("sug-002".to_string())),
                ("confidence".to_string(), serde_json::Value::Number(88.into())),
            ]),
        },
        TimelineEvent {
            timestamp: Utc::now() - chrono::Duration::minutes(5),
            event_type: "context_assembled".to_string(),
            description: "Gathered context for current directory".to_string(),
            details: HashMap::from([
                ("context_size".to_string(), serde_json::Value::String("2.4KB".to_string())),
                ("included_items".to_string(), serde_json::Value::Number(5.into())),
            ]),
        },
    ];

    let timeline = FactTimeline {
        entity: entity.unwrap_or_else(|| "current_session".to_string()),
        timeline: timeline_events,
        date_range: (Utc::now() - chrono::Duration::hours(2), Utc::now()),
    };

    Ok(timeline)
}

fn generate_statistics(entity: Option<String>, since: Option<String>) -> Result<FactStatistics> {
    Ok(FactStatistics {
        entity_type: entity.unwrap_or_else(|| "session".to_string()),
        total_facts: 156,
        facts_by_type: HashMap::from([
            ("command_execution".to_string(), 45),
            ("file_operation".to_string(), 32),
            ("error".to_string(), 12),
            ("suggestion".to_string(), 28),
            ("context".to_string(), 39),
        ]),
        facts_by_source: HashMap::from([
            ("shell_history".to_string(), 67),
            ("file_system".to_string(), 34),
            ("git_metadata".to_string(), 23),
            ("ai_analysis".to_string(), 32),
        ]),
        recent_activity: vec![
            "high command execution rate".to_string(),
            "frequent file modifications".to_string(),
            "active development session".to_string(),
        ],
        time_range: (Utc::now() - chrono::Duration::days(1), Utc::now()),
    })
}

fn get_mock_facts() -> Vec<Fact> {
    vec![
        Fact {
            id: "fact-001".to_string(),
            entity_type: "command".to_string(),
            entity_id: "git-status".to_string(),
            fact_type: "execution_frequency".to_string(),
            content: serde_json::to_value("Executed 23 times in last 7 days").unwrap(),
            confidence: 0.95,
            timestamp: Utc::now() - chrono::Duration::hours(2),
            source: "shell_history".to_string(),
            tags: vec!["git".to_string(), "status".to_string(), "frequent".to_string()],
        },
        Fact {
            id: "fact-002".to_string(),
            entity_type: "file".to_string(),
            entity_id: "src/main.rs".to_string(),
            fact_type: "modification_pattern".to_string(),
            content: serde_json::to_value("Modified 8 times today, avg 45 lines per modification").unwrap(),
            confidence: 0.88,
            timestamp: Utc::now() - chrono::Duration::minutes(30),
            source: "file_system".to_string(),
            tags: vec!["rust".to_string(), "main".to_string(), "active".to_string()],
        },
        Fact {
            id: "fact-003".to_string(),
            entity_type: "directory".to_string(),
            entity_id: "/workspace/omnimemory-cli".to_string(),
            fact_type: "project_type".to_string(),
            content: serde_json::to_value("Rust CLI project with cargo dependencies").unwrap(),
            confidence: 0.92,
            timestamp: Utc::now() - chrono::Duration::hours(1),
            source: "file_system".to_string(),
            tags: vec!["rust".to_string(), "cli".to_string(), "cargo".to_string()],
        },
        Fact {
            id: "fact-004".to_string(),
            entity_type: "suggestion".to_string(),
            entity_id: "cargo-test".to_string(),
            fact_type: "success_rate".to_string(),
            content: serde_json::to_value("Suggested 12 times, accepted 8 times (67% success rate)").unwrap(),
            confidence: 0.85,
            timestamp: Utc::now() - chrono::Duration::minutes(15),
            source: "ai_analysis".to_string(),
            tags: vec!["cargo".to_string(), "test".to_string(), "useful".to_string()],
        },
        Fact {
            id: "fact-005".to_string(),
            entity_type: "error".to_string(),
            entity_id: "compilation-error".to_string(),
            fact_type: "recurrence".to_string(),
            content: serde_json::to_value("Same error occurred 3 times this week, always related to proc-macro2").unwrap(),
            confidence: 0.78,
            timestamp: Utc::now() - chrono::Duration::minutes(45),
            source: "error_tracking".to_string(),
            tags: vec!["compilation".to_string(), "proc-macro2".to_string(), "recurring".to_string()],
        },
    ]
}

fn print_human_facts(result: &FactQueryResult) {
    println!("\n{}", "🔍 Fact Search Results".bold().blue());
    println!("Query: {}", result.query.cyan());
    println!("Found: {} facts in {}ms\n", result.total_found, result.search_time_ms);

    for (i, fact) in result.results.iter().enumerate() {
        println!("{}. [{}] {} - {} ({:.1}%)", 
            i + 1,
            fact.entity_type.cyan(),
            fact.entity_id.yellow(),
            fact.fact_type.green(),
            fact.confidence * 100.0
        );

        let content_str = match &fact.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            _ => fact.content.to_string(),
        };

        println!("   {}", content_str.dimmed());
        println!("   Source: {} | {}", 
            fact.source.dimmed(), 
            fact.timestamp.format("%H:%M:%S").to_string().dimmed()
        );

        if !fact.tags.is_empty() {
            println!("   Tags: {}", fact.tags.join(", ").dimmed());
        }
        println!();
    }

    if result.results.is_empty() {
        println!("{}", "No facts found matching your query.".yellow());
    }
}

fn print_human_timeline(timeline: &FactTimeline) {
    println!("\n{}", format!("📅 Timeline: {}", timeline.entity).bold().blue());
    println!("{} - {}\n", 
        timeline.date_range.0.format("%Y-%m-%d %H:%M"),
        timeline.date_range.1.format("%Y-%m-%d %H:%M")
    );

    for (i, event) in timeline.timeline.iter().enumerate() {
        let event_emoji = match event.event_type.as_str() {
            "command_executed" => "⚡",
            "file_modified" => "📝",
            "error_occurred" => "❌",
            "suggestion_received" => "🤖",
            "context_assembled" => "📋",
            _ => "📄",
        };

        println!("{}. {} [{}] {}", 
            i + 1,
            event_emoji,
            event.timestamp.format("%H:%M:%S").to_string().cyan(),
            event.description.bold()
        );

        if !event.details.is_empty() {
            println!("   Details:");
            for (key, value) in &event.details {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => value.to_string(),
                };
                println!("     {}: {}", key.cyan(), value_str.dimmed());
            }
        }
        println!();
    }

    if timeline.timeline.is_empty() {
        println!("{}", "No timeline events found for this entity.".yellow());
    }
}

fn print_human_statistics(stats: &FactStatistics) {
    println!("\n{}", "📊 Fact Statistics".bold().blue());
    println!("Entity Type: {}", stats.entity_type.cyan());
    println!("Total Facts: {}\n", stats.total_facts.to_string().cyan());

    println!("{}", "Facts by Type:".bold().magenta());
    for (fact_type, count) in &stats.facts_by_type {
        println!("  {}: {}", fact_type.yellow(), count);
    }
    println!();

    println!("{}", "Facts by Source:".bold().magenta());
    for (source, count) in &stats.facts_by_source {
        println!("  {}: {}", source.yellow(), count);
    }
    println!();

    println!("{}", "Recent Activity:".bold().magenta());
    for activity in &stats.recent_activity {
        println!("  • {}", activity);
    }
    println!();

    println!("Time Range: {} - {}",
        stats.time_range.0.format("%Y-%m-%d %H:%M").to_string().dimmed(),
        stats.time_range.1.format("%Y-%m-%d %H:%M").to_string().dimmed()
    );
}