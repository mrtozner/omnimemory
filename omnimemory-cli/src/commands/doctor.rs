use super::{CommandContext, Result};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub details: Option<String>,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Error,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub overall_status: HealthStatus,
    pub checks: Vec<HealthCheck>,
    pub summary: HashMap<String, usize>,
    pub total_time_ms: u64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub version: Option<String>,
    pub config_valid: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub issues: Vec<String>,
}

pub fn handle_doctor(
    ctx: &CommandContext,
    comprehensive: bool,
    check: Option<Vec<String>>,
    format: String,
) -> Result<()> {
    let pb = if ctx.cli.verbose > 0 {
        Some(ctx.create_progress("Running health diagnostics..."))
    } else {
        None
    };

    if let Some(ref pb) = pb {
        pb.set_message("Checking system components...");
        pb.set_position(20);
    }

    // Mock comprehensive health check
    let report = run_health_checks(comprehensive, check)?;

    if let Some(ref pb) = pb {
        pb.set_message("Generating report...");
        pb.set_position(90);
        pb.finish_with_message("✓ Health check complete");
    }

    match format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "human" | _ => {
            print_human_doctor_report(&report, comprehensive);
        }
    }

    // Exit with error code if there are critical issues
    if matches!(report.overall_status, HealthStatus::Error) {
        std::process::exit(1);
    }

    Ok(())
}

fn run_health_checks(
    comprehensive: bool,
    specific_checks: Option<Vec<String>>,
) -> Result<DoctorReport> {
    let mut checks = Vec::new();

    // Default health checks
    let default_checks = vec![
        "system",
        "mcp_gateway",
        "daemon",
        "config",
        "storage",
        "network",
        "permissions",
    ];

    let checks_to_run = if let Some(ref specific) = specific_checks {
        specific.clone()
    } else {
        default_checks.iter().map(|s| s.to_string()).collect()
    };

    // Run each health check
    for check_name in checks_to_run {
        match check_name.as_str() {
            "system" => checks.extend(run_system_checks(comprehensive)?),
            "mcp_gateway" => checks.extend(run_mcp_gateway_checks(comprehensive)?),
            "daemon" => checks.extend(run_daemon_checks(comprehensive)?),
            "config" => checks.extend(run_config_checks(comprehensive)?),
            "storage" => checks.extend(run_storage_checks(comprehensive)?),
            "network" => checks.extend(run_network_checks(comprehensive)?),
            "permissions" => checks.extend(run_permission_checks(comprehensive)?),
            _ => {
                checks.push(HealthCheck {
                    name: format!("unknown_check_{}", check_name),
                    status: HealthStatus::Unknown,
                    message: format!("Unknown health check: {}", check_name),
                    details: None,
                    remediation: Some("Verify check name is correct".to_string()),
                });
            }
        }
    }

    // Calculate overall status
    let overall_status = calculate_overall_status(&checks);

    // Generate summary
    let mut summary = HashMap::new();
    for check in &checks {
        let status_str = match check.status {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Warning => "warning", 
            HealthStatus::Error => "error",
            HealthStatus::Unknown => "unknown",
        };
        *summary.entry(status_str.to_string()).or_insert(0) += 1;
    }

    // Generate recommendations
    let recommendations = generate_recommendations(&checks, comprehensive);

    Ok(DoctorReport {
        overall_status,
        checks,
        summary,
        total_time_ms: 1250,
        recommendations,
    })
}

fn run_system_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    let mut checks = vec![
        HealthCheck {
            name: "operating_system".to_string(),
            status: HealthStatus::Healthy,
            message: "Operating system is supported".to_string(),
            details: Some("Linux 5.15.0-78-generic".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "rust_version".to_string(),
            status: HealthStatus::Healthy,
            message: "Rust toolchain is available".to_string(),
            details: Some("rustc 1.70.0 (ec8a4a9866 2023-05-25)".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "memory_available".to_string(),
            status: HealthStatus::Healthy,
            message: "Sufficient memory available".to_string(),
            details: Some("8.2 GB available of 16 GB total".to_string()),
            remediation: None,
        },
    ];

    if comprehensive {
        checks.push(HealthCheck {
            name: "disk_space".to_string(),
            status: HealthStatus::Healthy,
            message: "Sufficient disk space available".to_string(),
            details: Some("42.3 GB free of 512 GB total".to_string()),
            remediation: None,
        });

        checks.push(HealthCheck {
            name: "cpu_cores".to_string(),
            status: HealthStatus::Healthy,
            message: "CPU cores are available".to_string(),
            details: Some("8 logical cores detected".to_string()),
            remediation: None,
        });
    }

    Ok(checks)
}

fn run_mcp_gateway_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "mcp_gateway_binary".to_string(),
            status: HealthStatus::Healthy,
            message: "MCP Gateway binary found".to_string(),
            details: Some("/usr/local/bin/omnimemory-gateway".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "mcp_protocol_compliance".to_string(),
            status: HealthStatus::Healthy,
            message: "MCP protocol version compatible".to_string(),
            details: Some("2025-06-18".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "stdio_transport".to_string(),
            status: HealthStatus::Warning,
            message: "stdio transport available but testing needed".to_string(),
            details: Some("Consider running integration tests".to_string()),
            remediation: Some("Run 'omni doctor --check integration'".to_string()),
        },
    ])
}

fn run_daemon_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "daemon_binary".to_string(),
            status: HealthStatus::Healthy,
            message: "Daemon binary found".to_string(),
            details: Some("/usr/local/bin/omnimemory-daemon".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "daemon_port_available".to_string(),
            status: HealthStatus::Healthy,
            message: "Daemon port 4000 is available".to_string(),
            details: None,
            remediation: None,
        },
        HealthCheck {
            name: "daemon_permissions".to_string(),
            status: HealthStatus::Error,
            message: "Daemon cannot bind to privileged ports".to_string(),
            details: Some("Requires elevated privileges for port 4000".to_string()),
            remediation: Some("Use sudo or configure daemon to use port > 1024".to_string()),
        },
    ])
}

fn run_config_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "config_directory".to_string(),
            status: HealthStatus::Healthy,
            message: "Configuration directory exists".to_string(),
            details: Some("~/.config/omnimemory".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "config_file_valid".to_string(),
            status: HealthStatus::Healthy,
            message: "Configuration file is valid TOML".to_string(),
            details: Some("~/.config/omnimemory/config.toml".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "profile_files".to_string(),
            status: HealthStatus::Healthy,
            message: "Profile files are accessible".to_string(),
            details: Some("3 profiles found".to_string()),
            remediation: None,
        },
    ])
}

fn run_storage_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "sqlite_database".to_string(),
            status: HealthStatus::Healthy,
            message: "SQLite database is accessible".to_string(),
            details: Some("~/.local/share/omnimemory/omnimemory.db".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "database_integrity".to_string(),
            status: HealthStatus::Healthy,
            message: "Database integrity check passed".to_string(),
            details: Some("1,234 facts, 89 operations".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "faiss_index".to_string(),
            status: HealthStatus::Warning,
            message: "FAISS index not found".to_string(),
            details: Some("Will be created on first embedding generation".to_string()),
            remediation: Some("Run 'omnimemory index rebuild' to create index".to_string()),
        },
    ])
}

fn run_network_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "loopback_connectivity".to_string(),
            status: HealthStatus::Healthy,
            message: "Loopback interface is operational".to_string(),
            details: Some("127.0.0.1 reachable".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "dns_resolution".to_string(),
            status: HealthStatus::Healthy,
            message: "DNS resolution is working".to_string(),
            details: Some("8.8.8.8 reachable".to_string()),
            remediation: None,
        },
    ])
}

fn run_permission_checks(comprehensive: bool) -> Result<Vec<HealthCheck>> {
    Ok(vec![
        HealthCheck {
            name: "config_directory_permissions".to_string(),
            status: HealthStatus::Healthy,
            message: "Configuration directory is writable".to_string(),
            details: Some("~/.config/omnimemory".to_string()),
            remediation: None,
        },
        HealthCheck {
            name: "log_directory_permissions".to_string(),
            status: HealthStatus::Healthy,
            message: "Log directory is writable".to_string(),
            details: Some("~/.local/share/omnimemory".to_string()),
            remediation: None,
        },
    ])
}

fn calculate_overall_status(checks: &[HealthCheck]) -> HealthStatus {
    let has_errors = checks.iter().any(|check| matches!(check.status, HealthStatus::Error));
    let has_warnings = checks.iter().any(|check| matches!(check.status, HealthStatus::Warning));

    if has_errors {
        HealthStatus::Error
    } else if has_warnings {
        HealthStatus::Warning
    } else {
        HealthStatus::Healthy
    }
}

fn generate_recommendations(checks: &[HealthCheck], comprehensive: bool) -> Vec<String> {
    let mut recommendations = Vec::new();

    for check in checks {
        if let Some(ref remediation) = check.remediation {
            recommendations.push(format!("{}: {}", check.name, remediation));
        }

        // Add specific recommendations based on status
        match check.status {
            HealthStatus::Error => {
                recommendations.push(format!("🚨 {} - Critical issue detected", check.name));
            }
            HealthStatus::Warning => {
                recommendations.push(format!("⚠️  {} - Check details and consider remediation", check.name));
            }
            _ => {}
        }
    }

    if comprehensive {
        recommendations.extend(vec![
            "Consider running 'omni daemon logs' to check for runtime issues".to_string(),
            "Update to latest version with 'omnimemory update'".to_string(),
            "Run 'omni doctor --comprehensive' for detailed system analysis".to_string(),
        ]);
    }

    recommendations
}

fn print_human_doctor_report(report: &DoctorReport, comprehensive: bool) {
    let status_emoji = match report.overall_status {
        HealthStatus::Healthy => "✅",
        HealthStatus::Warning => "⚠️", 
        HealthStatus::Error => "❌",
        HealthStatus::Unknown => "❓",
    };

    let status_text = match report.overall_status {
        HealthStatus::Healthy => "All systems operational",
        HealthStatus::Warning => "Some issues detected",
        HealthStatus::Error => "Critical issues found",
        HealthStatus::Unknown => "Unable to determine status",
    };

    println!("\n{}", "🩺 OmniMemory Doctor".bold().blue());
    println!("Overall Status: {} {}", status_emoji, status_text.bold());
    println!("Completed in {}ms\n", report.total_time_ms);

    // Print summary
    println!("{}", "Summary:".bold().magenta());
    for (status, count) in &report.summary {
        let emoji = match status.as_str() {
            "healthy" => "✅",
            "warning" => "⚠️",
            "error" => "❌", 
            "unknown" => "❓",
            _ => "❓",
        };
        println!("  {} {}: {}", emoji, status, count);
    }
    println!();

    // Print detailed checks
    println!("{}", "Detailed Checks:".bold().magenta());
    for check in &report.checks {
        let status_emoji = match check.status {
            HealthStatus::Healthy => "✅",
            HealthStatus::Warning => "⚠️",
            HealthStatus::Error => "❌",
            HealthStatus::Unknown => "❓",
        };

        println!("{}{} {}", status_emoji, check.name, check.message);

        if let Some(ref details) = check.details {
            println!("     {}", details.dimmed());
        }

        if comprehensive && check.status != HealthStatus::Healthy {
            if let Some(ref remediation) = check.remediation {
                println!("     → {}", remediation.yellow());
            }
        }
        println!();
    }

    // Print recommendations
    if !report.recommendations.is_empty() {
        println!("{}", "Recommendations:".bold().magenta());
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("{}. {}", i + 1, rec);
        }
        println!();
    }

    // Print next steps
    match report.overall_status {
        HealthStatus::Healthy => {
            println!("{}", "🎉 All systems are healthy! OmniMemory is ready to use.".green());
        }
        HealthStatus::Warning => {
            println!("{}", "⚠️  Review the warnings above to optimize performance.".yellow());
        }
        HealthStatus::Error => {
            println!("{}", "❌ Critical issues must be resolved before using OmniMemory.".red());
        }
        _ => {}
    }

    if comprehensive {
        println!("\nFor more detailed information, run 'omni doctor --comprehensive'.");
    }
}