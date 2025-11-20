use super::{CommandContext, Result};
use colored::Colorize;

/// Handle remove command - delegate to Python universal remover
pub fn handle_remove(
    ctx: &CommandContext,
    dry_run: bool,
    remove_all: bool,
    tools: Option<&str>,
) -> Result<()> {
    println!("\n{}", "🗑️  OmniMemory Configuration Removal".bold().yellow());
    println!("{}", "═".repeat(60).dimmed());
    if dry_run {
        println!("{}", "Dry-run mode: previewing removal...".dimmed());
    } else {
        println!("{}", "Delegating to removal tool with automatic backups...".dimmed());
    }
    println!();

    // Find Python omnimemory CLI
    let omnimemory_cmd = find_omnimemory_cli()?;

    // Build command arguments
    let mut args = vec!["remove".to_string()];

    if dry_run {
        args.push("--dry-run".to_string());
    }

    if remove_all {
        args.push("--all".to_string());
    }

    if let Some(tool_list) = tools {
        args.push("--tools".to_string());
        args.push(tool_list.to_string());
    }

    // Execute Python removal tool
    println!("{}", "  Starting removal tool...".dimmed());
    println!();

    let status = if omnimemory_cmd == "python3" {
        // Using python -m invocation
        let mut full_args = vec!["-m".to_string(), "omnimemory.cli.cli".to_string()];
        full_args.extend(args);
        std::process::Command::new(&omnimemory_cmd)
            .args(&full_args)
            .status()?
    } else {
        // Using direct omnimemory command
        std::process::Command::new(&omnimemory_cmd)
            .args(&args)
            .status()?
    };

    if !status.success() {
        return Err(anyhow::anyhow!(
            "Removal tool failed with status: {}",
            status.code().unwrap_or(-1)
        ));
    }

    // Success message
    if !dry_run {
        println!();
        println!("{}", "═".repeat(60).dimmed());
        println!("{}", "✨ Removal complete!".bold().green());
        println!();
        println!("{}", "Quick commands:".bold());
        println!("  {} {} - Re-initialize if needed", "•".cyan(), "omni init".yellow());
        println!("  {} {} - View system status", "•".cyan(), "omni doctor".yellow());
        println!();
    }

    Ok(())
}

/// Find the omnimemory Python CLI (reuse from init.rs)
fn find_omnimemory_cli() -> Result<String> {
    // Try which command first
    if let Ok(output) = std::process::Command::new("which")
        .arg("omnimemory")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            let path = path.trim();
            if !path.is_empty() {
                return Ok(path.to_string());
            }
        }
    }

    // Try common installation locations
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;

    let locations = vec![
        home.join(".local/bin/omnimemory"),
        std::path::PathBuf::from("/usr/local/bin/omnimemory"),
        std::path::PathBuf::from("/opt/homebrew/bin/omnimemory"),
    ];

    for loc in locations {
        if loc.exists() {
            return Ok(loc.to_string_lossy().to_string());
        }
    }

    // Try python -m as last resort
    if let Ok(output) = std::process::Command::new("python3")
        .args(&["-c", "import omnimemory.cli.cli; print('found')"])
        .output()
    {
        if output.status.success() {
            return Ok("python3".to_string());
        }
    }

    Err(anyhow::anyhow!(
        "Python omnimemory CLI not found. Please install it first:\n\
        cd code/omnimemory && pip3 install -e ."
    ))
}
