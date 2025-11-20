use super::{CommandContext, Result};
use colored::Colorize;

/// Handle init command - delegate to Python universal installer
pub fn handle_init(
    ctx: &CommandContext,
    dry_run: bool,
    silent: bool,
    tools: Option<&str>,
    enterprise: bool,
    force: bool,
    skip_claude_md: bool,
    mcp_path: Option<String>,
) -> Result<()> {
    println!("\n{}", "🚀 OmniMemory Universal Initialization".bold().cyan());
    println!("{}", "═".repeat(60).dimmed());
    println!("{}", "Delegating to comprehensive universal installer...".dimmed());
    println!();

    // Find Python omnimemory CLI
    let omnimemory_cmd = find_omnimemory_cli()?;

    // Build command arguments
    let mut args = vec!["init".to_string()];

    if dry_run {
        args.push("--dry-run".to_string());
    }

    if silent || force {
        // Map --force to --silent for non-interactive mode
        args.push("--silent".to_string());
    }

    if let Some(tool_list) = tools {
        args.push("--tools".to_string());
        args.push(tool_list.to_string());
    }

    if enterprise {
        args.push("--enterprise".to_string());
    }

    // Note about options not directly supported by Python CLI
    if skip_claude_md {
        println!("  {} Note: Python installer will prompt for Claude.md integration", "ℹ".blue());
    }

    if let Some(path) = &mcp_path {
        println!("  {} Custom MCP path will be used by Python installer", "ℹ".blue());
        println!("    {}", path.dimmed());
    }

    // Interactive mode based on context
    if !ctx.is_interactive() && !silent {
        args.push("--silent".to_string());
    }

    // Execute Python universal installer
    println!("{}", "  Starting universal installer...".dimmed());
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
            "Universal installer failed with status: {}",
            status.code().unwrap_or(-1)
        ));
    }

    // Success message
    println!();
    println!("{}", "═".repeat(60).dimmed());
    println!("{}", "✨ Universal initialization complete!".bold().green());
    println!();
    println!("{}", "Quick commands:".bold());
    println!("  {} {} - View system status", "•".cyan(), "omni doctor".yellow());
    println!("  {} {} - Get AI suggestions", "•".cyan(), "omni suggest".yellow());
    println!("  {} {} - View dashboard", "•".cyan(), "omni dashboard".yellow());
    println!("  {} {} - Check tool performance", "•".cyan(), "omni tools-performance".yellow());
    println!();
    println!("{}", "For help: omni --help".dimmed());
    println!();

    Ok(())
}

/// Find the omnimemory Python CLI
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
