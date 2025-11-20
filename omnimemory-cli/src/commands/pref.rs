use super::{CommandContext, Result};
use crate::PrefScope;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Macro to create HashMap literals (since standard library doesn't have one)
macro_rules! hash_map {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = std::collections::HashMap::new();
            $(map.insert($key, $value);)*
            map
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub key: String,
    pub value: String,
    pub scope: PrefScope,
    pub source: String,
    pub description: Option<String>,
    pub default_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceSet {
    pub preferences: HashMap<String, Preference>,
    pub scope: PrefScope,
    pub total_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceSource {
    pub key: String,
    pub value: String,
    pub sources: Vec<PrefSourceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefSourceInfo {
    pub scope: PrefScope,
    pub value: Option<String>,
    pub source_file: Option<String>,
}

pub fn handle_pref(
    ctx: &CommandContext,
    action: crate::PrefCommand,
) -> Result<()> {
    match action {
        crate::PrefCommand::Set { key, value, scope } => {
            handle_set_pref(ctx, &key, &value, scope)?;
        }
        crate::PrefCommand::Get { key, scope, show_source } => {
            handle_get_pref(ctx, &key, scope, show_source)?;
        }
        crate::PrefCommand::List { scope, filter } => {
            handle_list_pref(ctx, scope, filter)?;
        }
        crate::PrefCommand::Reset { scope, keys } => {
            handle_reset_pref(ctx, scope, keys)?;
        }
    }

    Ok(())
}

fn handle_set_pref(
    ctx: &CommandContext,
    key: &str,
    value: &str,
    scope: super::PrefScope,
) -> Result<()> {
    // Validate the key format
    if !is_valid_key(key) {
        return Err(anyhow::anyhow!("Invalid preference key format: {}", key));
    }

    // Mock implementation - would save to appropriate config file
    let pref = Preference {
        key: key.to_string(),
        value: value.to_string(),
        scope: scope.clone(),
        source: get_config_file_path(&scope).to_string_lossy().to_string(),
        description: get_key_description(key),
        default_value: Some(get_default_value(key)),
    };

    if ctx.is_interactive() {
        println!("Setting preference '{}' to '{}' (scope: {})", 
            key.cyan(), value.green(), scope.to_string().yellow());
    }

    ctx.print_success(&format!("Preference '{}' set successfully", key));

    Ok(())
}

fn handle_get_pref(
    ctx: &CommandContext,
    key: &str,
    scope: Option<super::PrefScope>,
    show_source: bool,
) -> Result<()> {
    // Mock implementation - would read from config files
    let sources = get_mock_sources_for_key(key, scope.as_ref().map(|s| s.clone()));
    let resolved_value = resolve_preference_value(&sources);

    match ctx.output_format {
        super::OutputFormat::Json => {
            let result = PreferenceSource {
                key: key.to_string(),
                value: resolved_value.clone(),
                sources,
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        super::OutputFormat::Human => {
            println!("\n{}", format!("🔧 Preference: {}", key).bold().blue());
            println!("Value: {}", resolved_value.cyan());
            
            if show_source {
                println!("\n{}", "Value Sources:".bold().magenta());
                for source in &sources {
                    let value_str = source.value.as_ref().map(|v| v.as_str()).unwrap_or("(unset)");
                    let status = if source.value.is_some() { "✓" } else { "−" };
                    println!("  {} {}: {} ({})", 
                        status.green(), 
                        source.scope.to_string().yellow(),
                        value_str,
                        source.source_file.as_deref().unwrap_or("built-in")
                    );
                }
            }

            if let Some(desc) = get_key_description(key) {
                println!("\nDescription: {}", desc.dimmed());
            }
        }
        _ => {
            println!("{}", resolved_value);
        }
    }

    Ok(())
}

fn handle_list_pref(
    ctx: &CommandContext,
    scope: Option<super::PrefScope>,
    filter: Option<String>,
) -> Result<()> {
    let prefs = get_mock_preferences(scope.as_ref().map(|s| s.clone()));

    // Apply filter if provided
    let filtered_prefs: HashMap<String, Preference> = if let Some(filter_str) = filter {
        prefs.preferences.into_iter()
            .filter(|(key, _)| key.to_lowercase().contains(&filter_str.to_lowercase()))
            .collect()
    } else {
        prefs.preferences
    };

    match ctx.output_format {
        super::OutputFormat::Json => {
            let result = PreferenceSet {
                preferences: filtered_prefs.clone(),
                scope: prefs.scope,
                total_count: filtered_prefs.len(),
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        super::OutputFormat::Human => {
            println!("\n{}", format!("⚙️  Preferences ({})", prefs.scope).bold().blue());
            println!("Total: {} preferences\n", filtered_prefs.len().to_string().cyan());

            for (key, pref) in &filtered_prefs {
                println!("{} = {}", key.cyan(), pref.value.yellow());
                if let Some(ref desc) = pref.description {
                    println!("   {}", desc.dimmed());
                }
                println!("   Scope: {} | Source: {}\n",
                    pref.scope.to_string().dimmed(),
                    pref.source.dimmed()
                );
            }

            if filtered_prefs.is_empty() {
                println!("{}", "No preferences found matching the filter.".yellow());
            }
        }
        _ => {
            for (key, pref) in &filtered_prefs {
                println!("{} = {}", key, pref.value);
            }
        }
    }

    Ok(())
}

fn handle_reset_pref(
    ctx: &CommandContext,
    scope: Option<super::PrefScope>,
    keys: Option<Vec<String>>,
) -> Result<()> {
    let scope_str = scope.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "all scopes".to_string());

    if ctx.is_interactive() {
        if let Some(ref key_list) = keys {
            println!("Resetting preferences: {} (scope: {})", 
                key_list.join(", ").cyan(), scope_str.yellow());
        } else {
            println!("Resetting all preferences in scope: {}", scope_str.yellow());
        }
    }

    let action = if keys.is_some() { "reset" } else { "cleared" };
    ctx.print_success(&format!("Preferences {} successfully", action));

    Ok(())
}

// Helper functions

fn is_valid_key(key: &str) -> bool {
    // Key must contain only alphanumeric characters, dots, underscores, and hyphens
    key.chars().all(|c| c.is_alphanumeric() || c == '.' || c == '_' || c == '-') &&
    !key.starts_with('.') &&
    !key.ends_with('.') &&
    key.len() <= 100
}

fn get_config_file_path(scope: &super::PrefScope) -> std::path::PathBuf {
    match scope {
        super::PrefScope::User => {
            dirs::home_dir()
                .map(|home| home.join(".config").join("omnimemory").join("preferences.toml"))
                .unwrap_or_else(|| std::path::PathBuf::from("~/.config/omnimemory/preferences.toml"))
        }
        super::PrefScope::Project => {
            std::path::PathBuf::from(".omnimemory").join("preferences.toml")
        }
        super::PrefScope::System => {
            std::path::PathBuf::from("/etc/omnimemory/preferences.toml")
        }
    }
}

fn get_key_description(key: &str) -> Option<String> {
    let descriptions = hash_map![
        "suggestion.max_results" => "Maximum number of suggestions to generate",
        "context.max_size" => "Maximum context size for AI operations",
        "output.format" => "Default output format (human, json, plain, table)",
        "interactive.enabled" => "Enable interactive prompts and confirmations",
        "tools.docker.enabled" => "Enable Docker-related tools and suggestions",
        "tools.git.enabled" => "Enable Git-related tools and suggestions",
        "suggestion.domains" => "Preferred suggestion domains",
        "daemon.enabled" => "Enable OmniMemory daemon",
        "daemon.port" => "Port for daemon HTTP server",
        "logging.level" => "Logging level (error, warn, info, debug)",
    ];

    descriptions.get(key).map(|s| s.to_string())
}

fn get_default_value(key: &str) -> String {
    let defaults = hash_map![
        "suggestion.max_results" => "5",
        "context.max_size" => "10KB",
        "output.format" => "human",
        "interactive.enabled" => "true",
        "tools.docker.enabled" => "true",
        "tools.git.enabled" => "true",
        "suggestion.domains" => "[]",
        "daemon.enabled" => "false",
        "daemon.port" => "4000",
        "logging.level" => "info",
    ];

    defaults.get(key).unwrap_or(&"").to_string()
}

fn get_mock_sources_for_key(
    key: &str,
    scope: Option<super::PrefScope>,
) -> Vec<PrefSourceInfo> {
    // Determine which scopes to check
    let scopes = if let Some(ref specific_scope) = scope {
        vec![specific_scope.clone()]
    } else {
        vec![
            super::PrefScope::System,
            super::PrefScope::User,
            super::PrefScope::Project,
        ]
    };

    let mut sources = Vec::new();

    for scope in scopes {
        let value = match key {
            "suggestion.max_results" => match scope {
                super::PrefScope::System => None,
                super::PrefScope::User => Some("8".to_string()),
                super::PrefScope::Project => Some("10".to_string()),
            },
            "output.format" => match scope {
                super::PrefScope::System => Some("human".to_string()),
                super::PrefScope::User => Some("json".to_string()),
                super::PrefScope::Project => None,
            },
            _ => None,
        };

        let source_file = get_config_file_path(&scope).to_string_lossy().to_string();
        sources.push(PrefSourceInfo {
            scope,
            value,
            source_file: Some(source_file),
        });
    }

    sources
}

fn resolve_preference_value(sources: &[PrefSourceInfo]) -> String {
    // Return the first non-None value, starting from project scope
    for source in sources.iter().rev() { // Reverse to prioritize project over user over system
        if let Some(ref value) = source.value {
            return value.clone();
        }
    }

    // If no value found, return the default
    "default".to_string()
}

fn get_mock_preferences(scope: Option<super::PrefScope>) -> PreferenceSet {
    let all_prefs = hash_map![
        "suggestion.max_results" => Preference {
            key: "suggestion.max_results".to_string(),
            value: "10".to_string(),
            scope: super::PrefScope::User,
            source: "~/.config/omnimemory/preferences.toml".to_string(),
            description: Some("Maximum number of suggestions to generate".to_string()),
            default_value: Some("5".to_string()),
        },
        "context.max_size" => Preference {
            key: "context.max_size".to_string(),
            value: "50KB".to_string(),
            scope: super::PrefScope::User,
            source: "~/.config/omnimemory/preferences.toml".to_string(),
            description: Some("Maximum context size for AI operations".to_string()),
            default_value: Some("10KB".to_string()),
        },
        "output.format" => Preference {
            key: "output.format".to_string(),
            value: "json".to_string(),
            scope: super::PrefScope::User,
            source: "~/.config/omnimemory/preferences.toml".to_string(),
            description: Some("Default output format".to_string()),
            default_value: Some("human".to_string()),
        },
        "tools.docker.enabled" => Preference {
            key: "tools.docker.enabled".to_string(),
            value: "true".to_string(),
            scope: super::PrefScope::System,
            source: "/etc/omnimemory/preferences.toml".to_string(),
            description: Some("Enable Docker-related tools and suggestions".to_string()),
            default_value: Some("true".to_string()),
        },
        "daemon.enabled" => Preference {
            key: "daemon.enabled".to_string(),
            value: "false".to_string(),
            scope: super::PrefScope::System,
            source: "/etc/omnimemory/preferences.toml".to_string(),
            description: Some("Enable OmniMemory daemon".to_string()),
            default_value: Some("false".to_string()),
        },
    ];

    // Filter by scope if specified
    let filtered_prefs: HashMap<String, Preference> = if let Some(target_scope) = &scope {
        all_prefs.into_iter()
            .filter(|(_, pref)| &pref.scope == target_scope)
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    } else {
        all_prefs.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    };

    let total_count = filtered_prefs.len();
    let effective_scope = scope.unwrap_or(super::PrefScope::User);

    PreferenceSet {
        preferences: filtered_prefs,
        scope: effective_scope,
        total_count,
    }
}