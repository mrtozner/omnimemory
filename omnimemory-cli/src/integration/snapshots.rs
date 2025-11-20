//! Snapshot models and types for OmniMemory CLI
//!
//! This module contains all types related to command execution snapshots,
//! including their metadata, semantic context, and query results.

pub mod models {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::str::FromStr;

    /// Result of a snapshot query operation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SnapshotQueryResult {
        /// The matching snapshot
        pub snapshot: Snapshot,
        /// Similarity score (0.0-1.0)
        pub similarity_score: f32,
        /// Reason why this snapshot matched the query
        pub match_reason: String,
        /// Highlighted content snippets showing the match
        pub highlighted_content: Vec<String>,
    }

    /// A command execution snapshot
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Snapshot {
        /// Unique identifier for the snapshot
        pub id: String,
        /// Human-readable title
        pub title: String,
        /// Summary of the command execution
        pub summary: String,
        /// The command that was executed
        pub command: String,
        /// The command output
        pub output: String,
        /// Exit code of the command
        pub exit_code: Option<i32>,
        /// When the snapshot was created
        pub created_at: DateTime<Utc>,
        /// How long the command took to execute (milliseconds)
        pub execution_time_ms: u64,
        /// Semantic context of the command execution
        pub semantic_context: SemanticContext,
        /// Importance level of this snapshot
        pub importance: SnapshotImportance,
        /// ID of the embedding vector (if any)
        pub embedding_id: Option<String>,
        /// IDs of related snapshots
        pub related_snapshot_ids: Vec<String>,
        /// Custom metadata as key-value pairs
        pub custom_metadata: HashMap<String, serde_json::Value>,
        /// Storage information
        pub storage_info: SnapshotStorageInfo,
    }

    /// Importance level of a snapshot
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum SnapshotImportance {
        Low,
        Medium,
        High,
        Critical,
    }

    impl SnapshotImportance {
        /// Determine importance from command and context
        pub fn from_command_and_context(
            command: &str,
            exit_code: Option<i32>,
            _working_directory: &str,
            _custom_metadata: &HashMap<String, serde_json::Value>,
        ) -> Self {
            // Simple heuristic for now
            // In a real implementation, this would use ML or more sophisticated rules

            // Failed commands are more important
            if let Some(code) = exit_code {
                if code != 0 {
                    return SnapshotImportance::High;
                }
            }

            // Critical commands
            if command.contains("rm -rf")
                || command.contains("shutdown")
                || command.contains("reboot")
                || command.starts_with("sudo rm")
                || command.contains("DROP DATABASE")
                || command.contains("DROP TABLE") {
                return SnapshotImportance::Critical;
            }

            // Important commands
            if command.starts_with("git commit")
                || command.starts_with("git push")
                || command.starts_with("docker build")
                || command.starts_with("cargo build --release")
                || command.starts_with("npm publish")
                || command.contains("deploy") {
                return SnapshotImportance::High;
            }

            // Medium importance commands
            if command.starts_with("git")
                || command.starts_with("docker")
                || command.starts_with("cargo")
                || command.starts_with("npm")
                || command.starts_with("yarn")
                || command.starts_with("pip") {
                return SnapshotImportance::Medium;
            }

            // Default to low
            SnapshotImportance::Low
        }
    }

    impl FromStr for SnapshotImportance {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s.to_lowercase().as_str() {
                "low" => Ok(SnapshotImportance::Low),
                "medium" => Ok(SnapshotImportance::Medium),
                "high" => Ok(SnapshotImportance::High),
                "critical" => Ok(SnapshotImportance::Critical),
                _ => Err(format!("Invalid importance level: {}", s)),
            }
        }
    }

    /// Storage information for a snapshot
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SnapshotStorageInfo {
        /// Path to the snapshot file on disk
        pub file_path: String,
        /// Size of the snapshot in bytes
        pub size_bytes: u64,
        /// Vector index location (if indexed)
        pub vector_index: Option<String>,
        /// Database record ID (if stored in DB)
        pub db_record_id: Option<String>,
    }

    /// Semantic context of a command execution
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SemanticContext {
        /// Working directory when command was executed
        pub working_directory: String,
        /// Environment variables at execution time
        pub environment: HashMap<String, String>,
        /// Recent commands before this one
        pub recent_commands: Vec<String>,
        /// Git repository information (if applicable)
        pub git_info: Option<GitInfo>,
        /// Docker context information (if applicable)
        pub docker_context: Option<DockerContext>,
        /// System information
        pub system_info: SystemContext,
        /// Tags associated with this snapshot
        pub tags: Vec<String>,
        /// File paths referenced in the command
        pub file_paths: Vec<String>,
        /// Parsed command structure
        pub command_structure: CommandStructure,
    }

    /// Git repository information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GitInfo {
        /// Current branch
        pub branch: String,
        /// Latest commit hash
        pub commit_hash: String,
        /// Commit message
        pub commit_message: String,
        /// Whether there are uncommitted changes
        pub has_changes: bool,
        /// Remote repository URL
        pub remote_url: Option<String>,
    }

    /// Docker context information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DockerContext {
        /// Active containers
        pub active_containers: Vec<String>,
        /// Images used
        pub images: Vec<String>,
        /// Network mode
        pub network_mode: Option<String>,
        /// Volumes mounted
        pub volumes: Vec<String>,
    }

    /// System context information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SystemContext {
        /// Operating system platform
        pub platform: String,
        /// Shell being used
        pub shell: String,
        /// Current user
        pub user: String,
        /// Hostname
        pub hostname: String,
        /// CPU information
        pub cpu_info: String,
        /// Memory in GB
        pub memory_gb: f64,
    }

    /// Parsed command structure
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CommandStructure {
        /// Type of command (e.g., "git", "docker", "cargo")
        pub command_type: String,
        /// Primary tool being used
        pub primary_tool: Option<String>,
        /// Command parameters as key-value pairs
        pub parameters: HashMap<String, String>,
        /// Command flags
        pub flags: Vec<String>,
        /// Whether this is a batch command (contains && or ||)
        pub is_batch_command: bool,
    }

    /// Request to create a new snapshot
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CreateSnapshotRequest {
        /// The command that was executed
        pub command: String,
        /// The command output
        pub output: String,
        /// Exit code of the command
        pub exit_code: Option<i32>,
        /// Working directory at execution time
        pub working_directory: String,
        /// Execution time in milliseconds
        pub execution_time_ms: u64,
        /// Semantic context
        pub semantic_context: SemanticContext,
        /// Force creation even for low-importance commands
        pub force_create: bool,
        /// Custom title for the snapshot
        pub custom_title: Option<String>,
        /// Additional tags to add
        pub additional_tags: Vec<String>,
    }

    /// Request to query snapshots
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct QuerySnapshotsRequest {
        /// Natural language query
        pub query: String,
        /// Minimum importance level
        pub min_importance: Option<SnapshotImportance>,
        /// Time range filter
        pub time_range: Option<TimeRangeFilter>,
        /// Working directory filter
        pub working_directory: Option<String>,
        /// Required tags (all must be present)
        pub required_tags: Option<Vec<String>>,
        /// Maximum number of results
        pub limit: usize,
        /// Minimum similarity score (0.0-1.0)
        pub min_similarity: f32,
    }

    /// Time range filter for queries
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeRangeFilter {
        /// Start time (inclusive)
        pub start: Option<DateTime<Utc>>,
        /// End time (inclusive)
        pub end: Option<DateTime<Utc>>,
    }

    /// Statistics about snapshots
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SnapshotStats {
        /// Total number of snapshots
        pub total_snapshots: usize,
        /// Count by importance level
        pub by_importance: HashMap<String, usize>,
        /// Count by working directory
        pub by_working_directory: HashMap<String, usize>,
        /// Count by command type
        pub by_command_type: HashMap<String, usize>,
        /// Average execution time in milliseconds
        pub average_execution_time_ms: f64,
        /// Total storage size in bytes
        pub storage_size_bytes: u64,
        /// Oldest snapshot timestamp
        pub oldest_snapshot: Option<DateTime<Utc>>,
        /// Newest snapshot timestamp
        pub newest_snapshot: Option<DateTime<Utc>>,
    }
}
