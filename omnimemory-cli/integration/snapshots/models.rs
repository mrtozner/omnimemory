use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc};

/// Snapshot importance levels for context-aware creation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SnapshotImportance {
    Low,
    Medium,
    High,
    Critical,
}

impl SnapshotImportance {
    /// Determine importance based on command type and context
    pub fn from_command_and_context(command: &str, exit_code: Option<i32>, 
                                  working_dir: &str, context: &HashMap<String, String>) -> SnapshotImportance {
        // Critical: Commands with high impact or failures
        if command.starts_with("sudo ") || command.starts_with("rm ") || command.starts_with("git reset") {
            return SnapshotImportance::Critical;
        }
        
        if exit_code == Some(0) {
            // Successful commands - assess importance based on context
            if command.starts_with("git commit") || command.starts_with("docker build") {
                SnapshotImportance::High
            } else if command.starts_with("cargo") || command.starts_with("npm") || command.starts_with("pip") {
                SnapshotImportance::Medium
            } else {
                SnapshotImportance::Low
            }
        } else {
            // Failed commands are generally important
            if exit_code.unwrap_or(1) < 0 || exit_code.unwrap_or(1) > 100 {
                SnapshotImportance::Critical
            } else {
                SnapshotImportance::High
            }
        }
    }
}

/// Semantic context for snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Working directory when snapshot was created
    pub working_directory: String,
    /// Environment variables relevant to this context
    pub environment: HashMap<String, String>,
    /// Recently executed commands in the session
    pub recent_commands: Vec<String>,
    /// Current git branch and repository info
    pub git_info: Option<GitContext>,
    /// Docker container information
    pub docker_context: Option<DockerContext>,
    /// System information relevant to the snapshot
    pub system_info: SystemContext,
    /// Custom tags for semantic search
    pub tags: Vec<String>,
    /// File paths mentioned in the command/output
    pub file_paths: Vec<String>,
    /// Command arguments and their types
    pub command_structure: CommandStructure,
}

/// Git context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitContext {
    pub repository: String,
    pub branch: String,
    pub remote_url: Option<String>,
    pub uncommitted_changes: bool,
    pub last_commit_time: Option<DateTime<Utc>>,
}

/// Docker context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerContext {
    pub current_container: Option<String>,
    pub docker_compose_project: Option<String>,
    pub running_containers: Vec<String>,
}

/// System context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemContext {
    pub platform: String,
    pub shell: String,
    pub user: String,
    pub hostname: String,
    pub cpu_info: String,
    pub memory_gb: f32,
}

/// Command structure and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandStructure {
    pub command_type: String,
    pub primary_tool: Option<String>,
    pub parameters: HashMap<String, String>,
    pub flags: Vec<String>,
    pub is_batch_command: bool,
}

/// Snapshot metadata and structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Unique identifier for the snapshot
    pub id: String,
    /// Human-readable title
    pub title: String,
    /// Condensed summary (≤ 500 characters as per architecture)
    pub summary: String,
    /// Full command that was executed
    pub command: String,
    /// Command output (truncated if too large)
    pub output: String,
    /// Exit code of the command
    pub exit_code: Option<i32>,
    /// When the snapshot was created
    pub created_at: DateTime<Utc>,
    /// Execution duration
    pub execution_time_ms: u64,
    /// Semantic context
    pub semantic_context: SemanticContext,
    /// Importance level
    pub importance: SnapshotImportance,
    /// Vector embedding ID for semantic search
    pub embedding_id: Option<String>,
    /// Related snapshots (semantic similarity or workflow connections)
    pub related_snapshot_ids: Vec<String>,
    /// Custom metadata for extensibility
    pub custom_metadata: HashMap<String, serde_json::Value>,
    /// Storage location and indexing information
    pub storage_info: SnapshotStorageInfo,
}

/// Storage information for snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotStorageInfo {
    /// Path to the main snapshot file
    pub file_path: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Vector index information if applicable
    pub vector_index: Option<VectorIndexInfo>,
    /// Database record ID if stored in SQL
    pub db_record_id: Option<u64>,
}

/// Vector index information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexInfo {
    pub index_path: String,
    pub embedding_dimension: u32,
    pub index_type: String,
    pub last_updated: DateTime<Utc>,
}

/// Snapshot query and search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotQueryResult {
    /// The snapshot that matched the query
    pub snapshot: Snapshot,
    /// Semantic similarity score (0.0 to 1.0)
    pub similarity_score: f32,
    /// Reason why this snapshot was included
    pub match_reason: String,
    /// Highlighted content snippets
    pub highlighted_content: Vec<String>,
}

/// Snapshot creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSnapshotRequest {
    pub command: String,
    pub output: String,
    pub exit_code: Option<i32>,
    pub working_directory: String,
    pub execution_time_ms: u64,
    pub semantic_context: SemanticContext,
    /// Force creation even if importance is low
    pub force_create: bool,
    /// Custom title (optional, auto-generated if not provided)
    pub custom_title: Option<String>,
    /// Additional tags
    pub additional_tags: Vec<String>,
}

/// Snapshot query for semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySnapshotsRequest {
    /// Natural language query
    pub query: String,
    /// Filter by importance level
    pub min_importance: Option<SnapshotImportance>,
    /// Time range filter
    pub time_range: Option<TimeRangeFilter>,
    /// Working directory filter
    pub working_directory: Option<String>,
    /// Tag filters
    pub required_tags: Option<Vec<String>>,
    /// Maximum number of results
    pub limit: usize,
    /// Minimum similarity threshold
    pub min_similarity: f32,
}

/// Time range filter for snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRangeFilter {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

/// Snapshot management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotStats {
    pub total_snapshots: u64,
    pub by_importance: HashMap<String, u64>,
    pub by_working_directory: HashMap<String, u64>,
    pub by_command_type: HashMap<String, u64>,
    pub average_execution_time_ms: f64,
    pub storage_size_bytes: u64,
    pub oldest_snapshot: Option<DateTime<Utc>>,
    pub newest_snapshot: Option<DateTime<Utc>>,
}

/// Snapshot batch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSnapshotOperation {
    pub operation_type: BatchOperationType,
    pub snapshot_ids: Vec<String>,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchOperationType {
    /// Delete multiple snapshots
    Delete,
    /// Change importance level
    UpdateImportance,
    /// Add tags to snapshots
    AddTags,
    /// Remove tags from snapshots
    RemoveTags,
    /// Export snapshots to external format
    Export,
}

/// Snapshot export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Markdown,
    Html,
    Csv,
}

/// Snapshot configuration and settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Maximum summary length (defaults to 500)
    pub max_summary_length: usize,
    /// Maximum snapshot age before auto-deletion
    pub max_age_days: Option<u32>,
    /// Auto-importance detection enabled
    pub auto_importance: bool,
    /// Semantic search enabled
    pub semantic_search_enabled: bool,
    /// Vector storage configuration
    pub vector_storage: VectorStorageConfig,
    /// Auto-cleanup settings
    pub auto_cleanup: AutoCleanupConfig,
}

/// Vector storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStorageConfig {
    pub enabled: bool,
    pub index_path: String,
    pub dimension: u32,
    pub metric_type: String, // "cosine", "l2", "ip"
    pub index_type: String,  // "Flat", "IVFFlat", "HNSW"
}

/// Auto-cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCleanupConfig {
    pub enabled: bool,
    pub low_importance_after_days: u32,
    pub medium_importance_after_days: u32,
    pub high_importance_after_days: u32,
    pub critical_importance_never_delete: bool,
    pub max_snapshots_per_directory: Option<u32>,
}