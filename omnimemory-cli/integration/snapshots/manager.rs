use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use tokio::sync::RwLock;
use std::sync::Arc;

use super::models::*;
use super::vector_storage::VectorStoreInterface;

/// Snapshot manager for creating, storing, and retrieving snapshots
#[derive(Debug)]
pub struct SnapshotManager {
    /// Snapshots storage in memory (also persisted to disk)
    snapshots: Arc<RwLock<HashMap<String, Snapshot>>>,
    /// Vector storage for semantic search
    vector_store: Option<VectorStoreInterface>,
    /// Configuration
    config: SnapshotConfig,
    /// Base path for snapshot storage
    storage_base_path: std::path::PathBuf,
}

impl SnapshotManager {
    /// Create new snapshot manager
    pub fn new(config: SnapshotConfig, storage_base_path: std::path::PathBuf) -> Self {
        let vector_store = if config.vector_storage.enabled {
            Some(VectorStoreInterface::new(
                config.vector_storage.clone(),
                "integration/snapshots/vector_storage_interface.py".to_string(),
            ))
        } else {
            None
        };

        Self {
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            vector_store,
            config,
            storage_base_path,
        }
    }

    /// Initialize the snapshot manager
    pub async fn initialize(&self) -> Result<()> {
        // Create storage directories
        fs::create_dir_all(&self.storage_base_path)
            .with_context(|| "Failed to create snapshot storage directories")?;

        // Load existing snapshots
        self.load_snapshots().await?;

        // Initialize vector store if enabled
        if let Some(vector_store) = &self.vector_store {
            vector_store.initialize().await
                .with_context(|| "Failed to initialize vector storage")?;
        }

        Ok(())
    }

    /// Create a new snapshot
    pub async fn create_snapshot(&self, request: CreateSnapshotRequest) -> Result<Snapshot> {
        // Determine importance level
        let importance = if request.force_create {
            SnapshotImportance::Medium
        } else {
            SnapshotImportance::from_command_and_context(
                &request.command,
                request.exit_code,
                &request.working_directory,
                &HashMap::new(), // Would extract from semantic context
            )
        };

        // Skip if importance is too low and not forced
        if !request.force_create && importance == SnapshotImportance::Low {
            return Err(anyhow::anyhow!("Snapshot importance too low and force_create not specified"));
        }

        // Generate unique ID
        let id = format!("snap-{}", uuid::Uuid::new_v4());

        // Create snapshot summary (≤ 500 chars)
        let summary = self.generate_summary(&request, &importance);

        // Generate title
        let title = request.custom_title.unwrap_or_else(|| {
            self.generate_title(&request.command, request.exit_code, &importance)
        });

        // Create the snapshot
        let snapshot = Snapshot {
            id: id.clone(),
            title,
            summary,
            command: request.command.clone(),
            output: self.truncate_output(&request.output),
            exit_code: request.exit_code,
            created_at: chrono::Utc::now(),
            execution_time_ms: request.execution_time_ms,
            semantic_context: request.semantic_context,
            importance,
            embedding_id: None,
            related_snapshot_ids: vec![],
            custom_metadata: HashMap::new(),
            storage_info: SnapshotStorageInfo {
                file_path: self.storage_base_path.join(format!("{}.json", id)).to_string_lossy().to_string(),
                size_bytes: 0, // Will be calculated on save
                vector_index: None,
                db_record_id: None,
            },
        };

        // Store in memory
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(id.clone(), snapshot.clone());
        }

        // Save to disk
        self.save_snapshot(&snapshot).await?;

        // Add to vector store if enabled
        if let Some(vector_store) = &self.vector_store {
            let snapshot_with_embedding_id = self.add_embedding_id(&snapshot);
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(id.clone(), snapshot_with_embedding_id.clone());
            
            let vector_info = vector_store.add_embedding(&snapshot_with_embedding_id).await?;
            // Update storage info with vector index info
            let mut snapshots = self.snapshots.write().await;
            if let Some(existing) = snapshots.get_mut(&id) {
                existing.storage_info.vector_index = Some(vector_info);
            }
        }

        Ok(snapshot.clone())
    }

    /// Query snapshots using semantic search
    pub async fn query_snapshots(&self, request: QuerySnapshotsRequest) -> Result<Vec<SnapshotQueryResult>> {
        let mut results = Vec::new();

        // Vector search if enabled
        if let Some(vector_store) = &self.vector_store {
            let vector_results = vector_store.search_snapshots(&request).await?;
            for result in vector_results {
                // Apply additional filters
                if self.snapshot_matches_filters(&result.snapshot, &request)? {
                    results.push(result);
                }
            }
        }

        // If no vector store or no vector results, fall back to keyword search
        if results.is_empty() {
            let keyword_results = self.keyword_search(&request).await?;
            for result in keyword_results {
                results.push(result);
            }
        }

        // Sort by similarity score
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());

        // Apply limit
        results.truncate(request.limit);

        Ok(results)
    }

    /// Get snapshot by ID
    pub async fn get_snapshot(&self, id: &str) -> Option<Snapshot> {
        let snapshots = self.snapshots.read().await;
        snapshots.get(id).cloned()
    }

    /// List snapshots with optional filtering
    pub async fn list_snapshots(&self, 
                               limit: usize,
                               working_directory: Option<&str>,
                               importance: Option<SnapshotImportance>,
                               tags: Option<Vec<&str>>) -> Vec<Snapshot> {
        let snapshots = self.snapshots.read().await;
        let mut filtered: Vec<Snapshot> = snapshots.values().cloned().collect();

        // Filter by working directory
        if let Some(wd) = working_directory {
            filtered.retain(|s| s.semantic_context.working_directory == wd);
        }

        // Filter by importance
        if let Some(imp) = importance {
            filtered.retain(|s| s.importance == imp);
        }

        // Filter by tags
        if let Some(required_tags) = tags {
            filtered.retain(|s| {
                required_tags.iter().all(|tag| 
                    s.semantic_context.tags.contains(&tag.to_string())
                )
            });
        }

        // Sort by creation time (newest first)
        filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        filtered.truncate(limit);
        filtered
    }

    /// Update snapshot importance
    pub async fn update_importance(&self, id: &str, importance: SnapshotImportance) -> Result<()> {
        let mut snapshots = self.snapshots.write().await;
        if let Some(snapshot) = snapshots.get_mut(id) {
            snapshot.importance = importance;
            self.save_snapshot(snapshot).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Snapshot not found: {}", id))
        }
    }

    /// Delete snapshot
    pub async fn delete_snapshot(&self, id: &str) -> Result<()> {
        let mut snapshots = self.snapshots.write().await;
        
        if let Some(snapshot) = snapshots.remove(id) {
            // Delete from vector store
            if let (Some(vector_store), Some(embedding_id)) = (&self.vector_store, &snapshot.embedding_id) {
                vector_store.delete_embedding(embedding_id).await?;
            }

            // Delete file
            let file_path = Path::new(&snapshot.storage_info.file_path);
            if file_path.exists() {
                fs::remove_file(file_path)
                    .with_context(|| format!("Failed to delete snapshot file: {:?}", file_path))?;
            }

            Ok(())
        } else {
            Err(anyhow::anyhow!("Snapshot not found: {}", id))
        }
    }

    /// Get snapshot statistics
    pub async fn get_stats(&self) -> Result<SnapshotStats> {
        let snapshots = self.snapshots.read().await;
        
        let mut by_importance = HashMap::new();
        let mut by_working_directory = HashMap::new();
        let mut by_command_type = HashMap::new();
        let mut total_execution_time = 0u64;
        let mut storage_size = 0u64;
        let mut oldest_snapshot = None;
        let mut newest_snapshot = None;

        for snapshot in snapshots.values() {
            // Count by importance
            let importance_key = format!("{:?}", snapshot.importance);
            *by_importance.entry(importance_key).or_insert(0) += 1;

            // Count by working directory
            *by_working_directory.entry(snapshot.semantic_context.working_directory.clone()).or_insert(0) += 1;

            // Count by command type
            *by_command_type.entry(snapshot.semantic_context.command_structure.command_type.clone()).or_insert(0) += 1;

            total_execution_time += snapshot.execution_time_ms;
            storage_size += snapshot.storage_info.size_bytes;

            // Track time range
            if oldest_snapshot.is_none() || snapshot.created_at < oldest_snapshot.unwrap() {
                oldest_snapshot = Some(snapshot.created_at);
            }
            if newest_snapshot.is_none() || snapshot.created_at > newest_snapshot.unwrap() {
                newest_snapshot = Some(snapshot.created_at);
            }
        }

        let total_count = snapshots.len() as u64;
        let avg_execution_time = if total_count > 0 {
            total_execution_time as f64 / total_count as f64
        } else {
            0.0
        };

        Ok(SnapshotStats {
            total_snapshots: total_count,
            by_importance,
            by_working_directory,
            by_command_type,
            average_execution_time_ms: avg_execution_time,
            storage_size_bytes: storage_size,
            oldest_snapshot,
            newest_snapshot,
        })
    }

    /// Clean up old snapshots based on auto-cleanup config
    pub async fn cleanup_old_snapshots(&self) -> Result<u32> {
        if !self.config.auto_cleanup.enabled {
            return Ok(0);
        }

        let mut deleted_count = 0u32;
        let now = chrono::Utc::now();

        let snapshots_to_delete: Vec<String> = {
            let snapshots = self.snapshots.read().await;
            
            let mut to_delete = Vec::new();
            for (id, snapshot) in snapshots.iter() {
                let age_days = (now - snapshot.created_at).num_days() as u32;
                
                let should_delete = match snapshot.importance {
                    SnapshotImportance::Critical => !self.config.auto_cleanup.critical_importance_never_delete,
                    SnapshotImportance::High => age_days > self.config.auto_cleanup.high_importance_after_days,
                    SnapshotImportance::Medium => age_days > self.config.auto_cleanup.medium_importance_after_days,
                    SnapshotImportance::Low => age_days > self.config.auto_cleanup.low_importance_after_days,
                };

                // Also check max snapshots per directory
                let directory_limit_reached = if let Some(max_per_dir) = self.config.auto_cleanup.max_snapshots_per_directory {
                    let count_in_dir = snapshots.values()
                        .filter(|s| s.semantic_context.working_directory == snapshot.semantic_context.working_directory)
                        .count();
                    count_in_dir > max_per_dir as usize
                } else {
                    false
                };

                if should_delete || directory_limit_reached {
                    to_delete.push(id.clone());
                }
            }
            
            to_delete
        };

        // Delete snapshots
        for id in snapshots_to_delete {
            if self.delete_snapshot(&id).await.is_ok() {
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }

    // Private helper methods

    async fn load_snapshots(&self) -> Result<()> {
        let snapshots_dir = &self.storage_base_path;
        
        if !snapshots_dir.exists() {
            return Ok(());
        }

        let mut snapshots = self.snapshots.write().await;

        for entry in fs::read_dir(snapshots_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read snapshot file: {:?}", path))?;
                
                let snapshot: Snapshot = serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse snapshot: {:?}", path))?;
                
                snapshots.insert(snapshot.id.clone(), snapshot);
            }
        }

        Ok(())
    }

    async fn save_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        let file_path = Path::new(&snapshot.storage_info.file_path);
        
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize and save
        let content = serde_json::to_string_pretty(snapshot)?;
        fs::write(file_path, content)
            .with_context(|| format!("Failed to save snapshot to: {:?}", file_path))?;

        // Update size
        let mut snapshots = self.snapshots.write().await;
        if let Some(existing) = snapshots.get_mut(&snapshot.id) {
            let metadata = fs::metadata(file_path)?;
            existing.storage_info.size_bytes = metadata.len();
        }

        Ok(())
    }

    fn generate_summary(&self, request: &CreateSnapshotRequest, importance: &SnapshotImportance) -> String {
        let mut summary = String::new();
        
        // Command and result
        summary.push_str(&format!("Command: {}", request.command));
        
        if let Some(exit_code) = request.exit_code {
            if exit_code == 0 {
                summary.push_str(" - Success");
            } else {
                summary.push_str(&format!(" - Failed (exit code: {})", exit_code));
            }
        }
        
        // Add context from semantic context
        let ctx = &request.semantic_context;
        
        if !ctx.file_paths.is_empty() {
            summary.push_str(&format!(" | Files: {}", ctx.file_paths.join(", ")));
        }
        
        if !ctx.tags.is_empty() {
            summary.push_str(&format!(" | Tags: {}", ctx.tags.join(", ")));
        }
        
        // Add importance indicator
        if matches!(importance, SnapshotImportance::High | SnapshotImportance::Critical) {
            summary.push_str(" | Important");
        }

        // Ensure summary is ≤ 500 characters
        if summary.len() > 500 {
            summary.truncate(497);
            summary.push_str("...");
        }

        summary
    }

    fn generate_title(&self, command: &str, exit_code: Option<i32>, importance: &SnapshotImportance) -> String {
        let command_base = command.split_whitespace().next().unwrap_or("Unknown");
        
        match exit_code {
            Some(0) => format!("Successful {} operation", command_base),
            Some(code) => format!("Failed {} (code {})", command_base, code),
            None => {
                match importance {
                    SnapshotImportance::Critical => format!("Critical: {}", command_base),
                    SnapshotImportance::High => format!("Important: {}", command_base),
                    _ => format!("{}", command_base),
                }
            }
        }
    }

    fn truncate_output(&self, output: &str) -> String {
        // Truncate output to reasonable size (e.g., 10KB)
        let max_size = 10 * 1024;
        if output.len() > max_size {
            let truncated = &output[..max_size - 100];
            format!("{}... (truncated, {} total bytes)", truncated, output.len())
        } else {
            output.to_string()
        }
    }

    fn add_embedding_id(&self, snapshot: &Snapshot) -> Snapshot {
        let embedding_id = Some(format!("embed-{}", snapshot.id));
        let mut snapshot = snapshot.clone();
        snapshot.embedding_id = embedding_id;
        snapshot
    }

    fn snapshot_matches_filters(&self, snapshot: &Snapshot, request: &QuerySnapshotsRequest) -> Result<bool> {
        // Check importance filter
        if let Some(min_importance) = &request.min_importance {
            if !self.importance_meets_threshold(snapshot.importance.clone(), min_importance.clone()) {
                return Ok(false);
            }
        }

        // Check time range filter
        if let Some(time_range) = &request.time_range {
            if let Some(start) = time_range.start {
                if snapshot.created_at < start {
                    return Ok(false);
                }
            }
            if let Some(end) = time_range.end {
                if snapshot.created_at > end {
                    return Ok(false);
                }
            }
        }

        // Check working directory filter
        if let Some(required_wd) = &request.working_directory {
            if snapshot.semantic_context.working_directory != *required_wd {
                return Ok(false);
            }
        }

        // Check tags filter
        if let Some(required_tags) = &request.required_tags {
            for tag in required_tags {
                if !snapshot.semantic_context.tags.contains(tag) {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    fn importance_meets_threshold(&self, snapshot_importance: SnapshotImportance, threshold: SnapshotImportance) -> bool {
        use SnapshotImportance::*;
        
        match (snapshot_importance, threshold) {
            (Critical, _) => true,
            (High, High | Medium | Low) => true,
            (Medium, Medium | Low) => true,
            (Low, Low) => true,
            (_, Critical) => false,
            (High, Low) => false,
            (Medium, Low) => false,
            (Low, Critical | High | Medium) => false,
        }
    }

    async fn keyword_search(&self, request: &QuerySnapshotsRequest) -> Result<Vec<SnapshotQueryResult>> {
        let mut results = Vec::new();
        let query_lower = request.query.to_lowercase();
        
        let snapshots = self.snapshots.read().await;
        
        for snapshot in snapshots.values() {
            let mut score = 0.0f32;
            let mut match_reasons = Vec::new();

            // Check command
            if snapshot.command.to_lowercase().contains(&query_lower) {
                score += 0.9;
                match_reasons.push("Command match".to_string());
            }

            // Check summary
            if snapshot.summary.to_lowercase().contains(&query_lower) {
                score += 0.7;
                match_reasons.push("Summary match".to_string());
            }

            // Check tags
            for tag in &snapshot.semantic_context.tags {
                if tag.to_lowercase().contains(&query_lower) {
                    score += 0.8;
                    match_reasons.push(format!("Tag: {}", tag));
                    break;
                }
            }

            // Check file paths
            for file_path in &snapshot.semantic_context.file_paths {
                if file_path.to_lowercase().contains(&query_lower) {
                    score += 0.6;
                    match_reasons.push(format!("File: {}", file_path));
                    break;
                }
            }

            if score >= request.min_similarity && self.snapshot_matches_filters(snapshot, request)? {
                results.push(SnapshotQueryResult {
                    snapshot: snapshot.clone(),
                    similarity_score: score,
                    match_reason: match_reasons.join("; "),
                    highlighted_content: vec![snapshot.summary.clone()],
                });
            }
        }

        Ok(results)
    }
}