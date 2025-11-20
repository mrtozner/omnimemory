use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tokio::sync::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use super::models::{
    Snapshot, VectorIndexInfo, SnapshotQueryResult, QuerySnapshotsRequest,
    VectorStorageConfig, VectorIndexInfo
};

/// Vector storage interface for snapshot embeddings
#[derive(Debug, Clone)]
pub struct VectorStoreInterface {
    /// Configuration for vector storage
    config: VectorStorageConfig,
    /// Path to Python script for vector operations
    python_script_path: String,
    /// Cache for embedding lookups
    embedding_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

/// Query result from Python vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonSearchResult {
    id: String,
    score: f32,
    content: String,
    metadata: serde_json::Value,
}

impl VectorStoreInterface {
    /// Create new vector store interface
    pub fn new(config: VectorStorageConfig, python_script_path: String) -> Self {
        Self {
            config,
            python_script_path,
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Initialize the vector storage
    pub async fn initialize(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Ensure the Python script exists
        let script_path = Path::new(&self.python_script_path);
        if !script_path.exists() {
            return Err(anyhow!("Vector storage Python script not found: {}", self.python_script_path));
        }

        // Initialize the vector storage via Python
        self.call_python_method("initialize", None::<()>).await
            .with_context(|| "Failed to initialize vector storage")?;

        Ok(())
    }

    /// Add a snapshot embedding to the vector store
    pub async fn add_embedding(&self, snapshot: &Snapshot) -> Result<VectorIndexInfo> {
        if !self.config.enabled || snapshot.embedding_id.is_none() {
            return Err(anyhow!("Vector storage not enabled or no embedding ID"));
        }

        let embedding_id = snapshot.embedding_id.as_ref().unwrap();
        
        // Generate embedding from snapshot content
        let content_for_embedding = self.prepare_content_for_embedding(snapshot);
        
        // Call Python to add embedding
        let result: serde_json::Value = self.call_python_method(
            "add_embedding", 
            Some(&serde_json::json!({
                "memory_id": embedding_id,
                "content": content_for_embedding,
                "snapshot_id": snapshot.id,
                "memory_type": "snapshot"
            }))
        ).await?;

        // Update cache
        let mut cache = self.embedding_cache.lock().await;
        if let Some(embedding) = result.get("embedding").and_then(|e| e.as_array()) {
            let embedding_vec: Vec<f32> = embedding.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            cache.insert(embedding_id.clone(), embedding_vec);
        }

        Ok(VectorIndexInfo {
            index_path: self.config.index_path.clone(),
            embedding_dimension: self.config.dimension,
            index_type: self.config.index_type.clone(),
            last_updated: chrono::Utc::now(),
        })
    }

    /// Search for similar snapshots
    pub async fn search_snapshots(&self, request: &QuerySnapshotsRequest) -> Result<Vec<SnapshotQueryResult>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        // Call Python vector search
        let search_params = serde_json::json!({
            "query": request.query,
            "limit": request.limit,
            "threshold": request.min_similarity,
            "memory_types": ["snapshot"]
        });

        let results: Vec<PythonSearchResult> = self.call_python_method(
            "search", 
            Some(&search_params)
        ).await?;

        // Convert Python results to our format
        let mut snapshot_results = Vec::new();
        for result in results {
            let match_reason = self.determine_match_reason(&request.query, &result.content);
            
            // Here we would reconstruct the full snapshot from storage
            // For now, we'll return a minimal result
            let query_result = SnapshotQueryResult {
                similarity_score: result.score,
                match_reason,
                highlighted_content: vec![result.content],
                snapshot: self.reconstruct_snapshot(&result.id).await?
            };
            
            snapshot_results.push(query_result);
        }

        Ok(snapshot_results)
    }

    /// Delete a snapshot embedding
    pub async fn delete_embedding(&self, embedding_id: &str) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.call_python_method(
            "delete_embedding", 
            Some(&serde_json::json!({ "memory_id": embedding_id }))
        ).await?;

        // Remove from cache
        let mut cache = self.embedding_cache.lock().await;
        cache.remove(embedding_id);

        Ok(())
    }

    /// Get vector storage statistics
    pub async fn get_stats(&self) -> Result<serde_json::Value> {
        if !self.config.enabled {
            return Ok(serde_json::json!({
                "enabled": false,
                "total_embeddings": 0
            }));
        }

        self.call_python_method("get_stats", None::<()>).await
    }

    /// Optimize the vector index
    pub async fn optimize_index(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.call_python_method("optimize_index", None::<()>).await
    }

    /// Call a Python method with parameters
    async fn call_python_method<T>(&self, method_name: &str, params: Option<&T>) -> Result<serde_json::Value>
    where
        T: Serialize,
    {
        let mut child = Command::new("python3")
            .arg(&self.python_script_path)
            .arg("--method")
            .arg(method_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn Python process for method: {}", method_name))?;

        // Send parameters as JSON if provided
        if let Some(params) = params {
            if let Some(mut stdin) = child.stdin.take() {
                let params_json = serde_json::to_string(params)?;
                stdin.write_all(params_json.as_bytes())?;
                stdin.write_all(b"\n")?;
            }
        }

        // Wait for process to complete
        let output = child.wait_with_output()
            .with_context(|| format!("Python process failed for method: {}", method_name))?;

        if !output.status.success() {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Python method '{}' failed: {}", method_name, error_msg));
        }

        // Parse JSON response
        let stdout_str = String::from_utf8(output.stdout)
            .with_context(|| format!("Invalid UTF-8 from Python method: {}", method_name))?;

        serde_json::from_str(&stdout_str)
            .with_context(|| format!("Invalid JSON response from Python method: {}", method_name))
    }

    /// Prepare snapshot content for embedding generation
    fn prepare_content_for_embedding(&self, snapshot: &Snapshot) -> String {
        let mut content = String::new();
        
        // Command
        content.push_str(&format!("Command: {}\n", snapshot.command));
        
        // Summary
        content.push_str(&format!("Summary: {}\n", snapshot.summary));
        
        // Context tags
        if !snapshot.semantic_context.tags.is_empty() {
            content.push_str(&format!("Tags: {}\n", snapshot.semantic_context.tags.join(", ")));
        }
        
        // Working directory
        content.push_str(&format!("Working Directory: {}\n", snapshot.semantic_context.working_directory));
        
        // File paths
        if !snapshot.semantic_context.file_paths.is_empty() {
            content.push_str(&format!("Files: {}\n", snapshot.semantic_context.file_paths.join(", ")));
        }
        
        // Recent commands
        if !snapshot.semantic_context.recent_commands.is_empty() {
            content.push_str(&format!("Recent Commands: {}\n", 
                snapshot.semantic_context.recent_commands.join("; ")));
        }

        content
    }

    /// Determine why a snapshot matched the query
    fn determine_match_reason(&self, query: &str, content: &str) -> String {
        let query_lower = query.to_lowercase();
        let content_lower = content.to_lowercase();
        
        if content_lower.contains(&query_lower) {
            format!("Query terms found in content")
        } else {
            format!("Semantic similarity match")
        }
    }

    /// Reconstruct a full snapshot from storage (placeholder implementation)
    async fn reconstruct_snapshot(&self, snapshot_id: &str) -> Result<Snapshot> {
        // This would typically load from persistent storage
        // For now, return a minimal snapshot
        Ok(Snapshot {
            id: snapshot_id.to_string(),
            title: "Snapshot".to_string(),
            summary: "Snapshot summary".to_string(),
            command: "unknown".to_string(),
            output: "".to_string(),
            exit_code: None,
            created_at: chrono::Utc::now(),
            execution_time_ms: 0,
            semantic_context: super::models::SemanticContext {
                working_directory: "unknown".to_string(),
                environment: HashMap::new(),
                recent_commands: vec![],
                git_info: None,
                docker_context: None,
                system_info: super::models::SystemContext {
                    platform: "unknown".to_string(),
                    shell: "unknown".to_string(),
                    user: "unknown".to_string(),
                    hostname: "unknown".to_string(),
                    cpu_info: "unknown".to_string(),
                    memory_gb: 0.0,
                },
                tags: vec![],
                file_paths: vec![],
                command_structure: super::models::CommandStructure {
                    command_type: "unknown".to_string(),
                    primary_tool: None,
                    parameters: HashMap::new(),
                    flags: vec![],
                    is_batch_command: false,
                },
            },
            importance: super::models::SnapshotImportance::Low,
            embedding_id: None,
            related_snapshot_ids: vec![],
            custom_metadata: HashMap::new(),
            storage_info: super::models::SnapshotStorageInfo {
                file_path: "unknown".to_string(),
                size_bytes: 0,
                vector_index: None,
                db_record_id: None,
            },
        })
    }
}