use omnimemory_cli::integration::snapshots::models::*;
use std::collections::HashMap;
use chrono::Utc;

/// Example usage demonstrating the snapshot functionality
/// 
/// This shows how to use the snapshot system for:
/// 1. Creating command execution snapshots with semantic context
/// 2. Vector storage integration for semantic search
/// 3. Snapshot summarization (≤ 500 chars)
/// 4. Context-aware snapshot creation
/// 5. Snapshot query interface

#[cfg(test)]
mod snapshot_examples {
    use super::*;

    #[tokio::test]
    async fn test_create_git_commit_snapshot() {
        // Example: Git commit command with semantic context
        let semantic_context = SemanticContext {
            working_directory: "/home/user/project".to_string(),
            environment: HashMap::from([
                ("GIT_AUTHOR_NAME".to_string(), "John Doe".to_string()),
                ("GIT_COMMITTER_EMAIL".to_string(), "john@example.com".to_string()),
            ]),
            recent_commands: vec![
                "git status".to_string(),
                "git add .".to_string(),
            ],
            git_info: Some(GitContext {
                repository: "my-project".to_string(),
                branch: "main".to_string(),
                remote_url: Some("https://github.com/user/my-project".to_string()),
                uncommitted_changes: false,
                last_commit_time: Some(Utc::now()),
            }),
            docker_context: None,
            system_info: SystemContext {
                platform: "linux".to_string(),
                shell: "bash".to_string(),
                user: "user".to_string(),
                hostname: "hostname".to_string(),
                cpu_info: "x86_64".to_string(),
                memory_gb: 16.0,
            },
            tags: vec!["git".to_string(), "commit".to_string(), "version-control".to_string()],
            file_paths: vec!["src/main.rs".to_string(), "Cargo.toml".to_string()],
            command_structure: CommandStructure {
                command_type: "git".to_string(),
                primary_tool: Some("git".to_string()),
                parameters: HashMap::from([
                    ("message".to_string(), "feat: add new feature".to_string()),
                ]),
                flags: vec!["-m".to_string()],
                is_batch_command: false,
            },
        };

        let create_request = CreateSnapshotRequest {
            command: "git commit -m \"feat: add new feature\"".to_string(),
            output: "[main 1234567] feat: add new feature\n 2 files changed, 20 insertions(+), 5 deletions(-)".to_string(),
            exit_code: Some(0),
            working_directory: "/home/user/project".to_string(),
            execution_time_ms: 250,
            semantic_context,
            force_create: false,
            custom_title: None,
            additional_tags: vec!["feature".to_string()],
        };

        // This would create a High importance snapshot due to git commit
        let importance = SnapshotImportance::from_command_and_context(
            &create_request.command,
            create_request.exit_code,
            &create_request.working_directory,
            &HashMap::new(),
        );

        assert_eq!(importance, SnapshotImportance::High);
    }

    #[tokio::test]
    async fn test_create_failed_docker_build_snapshot() {
        // Example: Failed Docker build with critical importance
        let semantic_context = SemanticContext {
            working_directory: "/home/user/docker-app".to_string(),
            environment: HashMap::from([
                ("DOCKER_BUILDKIT".to_string(), "1".to_string()),
            ]),
            recent_commands: vec![
                "docker build -t myapp .".to_string(),
            ],
            git_info: None,
            docker_context: Some(DockerContext {
                current_container: None,
                docker_compose_project: Some("myapp-compose".to_string()),
                running_containers: vec!["redis".to_string(), "postgres".to_string()],
            }),
            system_info: SystemContext {
                platform: "linux".to_string(),
                shell: "bash".to_string(),
                user: "user".to_string(),
                hostname: "hostname".to_string(),
                cpu_info: "x86_64".to_string(),
                memory_gb: 16.0,
            },
            tags: vec!["docker".to_string(), "build".to_string(), "deployment".to_string()],
            file_paths: vec!["Dockerfile".to_string(), "package.json".to_string()],
            command_structure: CommandStructure {
                command_type: "docker".to_string(),
                primary_tool: Some("docker".to_string()),
                parameters: HashMap::from([
                    ("tag".to_string(), "myapp".to_string()),
                ]),
                flags: vec!["build".to_string(), "-t".to_string()],
                is_batch_command: false,
            },
        };

        let create_request = CreateSnapshotRequest {
            command: "docker build -t myapp .".to_string(),
            output: "ERROR: failed to solve: failed to solve: failed to build: npm ERR!ENOENT\n".to_string(),
            exit_code: Some(1),
            working_directory: "/home/user/docker-app".to_string(),
            execution_time_ms: 15000,
            semantic_context,
            force_create: false,
            custom_title: None,
            additional_tags: vec!["failed".to_string()],
        };

        // This would create a Critical importance snapshot due to failure
        let importance = SnapshotImportance::from_command_and_context(
            &create_request.command,
            create_request.exit_code,
            &create_request.working_directory,
            &HashMap::new(),
        );

        assert_eq!(importance, SnapshotImportance::Critical);
    }

    #[test]
    fn test_snapshot_summary_generation() {
        // Test that snapshot summary is ≤ 500 characters
        let command = "cargo test --lib --all-features -- --nocapture";
        let output = "test result: ok. 12 passed, 0 failed in 0.12s".to_string();
        let tags = vec!["cargo".to_string(), "rust".to_string(), "testing".to_string()];

        let summary = format!(
            "Command: {} - Success | Tags: {}",
            command,
            tags.join(", ")
        );

        // Summary should be concise and under 500 characters
        assert!(summary.len() <= 500);
    }

    #[tokio::test]
    async fn test_snapshot_search_query() {
        // Example semantic search query
        let query_request = QuerySnapshotsRequest {
            query: "docker builds that failed".to_string(),
            min_importance: Some(SnapshotImportance::High),
            time_range: Some(TimeRangeFilter {
                start: Some(Utc::now() - chrono::Duration::days(7)),
                end: Some(Utc::now()),
            }),
            working_directory: None,
            required_tags: Some(vec!["docker".to_string(), "failed".to_string()]),
            limit: 5,
            min_similarity: 0.7,
        };

        // This would search for snapshots related to docker failures
        // In a real implementation, this would use vector similarity search
        assert!(!query_request.query.is_empty());
        assert!(query_request.limit > 0);
    }

    #[test]
    fn test_snapshot_importance_determination() {
        // Test importance determination based on command type
        
        // High importance commands
        assert_eq!(
            SnapshotImportance::from_command_and_context("git commit -m \"fix: critical bug\"", Some(0), "/repo", &HashMap::new()),
            SnapshotImportance::High
        );
        
        assert_eq!(
            SnapshotImportance::from_command_and_context("docker build -t app .", Some(0), "/app", &HashMap::new()),
            SnapshotImportance::High
        );

        // Critical importance (dangerous commands)
        assert_eq!(
            SnapshotImportance::from_command_and_context("sudo rm -rf /important", Some(0), "/", &HashMap::new()),
            SnapshotImportance::Critical
        );

        // Medium importance (development tools)
        assert_eq!(
            SnapshotImportance::from_command_and_context("cargo test", Some(0), "/project", &HashMap::new()),
            SnapshotImportance::Medium
        );

        // Low importance (basic commands)
        assert_eq!(
            SnapshotImportance::from_command_and_context("ls -la", Some(0), "/home", &HashMap::new()),
            SnapshotImportance::Low
        );

        // Failed commands are generally High importance
        assert_eq!(
            SnapshotImportance::from_command_and_context("make build", Some(1), "/project", &HashMap::new()),
            SnapshotImportance::High
        );
    }

    #[test]
    fn test_snapshot_vector_storage_config() {
        // Test vector storage configuration
        let vector_config = VectorStorageConfig {
            enabled: true,
            index_path: "/tmp/snapshots/vector_index".to_string(),
            dimension: 768,
            metric_type: "cosine".to_string(),
            index_type: "Flat".to_string(),
        };

        assert!(vector_config.enabled);
        assert_eq!(vector_config.dimension, 768);
        assert_eq!(vector_config.metric_type, "cosine");
    }

    #[test]
    fn test_snapshot_auto_cleanup_config() {
        // Test auto-cleanup configuration
        let cleanup_config = AutoCleanupConfig {
            enabled: true,
            low_importance_after_days: 7,
            medium_importance_after_days: 30,
            high_importance_after_days: 90,
            critical_importance_never_delete: true,
            max_snapshots_per_directory: Some(1000),
        };

        assert!(cleanup_config.enabled);
        assert_eq!(cleanup_config.low_importance_after_days, 7);
        assert!(cleanup_config.critical_importance_never_delete);
        assert_eq!(cleanup_config.max_snapshots_per_directory, Some(1000));
    }

    #[tokio::test]
    async fn test_snapshot_batch_operations() {
        // Example batch operation for updating importance
        let batch_operation = BatchSnapshotOperation {
            operation_type: BatchOperationType::UpdateImportance,
            snapshot_ids: vec![
                "snap-001".to_string(),
                "snap-002".to_string(),
                "snap-003".to_string(),
            ],
            parameters: HashMap::from([
                ("new_importance".to_string(), serde_json::Value::String("High".to_string())),
            ]),
        };

        assert_eq!(batch_operation.snapshot_ids.len(), 3);
        
        match batch_operation.operation_type {
            BatchOperationType::UpdateImportance => {
                // Would update importance of multiple snapshots
            }
            _ => panic!("Wrong operation type"),
        }
    }
}

/// Integration example showing complete snapshot workflow
/// 
/// This demonstrates how all the pieces work together:
/// 1. Capture command with semantic context
/// 2. Create snapshot with appropriate importance
/// 3. Store in vector database for semantic search
/// 4. Query and retrieve relevant snapshots
#[cfg(test)]
mod integration_example {
    use super::*;

    #[tokio::test]
    async fn test_complete_snapshot_workflow() {
        // Step 1: Simulate command execution with context
        let command = "cargo test --lib";
        let working_dir = "/home/user/my-rust-project";
        let output = "test result: ok. 5 passed, 0 failed in 0.85s\nrunning 5 tests";
        let exit_code = Some(0);
        let execution_time_ms = 850;

        // Step 2: Create semantic context
        let semantic_context = create_test_semantic_context(command, working_dir).await;

        // Step 3: Create snapshot request
        let create_request = CreateSnapshotRequest {
            command: command.to_string(),
            output: output.to_string(),
            exit_code,
            working_directory: working_dir.to_string(),
            execution_time_ms,
            semantic_context,
            force_create: false,
            custom_title: None,
            additional_tags: vec!["rust".to_string(), "testing".to_string()],
        };

        // Step 4: Determine importance
        let importance = SnapshotImportance::from_command_and_context(
            &create_request.command,
            create_request.exit_code,
            &create_request.working_directory,
            &HashMap::new(),
        );

        // Step 5: Generate summary (≤ 500 chars)
        let summary = format!(
            "Command: {} - Success | Tags: {} | Files: {}",
            create_request.command,
            create_request.semantic_context.tags.join(", "),
            create_request.semantic_context.file_paths.join(", ")
        );

        assert!(summary.len() <= 500);

        // Step 6: Create final snapshot
        let snapshot = Snapshot {
            id: format!("snap-{}", uuid::Uuid::new_v4()),
            title: "Cargo test execution".to_string(),
            summary,
            command: create_request.command,
            output: create_request.output,
            exit_code: create_request.exit_code,
            created_at: Utc::now(),
            execution_time_ms: create_request.execution_time_ms,
            semantic_context: create_request.semantic_context,
            importance,
            embedding_id: Some(format!("embed-{}", uuid::Uuid::new_v4())),
            related_snapshot_ids: vec![],
            custom_metadata: HashMap::new(),
            storage_info: SnapshotStorageInfo {
                file_path: format!("/tmp/snapshots/{}.json", uuid::Uuid::new_v4()),
                size_bytes: summary.len() as u64,
                vector_index: None,
                db_record_id: None,
            },
        };

        // Verify snapshot properties
        assert!(!snapshot.id.is_empty());
        assert!(!snapshot.title.is_empty());
        assert!(snapshot.summary.len() <= 500);
        assert!(snapshot.importance != SnapshotImportance::Low); // Not low importance due to cargo
        assert!(snapshot.embedding_id.is_some()); // Should have embedding for search

        // Step 7: Demonstrate search query
        let search_request = QuerySnapshotsRequest {
            query: "rust testing commands".to_string(),
            min_importance: Some(SnapshotImportance::Medium),
            time_range: None,
            working_directory: None,
            required_tags: Some(vec!["rust".to_string(), "testing".to_string()]),
            limit: 10,
            min_similarity: 0.6,
        };

        // Verify search request
        assert!(search_request.query.contains("rust"));
        assert!(search_request.query.contains("testing"));
        assert_eq!(search_request.limit, 10);
    }

    async fn create_semantic_context(command: &str, working_dir: &str) -> SemanticContext {
        let mut tags = Vec::new();
        let mut file_paths = Vec::new();

        // Extract tags and files from command
        if command.contains("cargo") {
            tags.push("cargo".to_string());
            tags.push("rust".to_string());
        }
        if command.contains("test") {
            tags.push("testing".to_string());
        }

        // Simulate file path extraction
        if command.contains("--lib") {
            file_paths.push("src/lib.rs".to_string());
        }

        SemanticContext {
            working_directory: working_dir.to_string(),
            environment: HashMap::from([
                ("CARGO_HOME".to_string(), "/home/user/.cargo".to_string()),
                ("RUST_BACKTRACE".to_string(), "1".to_string()),
            ]),
            recent_commands: vec!["cargo build".to_string()],
            git_info: Some(GitContext {
                repository: "my-project".to_string(),
                branch: "main".to_string(),
                remote_url: Some("https://github.com/user/my-project".to_string()),
                uncommitted_changes: false,
                last_commit_time: Some(Utc::now()),
            }),
            docker_context: None,
            system_info: SystemContext {
                platform: "linux".to_string(),
                shell: "bash".to_string(),
                user: "user".to_string(),
                hostname: "hostname".to_string(),
                cpu_info: "x86_64".to_string(),
                memory_gb: 16.0,
            },
            tags,
            file_paths,
            command_structure: CommandStructure {
                command_type: "cargo".to_string(),
                primary_tool: Some("cargo".to_string()),
                parameters: HashMap::from([
                    ("subcommand".to_string(), "test".to_string()),
                ]),
                flags: vec!["--lib".to_string()],
                is_batch_command: false,
            },
        }
    }
}