//! Snapshot functionality for OmniMemory CLI
//! 
//! This module provides comprehensive snapshot management including:
//! - Command execution snapshots with semantic context
//! - Vector storage integration for semantic search
//! - Snapshot summarization (≤ 500 characters)
//! - Context-aware snapshot creation based on command importance
//! - Snapshot query interface for memory retrieval

pub mod models;
pub mod manager;
pub mod vector_storage;

pub use models::*;
pub use manager::SnapshotManager;