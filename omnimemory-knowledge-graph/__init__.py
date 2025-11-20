"""
OmniMemory Knowledge Graph Service

Phase 2 Semantic Intelligence for building and querying file relationships.
Phase 3 Workflow Checkpoint Service for cross-session continuity.
"""

from .knowledge_graph_service import KnowledgeGraphService
from .workflow_checkpoint_service import WorkflowCheckpointService

__version__ = "0.2.0"
__all__ = ["KnowledgeGraphService", "WorkflowCheckpointService"]
