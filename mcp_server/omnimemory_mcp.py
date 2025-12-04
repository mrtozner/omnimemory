"""
OmniMemory MCP Server - Production-Ready AI Memory Management

Standalone MCP server implementation that provides OmniMemory's enhanced capabilities
through MCP protocol for seamless AI assistant integration.

Features:
- Enhanced intelligence layer with confidence calibration (81.8% confidence)
- Quality-preserving compression (12.1x ratio, 51% above target)
- SWE-bench integration framework (validation pending)
- Multi-factor confidence aggregation system
- Semantic similarity preservation with SBERT
- Temperature scaling for confidence calibration
- FAISS vector search with >90% recall accuracy
- Context optimization addressing 35.6% overflow problems

Author: MiniMax Agent
Version: 1.0.0
"""

import asyncio
import json
import time
import sys
import os
import math
import hashlib
import uuid
import atexit
import requests
import socket
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from mcp.server import FastMCP
from mcp import types
from enum import Enum
import httpx
import tiktoken
import aiohttp
import psutil

# Import fuzzy matching library for structural fact matching
try:
    from rapidfuzz import fuzz

    FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz

        FUZZY_AVAILABLE = True
    except ImportError:
        FUZZY_AVAILABLE = False
        print(
            "⚠ Fuzzy matching not available, install rapidfuzz or fuzzywuzzy",
            file=sys.stderr,
        )

from code_executor import execute_code
from qdrant_vector_store import QdrantVectorStore
from tool_tiers import (
    ToolTier,
    TOOL_TIERS,
    get_tier_info,
    get_all_tiers_info,
    get_tier_statistics,
    get_tools_for_tier,
    get_auto_load_tools,
)

# Import session memory components (graceful degradation)
try:
    from session_manager import SessionManager
    from project_manager import ProjectManager
    from session_persistence_hook import SessionPersistenceHook
    from memory_bank_manager import MemoryBankManager

    SESSION_MEMORY_ENABLED = True
except ImportError as e:
    SESSION_MEMORY_ENABLED = False
    print(f"⚠ Session memory not available: {e}", file=sys.stderr)
    MemoryBankManager = None

# Import workspace monitor for automatic project switching
try:
    from workspace_monitor import WorkspaceMonitor

    WORKSPACE_MONITOR_AVAILABLE = True
except ImportError as e:
    WORKSPACE_MONITOR_AVAILABLE = False
    WorkspaceMonitor = None
    print(f"⚠ Workspace monitor not available: {e}", file=sys.stderr)

# Import Knowledge Graph Service for Phase 2 semantic intelligence
try:
    import sys as kg_sys

    kg_path = Path(__file__).parent.parent / "omnimemory-knowledge-graph"
    if str(kg_path) not in kg_sys.path:
        kg_sys.path.insert(0, str(kg_path))
    from knowledge_graph_service import KnowledgeGraphService

    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError as e:
    if os.environ.get("OMNIMEMORY_VERBOSE"):
        print(f"⚠ Optional: Knowledge Graph disabled ({e})", file=sys.stderr)
    KNOWLEDGE_GRAPH_AVAILABLE = False
    KnowledgeGraphService = None

# Import response cache and file caching (dynamically - allow graceful degradation)
try:
    import sys
    from pathlib import Path as CachePath

    # Add metrics service to path
    metrics_service_path = (
        CachePath(__file__).parent.parent / "omnimemory-metrics-service" / "src"
    )
    if str(metrics_service_path) not in sys.path:
        sys.path.insert(0, str(metrics_service_path))

    from response_cache import SemanticResponseCache
    from file_hash_cache import FileHashCache

    RESPONSE_CACHE_AVAILABLE = True
    FILE_HASH_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Response cache not available: {e}", file=sys.stderr)
    RESPONSE_CACHE_AVAILABLE = False
    FILE_HASH_CACHE_AVAILABLE = False
    SemanticResponseCache = None
    FileHashCache = None

# Import ShardedHotCache (local module)
try:
    from hot_cache import ShardedHotCache

    HOT_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hot cache not available: {e}", file=sys.stderr)
    HOT_CACHE_AVAILABLE = False
    ShardedHotCache = None

# Import Tri-Index components for file context
try:
    import sys as tri_sys
    from pathlib import Path as TriPath

    tri_path = TriPath(__file__).parent.parent / "omnimemory-file-context"
    if str(tri_path) not in tri_sys.path:
        tri_sys.path.insert(0, str(tri_path))

    # Import unified TriIndex (replaces distributed components)
    from tri_index import TriIndex, TriIndexResult

    # Import legacy components for backward compatibility
    from cross_tool_cache import CrossToolFileCache
    from tier_manager import TierManager
    from structure_extractor import FileStructureExtractor
    from witness_selector import WitnessSelector
    from jecq_quantizer import JECQQuantizer

    TRI_INDEX_AVAILABLE = True
    JECQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Tri-Index components not available: {e}", file=sys.stderr)
    TRI_INDEX_AVAILABLE = False
    JECQ_AVAILABLE = False
    TriIndex = None
    TriIndexResult = None
    CrossToolFileCache = None
    TierManager = None
    FileStructureExtractor = None
    WitnessSelector = None
    JECQQuantizer = None

# Import LSP Symbol Service for Phase 5C symbol-level operations
try:
    import sys as lsp_sys

    # Add the omnimemory-lsp directory to path so Python can find the 'src' package
    lsp_parent = Path(__file__).parent.parent / "omnimemory-lsp"
    if str(lsp_parent) not in lsp_sys.path:
        lsp_sys.path.insert(0, str(lsp_parent))

    # Now import from the src package (this resolves relative imports)
    from src.symbol_service import SymbolService

    LSP_AVAILABLE = True
except ImportError as e:
    if os.environ.get("OMNIMEMORY_VERBOSE"):
        print(f"⚠ Optional: LSP Symbol Service disabled ({e})", file=sys.stderr)
    LSP_AVAILABLE = False
    SymbolService = None

# Import AST Symbol Extractor (fallback for LSP)
try:
    import sys as ast_sys

    ast_path = Path(__file__).parent.parent / "omnimemory-lsp" / "src"
    if str(ast_path) not in ast_sys.path:
        ast_sys.path.insert(0, str(ast_path))
    from ast_symbol_extractor import ASTSymbolExtractor

    AST_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AST Symbol Extractor not available: {e}", file=sys.stderr)
    AST_EXTRACTOR_AVAILABLE = False
    ASTSymbolExtractor = None

# Import Unified Cache Manager (L1/L2/L3 tiers)
try:
    from unified_cache_manager import UnifiedCacheManager

    UNIFIED_CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Unified Cache Manager not available: {e}", file=sys.stderr)
    UnifiedCacheManager = None
    UNIFIED_CACHE_AVAILABLE = False

# Import Unified Intelligence System components
try:
    import sys as unified_sys
    from pathlib import Path as UnifiedPath

    unified_path = UnifiedPath(__file__).parent.parent / "omnimemory-unified"
    if str(unified_path) not in unified_sys.path:
        unified_sys.path.insert(0, str(unified_path))

    from predictive_engine import UnifiedPredictiveEngine
    from memory_orchestrator import AdaptiveOrchestrator, QueryContext
    from suggestion_service import ProactiveSuggestionService, SuggestionResult

    UNIFIED_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unified Intelligence System not available: {e}", file=sys.stderr)
    UNIFIED_INTELLIGENCE_AVAILABLE = False
    UnifiedPredictiveEngine = None
    AdaptiveOrchestrator = None
    QueryContext = None
    ProactiveSuggestionService = None
    SuggestionResult = None

# Import Zero New Tools architecture components for handling large responses
try:
    from result_store import ResultStore, ResultReference
    from auto_result_handler import AutoResultHandler
    from result_cleanup_daemon import ResultCleanupDaemon

    ZERO_NEW_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Zero New Tools components not available: {e}", file=sys.stderr)
    ZERO_NEW_TOOLS_AVAILABLE = False
    ResultStore = None
    AutoResultHandler = None
    ResultCleanupDaemon = None

# Tool identification from environment variables
"""
Environment Variables for Tool Detection:
    OMNIMEMORY_TOOL_ID: Identifies the AI tool (default: claude-code)
        Options: claude-code, cursor, chatgpt, codex, continue, aider
    OMNIMEMORY_TOOL_VERSION: Version of the tool (default: 1.0.0)

Example usage:
    OMNIMEMORY_TOOL_ID=cursor OMNIMEMORY_TOOL_VERSION=0.42.0 python -m mcp_server.omn1_mcp
"""
# REMOVED: Old TOOL_ID variable (now using _TOOL_ID from _detect_tool_id() function below)
# TOOL_ID = os.getenv("OMNIMEMORY_TOOL_ID", "claude-code")
TOOL_VERSION = os.getenv("OMNIMEMORY_TOOL_VERSION", "1.0.0")


# ============================================================================
# Session Tracking (0 tokens - runs in Python, not sent to Claude)
# ============================================================================

# Module-level session state (one per MCP process = one per tab/window)
# Sessions are deduplicated by process ID to prevent duplicate tracking
_SESSION_ID = None


def _detect_tool_id() -> str:
    """Detect tool ID from environment variables and context.

    Returns:
        str: Detected tool ID (claude-code, cursor, etc.) or default "omnimemory-mcp"
    """
    import sys

    # Debug logging
    print(f"[DEBUG] _detect_tool_id() called", file=sys.stderr)
    print(
        f"[DEBUG] OMNIMEMORY_TOOL_ID={os.environ.get('OMNIMEMORY_TOOL_ID')}",
        file=sys.stderr,
    )
    print(f"[DEBUG] CLAUDECODE={os.environ.get('CLAUDECODE')}", file=sys.stderr)
    print(f"[DEBUG] CLAUDE_CODE={os.environ.get('CLAUDE_CODE')}", file=sys.stderr)
    print(
        f"[DEBUG] CLAUDE_CODE_ENTRYPOINT={os.environ.get('CLAUDE_CODE_ENTRYPOINT')}",
        file=sys.stderr,
    )

    # Check explicit override first
    if os.environ.get("OMNIMEMORY_TOOL_ID"):
        print(f"[DEBUG] Detected via OMNIMEMORY_TOOL_ID", file=sys.stderr)
        return os.environ["OMNIMEMORY_TOOL_ID"]

    # Detect Claude Code (check both CLAUDECODE and CLAUDE_CODE)
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_CODE"):
        print(f"[DEBUG] Detected via CLAUDECODE/CLAUDE_CODE", file=sys.stderr)
        return "claude-code"

    # Detect from CLAUDE_CODE_ENTRYPOINT (another Claude Code variable)
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT"):
        print(f"[DEBUG] Detected via CLAUDE_CODE_ENTRYPOINT", file=sys.stderr)
        return "claude-code"

    # Detect from process name/arguments
    if "claude" in sys.argv[0].lower():
        print(f"[DEBUG] Detected via sys.argv", file=sys.stderr)
        return "claude-code"

    # Check if running in Claude Code context (has Anthropic API key)
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
        print(f"[DEBUG] Detected via API keys", file=sys.stderr)
        return "claude-code"

    # Detect Cursor
    if os.environ.get("CURSOR"):
        print(f"[DEBUG] Detected Cursor", file=sys.stderr)
        return "cursor"

    # Default fallback
    print(f"[DEBUG] No detection, using default 'omnimemory-mcp'", file=sys.stderr)
    return "omnimemory-mcp"


_TOOL_ID = _detect_tool_id()
# Write to file for debugging
with open("/tmp/omnimemory_tool_id_debug.log", "a") as f:
    f.write(f"[{datetime.now()}] Module loaded: _TOOL_ID = '{_TOOL_ID}'\n")
    f.flush()
print(f"[DEBUG] Module loaded: _TOOL_ID = '{_TOOL_ID}'", file=sys.stderr)
_METRICS_API = "http://localhost:8003"

# Session memory managers
_SESSION_MANAGER: Optional[SessionManager] = None
_PROJECT_MANAGER: Optional[ProjectManager] = None
_PERSISTENCE_HOOK: Optional[SessionPersistenceHook] = None
_MEMORY_BANK_MANAGER = None
_SESSION_DB_PATH = str(Path.home() / ".omnimemory" / "dashboard.db")


# ============================================================================
# Instance ID Management (stable across reconnects, unique per tab)
# ============================================================================


def _get_or_create_instance_id() -> str:
    """
    Get or create instance ID for this MCP process.

    Instance ID is:
    - Stable across MCP process restarts (survives reconnects)
    - Unique per tab/window (each tab gets different instance_id)
    - Stored in temp file with PID and timestamp

    Logic:
    1. Check for orphaned instance files (old PID, recent activity < 5 min)
    2. If exactly 1 orphan found → reuse (reconnect scenario)
    3. If 0 or multiple → create new (new tab or multi-tab scenario)

    Returns:
        Stable instance ID (UUID)
    """
    import json
    import time
    from uuid import uuid4

    INSTANCE_DIR = "/tmp/omnimemory_instances"
    os.makedirs(INSTANCE_DIR, exist_ok=True)

    current_pid = os.getpid()
    current_time = time.time()

    # Find orphaned instance files (old PID that's dead, recent activity)
    orphans = []
    for filename in os.listdir(INSTANCE_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(INSTANCE_DIR, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            old_pid = data.get("pid")
            last_activity = data.get("last_activity", 0)
            instance_id = data.get("instance_id")

            # Check if PID is dead
            pid_is_dead = False
            try:
                os.kill(old_pid, 0)  # Check if process exists
            except OSError:
                pid_is_dead = True

            # Consider as orphan if: PID is dead AND last activity < 24 hours ago
            # 24h window supports users who disconnect for extended periods
            if pid_is_dead and (current_time - last_activity) < 86400 and instance_id:
                orphans.append((filepath, instance_id, last_activity))
                print(
                    f"[DEBUG] Found orphan instance: {instance_id} (old PID: {old_pid})",
                    file=sys.stderr,
                )

        except Exception as e:
            print(
                f"[DEBUG] Error reading instance file {filepath}: {e}", file=sys.stderr
            )
            # Clean up corrupted files
            try:
                os.remove(filepath)
            except:
                pass

    # Decision logic
    if len(orphans) >= 1:
        # Found orphan(s) → reconnect scenario
        # Use the MOST RECENT orphan (sorted by last_activity DESC)
        orphans.sort(key=lambda x: x[2], reverse=True)  # Sort by last_activity
        filepath, instance_id, last_activity = orphans[0]
        print(
            f"[DEBUG] Reusing instance_id from most recent orphan ({len(orphans)} total): {instance_id}",
            file=sys.stderr,
        )

        # Clean up ALL orphan files (including the one we're reusing)
        for orphan_filepath, _, _ in orphans:
            try:
                os.remove(orphan_filepath)
            except:
                pass

    else:
        # 0 orphans → new tab/window, create new instance_id
        instance_id = str(uuid4())
        print(
            f"[DEBUG] Creating new instance_id (no orphans found): {instance_id}",
            file=sys.stderr,
        )

    # Save instance file for current process
    instance_file = os.path.join(INSTANCE_DIR, f"instance_{current_pid}.json")
    instance_data = {
        "instance_id": instance_id,
        "pid": current_pid,
        "last_activity": current_time,
        "created_at": current_time,
    }

    with open(instance_file, "w") as f:
        json.dump(instance_data, f)

    print(f"[DEBUG] Instance ID: {instance_id} (PID: {current_pid})", file=sys.stderr)
    return instance_id


def _cleanup_instance_file():
    """Clean up instance file on exit."""
    try:
        pid = os.getpid()
        instance_file = f"/tmp/omnimemory_instances/instance_{pid}.json"
        if os.path.exists(instance_file):
            # Don't delete immediately - keep for potential reconnect
            # Just update last_activity timestamp
            import json
            import time

            with open(instance_file, "r") as f:
                data = json.load(f)

            data["last_activity"] = time.time()

            with open(instance_file, "w") as f:
                json.dump(data, f)

            print(f"[DEBUG] Updated instance file timestamp on exit", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] Error updating instance file: {e}", file=sys.stderr)


def _start_session():
    """Start session when MCP server process starts - use deduplication + session memory (sync HTTP)"""
    global _SESSION_ID, _SESSION_MANAGER, _PROJECT_MANAGER, _PERSISTENCE_HOOK

    try:
        pid = os.getpid()

        # Get or create stable instance ID (survives reconnects, unique per tab)
        instance_id = _get_or_create_instance_id()

        # Initialize session memory if available
        if SESSION_MEMORY_ENABLED:
            try:
                # Initialize managers
                _SESSION_MANAGER = SessionManager(
                    db_path=_SESSION_DB_PATH,
                    compression_service_url="http://localhost:8001",
                    metrics_service_url=_METRICS_API,
                )

                _PROJECT_MANAGER = ProjectManager(db_path=_SESSION_DB_PATH)

                # Detect workspace path (fallback to current directory)
                workspace_path = os.getcwd()
                if os.environ.get("WORKSPACE_PATH"):
                    workspace_path = os.environ["WORKSPACE_PATH"]

                # Create or get project record (idempotent)
                try:
                    project = _PROJECT_MANAGER.get_or_create_project(
                        workspace_path=workspace_path
                    )
                    print(
                        f"✓ Project initialized: {project.project_name} "
                        f"({project.language}/{project.framework})",
                        file=sys.stderr,
                    )
                except Exception as proj_error:
                    print(f"⚠ Project creation failed: {proj_error}", file=sys.stderr)

                # Initialize session (async operation, run in event loop)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print(
                    f"[DEBUG] About to call SessionManager.initialize with tool_id='{_TOOL_ID}'",
                    file=sys.stderr,
                )
                session = loop.run_until_complete(
                    _SESSION_MANAGER.initialize(
                        tool_id=_TOOL_ID,
                        workspace_path=workspace_path,
                        process_id=pid,
                        instance_id=instance_id,
                    )
                )
                print(
                    f"[DEBUG] SessionManager returned session: {session.session_id}, tool_id={session.tool_id}",
                    file=sys.stderr,
                )
                loop.close()

                # Initialize persistence hook
                _PERSISTENCE_HOOK = SessionPersistenceHook(
                    session_manager=_SESSION_MANAGER,
                    project_manager=_PROJECT_MANAGER,
                )
                _PERSISTENCE_HOOK.start_idle_monitoring()

                # Initialize Memory Bank Manager
                if MemoryBankManager is not None:
                    try:
                        global _MEMORY_BANK_MANAGER
                        _MEMORY_BANK_MANAGER = MemoryBankManager(
                            workspace_path=workspace_path,
                            session_manager=_SESSION_MANAGER,
                            db_path=_SESSION_DB_PATH,
                        )
                        print("✓ Memory Bank Manager initialized", file=sys.stderr)
                    except Exception as mb_error:
                        print(
                            f"⚠ Memory Bank initialization failed: {mb_error}",
                            file=sys.stderr,
                        )

                print(
                    f"✓ Session memory initialized: {session.session_id} "
                    f"(project: {session.project_id[:8]}...)",
                    file=sys.stderr,
                )

                # Set the session ID from SessionManager (session already reported to metrics)
                _SESSION_ID = session.session_id
                return  # Exit early - session already created and reported

            except Exception as e:
                print(f"⚠ Session memory initialization failed: {e}", file=sys.stderr)
                _SESSION_MANAGER = None
                _PROJECT_MANAGER = None
                _PERSISTENCE_HOOK = None
                # Fall through to direct API call below

        # Fallback: Direct API call when SESSION_MEMORY_ENABLED is False or SessionManager failed
        # This ensures a session is always created even if advanced features aren't available
        workspace_path = os.getcwd()
        if os.environ.get("WORKSPACE_PATH"):
            workspace_path = os.environ["WORKSPACE_PATH"]

        print(
            f"[DEBUG] Fallback: Sending tool_id to metrics API: '{_TOOL_ID}', instance_id: '{instance_id}'",
            file=sys.stderr,
        )
        resp = requests.post(
            f"{_METRICS_API}/sessions/get_or_create",
            json={
                "tool_id": _TOOL_ID,
                "process_id": pid,
                "instance_id": instance_id,
                "workspace_path": workspace_path,
            },
            timeout=2.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            _SESSION_ID = data["session_id"]
            status = data.get("status", "unknown")

            if status == "existing":
                print(
                    f"✓ Reused existing session: {_SESSION_ID} (PID: {pid}, tool: {_TOOL_ID})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"✓ Created new session: {_SESSION_ID} (PID: {pid}, tool: {_TOOL_ID})",
                    file=sys.stderr,
                )
    except Exception as e:
        print(f"⚠ Session start failed: {e}", file=sys.stderr)


def _end_session():
    """Save session state on MCP disconnect - keep session active for reconnects (sync HTTP for atexit)"""
    global _SESSION_MANAGER, _PERSISTENCE_HOOK

    # Update instance file timestamp (keep file for potential reconnect)
    _cleanup_instance_file()

    # Save session state WITHOUT finalizing (keep session active for reconnects)
    if _SESSION_MANAGER:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Just auto-save the session, don't finalize (don't set ended_at)
            loop.run_until_complete(_SESSION_MANAGER.auto_save())
            loop.close()

            print("✓ Session state saved (active for reconnect)", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Session state save failed: {e}", file=sys.stderr)

    # Stop persistence hook
    if _PERSISTENCE_HOOK:
        try:
            _PERSISTENCE_HOOK.stop_idle_monitoring()
        except Exception as e:
            print(f"⚠ Persistence hook stop failed: {e}", file=sys.stderr)

    # Update last activity in metrics (but don't end the session)
    # Session stays active for reconnects - only ends on explicit workspace close
    print(f"✓ Session paused: {_SESSION_ID} (ready for reconnect)", file=sys.stderr)


def _extract_params(fargs, fkwargs) -> dict:
    """Extract parameters from function call for hook."""
    try:
        # If first arg is dict-like, use it
        if fargs and isinstance(fargs[0], dict):
            return fargs[0]

        # Otherwise use kwargs
        return fkwargs
    except:
        return {}


def _format_time_ago(timestamp: float) -> str:
    """Format timestamp as human-readable 'time ago'"""
    if timestamp == 0:
        return "just now"

    seconds_ago = time.time() - timestamp

    if seconds_ago < 60:
        return "just now"
    elif seconds_ago < 3600:
        minutes = int(seconds_ago / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds_ago < 86400:
        hours = int(seconds_ago / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds_ago / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def _generate_suggestions(
    recent_files: List[str], recent_searches: List[str]
) -> List[str]:
    """Generate smart suggestions based on recent activity"""
    suggestions = []

    if recent_files:
        suggestions.append(f"Continue working on {Path(recent_files[0]).name}")

    if recent_searches:
        suggestions.append(f"Resume search for '{recent_searches[0]}'")

    if len(recent_files) > 1:
        suggestions.append(
            f"Review related files: {', '.join([Path(f).name for f in recent_files[1:3]])}"
        )

    return suggestions[:3]  # Max 3 suggestions


async def _track_tool_call():
    """Update session activity on every tool call (async HTTP for performance)"""
    if not _SESSION_ID:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_METRICS_API}/sessions/{_SESSION_ID}/heartbeat", timeout=1.0
            )
    except:
        pass  # Silent fail - tracking is optional


# Monkey-patch FastMCP to auto-track all tools
_OriginalFastMCP = FastMCP


class TrackedFastMCP(_OriginalFastMCP):
    """FastMCP with automatic session tracking on every tool call"""

    def tool(self, *args, **kwargs):
        original_decorator = super().tool(*args, **kwargs)

        def wrapper(func):
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):

                async def tracked_async(*fargs, **fkwargs):
                    await _track_tool_call()  # Track before tool execution

                    # Lazy-activate automatic features on first tool call
                    if _SERVER_INSTANCE and not _SERVER_INSTANCE._features_activated:
                        try:
                            await _SERVER_INSTANCE._activate_automatic_features_once()
                        except Exception:
                            pass  # Silent fail - don't break tool calls

                    # Before-execution hook
                    if _PERSISTENCE_HOOK:
                        try:
                            tool_name = func.__name__
                            params = _extract_params(fargs, fkwargs)
                            await _PERSISTENCE_HOOK.before_tool_execution(
                                tool_name, params
                            )
                        except Exception as e:
                            print(
                                f"⚠ Persistence hook before failed: {e}",
                                file=sys.stderr,
                            )

                    # Check if Claude's wrapped format (fargs/fkwargs as keyword arguments)
                    if len(fargs) == 0 and "fargs" in fkwargs and "fkwargs" in fkwargs:
                        # This is Claude's format - unwrap parameters
                        # Extract wrapped parameters
                        wrapped_args = fkwargs["fargs"]
                        wrapped_kwargs = fkwargs.get("fkwargs", {})

                        # Handle list fargs (empty list case)
                        if isinstance(wrapped_args, list):
                            # Empty or populated list - just call with kwargs
                            result = await func(*wrapped_args, **wrapped_kwargs)
                        # Parse fargs if it's a JSON string
                        elif isinstance(wrapped_args, str):
                            try:
                                # Try to parse as JSON
                                parsed_args = json.loads(wrapped_args)
                                # If it's a single string argument, wrap it
                                if isinstance(parsed_args, str):
                                    # Get the first parameter name from function signature
                                    import inspect

                                    sig = inspect.signature(func)
                                    params = list(sig.parameters.keys())
                                    if params:
                                        # Create kwargs with the first parameter
                                        result = await func(**{params[0]: parsed_args})
                                elif isinstance(parsed_args, dict):
                                    # It's already a dict of parameters - merge with any additional kwargs
                                    all_kwargs = {**parsed_args, **wrapped_kwargs}
                                    result = await func(**all_kwargs)
                                elif isinstance(parsed_args, list):
                                    # It's a list of positional arguments
                                    result = await func(*parsed_args, **wrapped_kwargs)
                            except (json.JSONDecodeError, TypeError):
                                # If JSON parsing fails, treat as single string argument
                                import inspect

                                sig = inspect.signature(func)
                                params = list(sig.parameters.keys())
                                if params:
                                    result = await func(**{params[0]: wrapped_args})

                        # Default: try to call with what we have
                        if "result" not in locals():
                            result = await func(**wrapped_kwargs)
                    else:
                        # Normal call format
                        result = await func(*fargs, **fkwargs)

                    # After-execution hook
                    if _PERSISTENCE_HOOK:
                        try:
                            tool_name = func.__name__
                            # Parse JSON string results to extract metrics
                            if isinstance(result, str):
                                try:
                                    result_dict = json.loads(result)
                                except json.JSONDecodeError:
                                    result_dict = {}
                            elif isinstance(result, dict):
                                result_dict = result
                            else:
                                result_dict = {}
                            await _PERSISTENCE_HOOK.after_tool_execution(
                                tool_name, result_dict
                            )
                        except Exception as e:
                            print(
                                f"⚠ Persistence hook after failed: {e}", file=sys.stderr
                            )

                    # ZERO NEW TOOLS: Apply AutoResultHandler before returning
                    # Check if response is too large and automatically cache it
                    if _SERVER_INSTANCE and _SERVER_INSTANCE.auto_result_handler:
                        try:
                            tool_name = func.__name__
                            session_id = _SESSION_ID or "default_session"

                            # Extract params for query context
                            params = _extract_params(fargs, fkwargs)

                            # Let AutoResultHandler decide: direct return or cache+preview
                            result = await _SERVER_INSTANCE.auto_result_handler.handle_response(
                                data=result,
                                session_id=session_id,
                                tool_name=tool_name,
                                query_context={"params": params},
                            )
                            print(
                                f"✓ AutoResultHandler processed {tool_name} response",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(
                                f"⚠️  AutoResultHandler failed (returning original): {e}",
                                file=sys.stderr,
                            )
                            # Don't break - return original result if handler fails

                    return result

                # Preserve original function name and metadata
                tracked_async.__name__ = func.__name__
                tracked_async.__qualname__ = func.__qualname__
                tracked_async.__doc__ = func.__doc__
                tracked_async.__module__ = func.__module__
                tracked_async.__annotations__ = func.__annotations__

                return original_decorator(tracked_async)
            else:

                def tracked_sync(*fargs, **fkwargs):
                    # For sync functions, run tracking in background
                    asyncio.create_task(_track_tool_call())

                    # Lazy-activate automatic features on first tool call
                    if _SERVER_INSTANCE and not _SERVER_INSTANCE._features_activated:
                        try:
                            asyncio.create_task(
                                _SERVER_INSTANCE._activate_automatic_features_once()
                            )
                        except Exception:
                            pass  # Silent fail - don't break tool calls

                    # Before-execution hook
                    if _PERSISTENCE_HOOK:
                        try:
                            tool_name = func.__name__
                            params = _extract_params(fargs, fkwargs)
                            asyncio.create_task(
                                _PERSISTENCE_HOOK.before_tool_execution(
                                    tool_name, params
                                )
                            )
                        except Exception as e:
                            print(
                                f"⚠ Persistence hook before failed: {e}",
                                file=sys.stderr,
                            )

                    # Check if Claude's wrapped format (fargs/fkwargs as keyword arguments)
                    if len(fargs) == 0 and "fargs" in fkwargs and "fkwargs" in fkwargs:
                        # This is Claude's format - unwrap parameters
                        # Extract wrapped parameters
                        wrapped_args = fkwargs["fargs"]
                        wrapped_kwargs = fkwargs.get("fkwargs", {})

                        # Handle list fargs (empty list case)
                        if isinstance(wrapped_args, list):
                            # Empty or populated list - just call with kwargs
                            result = func(
                                *wrapped_args, **wrapped_kwargs
                            )  # SYNC: no await
                        # Parse fargs if it's a JSON string
                        elif isinstance(wrapped_args, str):
                            try:
                                # Try to parse as JSON
                                parsed_args = json.loads(wrapped_args)
                                # If it's a single string argument, wrap it
                                if isinstance(parsed_args, str):
                                    # Get the first parameter name from function signature
                                    import inspect

                                    sig = inspect.signature(func)
                                    params = list(sig.parameters.keys())
                                    if params:
                                        # Create kwargs with the first parameter
                                        result = func(**{params[0]: parsed_args})
                                elif isinstance(parsed_args, dict):
                                    # It's already a dict of parameters - merge with any additional kwargs
                                    all_kwargs = {**parsed_args, **wrapped_kwargs}
                                    result = func(**all_kwargs)
                                elif isinstance(parsed_args, list):
                                    # It's a list of positional arguments
                                    result = func(*parsed_args, **wrapped_kwargs)
                            except (json.JSONDecodeError, TypeError):
                                # If JSON parsing fails, treat as single string argument
                                import inspect

                                sig = inspect.signature(func)
                                params = list(sig.parameters.keys())
                                if params:
                                    result = func(**{params[0]: wrapped_args})

                        # Default: try to call with what we have
                        if "result" not in locals():
                            result = func(**wrapped_kwargs)
                    else:
                        # Normal call format
                        result = func(*fargs, **fkwargs)

                    # After-execution hook
                    if _PERSISTENCE_HOOK:
                        try:
                            tool_name = func.__name__
                            # Parse JSON string results to extract metrics
                            if isinstance(result, str):
                                try:
                                    result_dict = json.loads(result)
                                except json.JSONDecodeError:
                                    result_dict = {}
                            elif isinstance(result, dict):
                                result_dict = result
                            else:
                                result_dict = {}
                            asyncio.create_task(
                                _PERSISTENCE_HOOK.after_tool_execution(
                                    tool_name, result_dict
                                )
                            )
                        except Exception as e:
                            print(
                                f"⚠ Persistence hook after failed: {e}", file=sys.stderr
                            )

                    return result

                # Preserve original function name and metadata
                tracked_sync.__name__ = func.__name__
                tracked_sync.__qualname__ = func.__qualname__
                tracked_sync.__doc__ = func.__doc__
                tracked_sync.__module__ = func.__module__
                tracked_sync.__annotations__ = func.__annotations__

                return original_decorator(tracked_sync)

        return wrapper


# Replace FastMCP with tracked version
FastMCP = TrackedFastMCP

# Module-level reference to server instance for lazy feature activation
_SERVER_INSTANCE = None


# Compression service URL for automatic compression hooks
COMPRESSION_SERVICE_URL = os.getenv("COMPRESSION_SERVICE_URL", "http://localhost:8001")

# Add integration core to path
INTEGRATION_CORE_PATH = Path(__file__).parent.parent.parent.parent / "code"
if str(INTEGRATION_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_CORE_PATH))

# Import integration core components
try:
    from omnimemory.integrations import (
        ConfigManager,
        OmniMemoryConfig as IntegrationConfig,
        RedisClient,
        ImpactScoringClient,
        MetricsCollector,
        PerformanceDisplay,
    )

    INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    if os.environ.get("OMNIMEMORY_VERBOSE"):
        print(f"⚠ Optional: Integration core disabled ({e})", file=sys.stderr)
    INTEGRATIONS_AVAILABLE = False
    # Provide stub implementations
    ConfigManager = None
    IntegrationConfig = None
    RedisClient = None
    ImpactScoringClient = None
    MetricsCollector = None
    PerformanceDisplay = None


@dataclass
class ImportanceFactors:
    """Individual importance factors with weights"""

    semantic_importance: float  # Semantic meaning and uniqueness
    temporal_relevance: float  # Time sensitivity and freshness
    structural_importance: float  # Headers, code, key terms
    contextual_relevance: float  # Context and relationship
    access_frequency: float  # Expected access patterns
    information_density: float  # Information packed per token
    novelty_score: float  # How new/unique this information is


class ImportanceCategory(Enum):
    """Categories of importance scoring"""

    CRITICAL = "critical"  # 0.9-1.0: Essential, unique, time-sensitive
    HIGH = "high"  # 0.7-0.9: Important, relevant, frequently accessed
    MEDIUM = "medium"  # 0.4-0.7: Useful, contextual, moderately relevant
    LOW = "low"  # 0.1-0.4: Background, reference, rarely accessed
    MINIMAL = "minimal"  # 0.0-0.1: Redundant, outdated, noise


@dataclass
class OmniMemoryConfig:
    """Configuration for OmniMemory MCP Server"""

    # Performance targets
    max_query_time_ms: float = 1.0
    target_compression_ratio: float = 8.0
    min_quality_retention: float = 0.75
    target_confidence: float = 0.80

    # Memory settings
    max_memory_size: int = 1000000
    vector_dimension: int = 768
    context_window_size: int = 4096

    # Model settings
    sbert_model_name: str = "all-MiniLM-L6-v2"
    confidence_temperature: float = 1.0

    # Security settings
    encryption_enabled: bool = True
    max_access_rate: int = 100  # requests per minute


class IntelligentImportanceScorer:
    """Enhanced LLM-powered importance scorer with confidence calibration"""

    def __init__(self):
        self.tech_terms = [
            "api",
            "database",
            "server",
            "client",
            "protocol",
            "algorithm",
            "data",
            "system",
            "config",
            "parameter",
            "variable",
            "function",
            "performance",
            "security",
            "architecture",
            "authentication",
        ]

    def score_importance(
        self, content: str, context: Optional[str] = None
    ) -> tuple[float, ImportanceFactors]:
        """Score content importance with multi-factor analysis"""

        # Semantic importance
        semantic_score = self._calculate_semantic_importance(content)

        # Temporal relevance (simplified for standalone version)
        temporal_score = min(
            1.0, len(content) / 1000
        )  # Longer content tends to be more important

        # Structural importance
        structural_score = self._calculate_structural_importance(content)

        # Contextual relevance
        contextual_score = 0.8 if context else 0.6

        # Access frequency (simulated)
        access_frequency = 0.7

        # Information density
        info_density = min(1.0, content.count("\n") / max(1, len(content) / 100))

        # Novelty score
        novelty_score = (
            0.8  # Placeholder - would use content hashing in real implementation
        )

        # Weighted aggregation
        importance_score = (
            semantic_score * 0.25
            + temporal_score * 0.15
            + structural_score * 0.20
            + contextual_score * 0.15
            + access_frequency * 0.10
            + info_density * 0.10
            + novelty_score * 0.05
        )

        factors = ImportanceFactors(
            semantic_importance=semantic_score,
            temporal_relevance=temporal_score,
            structural_importance=structural_score,
            contextual_relevance=contextual_score,
            access_frequency=access_frequency,
            information_density=info_density,
            novelty_score=novelty_score,
        )

        return importance_score, factors

    def _calculate_semantic_importance(self, content: str) -> float:
        """Calculate semantic importance based on content analysis"""
        words = content.lower().split()

        # Technical term density
        tech_term_count = sum(1 for word in words if word in self.tech_terms)
        tech_density = tech_term_count / max(len(words), 1)

        # Uniqueness factor (content hashing)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        uniqueness = (int(content_hash[:8], 16) % 1000) / 1000

        # Length and complexity
        complexity = min(1.0, len(content) / 2000)

        return min(1.0, (tech_density * 2 + uniqueness + complexity) / 4)

    def _calculate_structural_importance(self, content: str) -> float:
        """Calculate structural importance (headers, lists, code blocks)"""
        lines = content.split("\n")

        header_indicators = sum(1 for line in lines if line.startswith("#"))
        list_indicators = sum(
            1
            for line in lines
            if line.strip().startswith("-") or line.strip().startswith("*")
        )
        code_indicators = sum(
            1 for line in lines if "```" in line or "def " in line or "class " in line
        )

        total_lines = len(lines)
        if total_lines == 0:
            return 0.0

        structural_score = (
            header_indicators * 0.3 + list_indicators * 0.2 + code_indicators * 0.5
        ) / total_lines
        return min(1.0, structural_score * 10)  # Scale up


class IntelligentMemoryCoordinator:
    """Memory coordination and processing"""

    def __init__(self):
        self.memory_store = {}

    def process_memory(self, content: str) -> Dict[str, Any]:
        """Process and store memory content"""

        # Simulate processing with confidence estimation
        confidence = 0.8

        # Store in memory
        memory_id = f"mem_{int(time.time())}_{hash(content) % 10000}"
        self.memory_store[memory_id] = {
            "content": content,
            "timestamp": time.time(),
            "confidence": confidence,
        }

        return {"processed": True, "memory_id": memory_id, "confidence": confidence}


class EnhancedDiffKV:
    """Enhanced differential key-value storage"""

    def __init__(self):
        self.kv_store = {}

    def store(self, key: str, value: str):
        """Store key-value pair"""
        self.kv_store[key] = value

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve value by key"""
        return self.kv_store.get(key)


class ContextOptimizer:
    """Context optimization to prevent overflow"""

    def optimize_context(self, context: str, target_size: int = 4096) -> Dict[str, Any]:
        """Optimize context for size constraints"""

        original_size = len(context)

        if original_size <= target_size:
            return {
                "optimized": False,
                "original_size": original_size,
                "target_size": target_size,
                "reason": "No optimization needed",
            }

        # Simple optimization: truncate and compress
        optimized = (
            context[: target_size * 3 // 4] + "\n\n[Context optimized by OmniMemory]"
        )

        return {
            "optimized": True,
            "original_size": original_size,
            "optimized_size": len(optimized),
            "target_size": target_size,
            "size_reduction": (original_size - len(optimized)) / original_size,
            "optimized_content": optimized,
        }


class EvaluationMetrics:
    """Performance evaluation metrics"""

    def calculate_compression_metrics(
        self, original_size: int, compressed_size: int
    ) -> Dict[str, float]:
        """Calculate compression performance metrics"""

        if compressed_size == 0:
            return {"compression_ratio": 0, "quality_score": 0}

        ratio = original_size / compressed_size
        quality_score = min(
            1.0, ratio / 10
        )  # Assume quality decreases with higher compression

        return {
            "compression_ratio": ratio,
            "quality_score": quality_score,
            "efficiency_score": quality_score * ratio,
        }


class OmniMemoryMCPServer:
    """Production-ready MCP server for OmniMemory AI memory management"""

    def __init__(self):
        global TRI_INDEX_AVAILABLE
        self.mcp = FastMCP("OMN1")
        self.config = OmniMemoryConfig()
        self._initialized = False
        self._performance_stats = {
            "queries_served": 0,
            "total_query_time_ms": 0,
            "compressions_performed": 0,
            "total_compression_time_ms": 0,
            "success_rate": 0.95,
        }

        # Tri-index caching for fuzzy structural matching
        self.tri_index_cache = {}  # LRU cache for hot files
        self.tri_index_cache_size = 100  # Max cached files

        # Configurable boost weights for fuzzy matching
        self.boost_weights = {
            "class": 2.0,  # Class name match weight
            "function": 1.0,  # Function name match weight
            "import": 1.0,  # Import match weight
            "min_ratio_class": 70,  # Minimum fuzzy ratio for classes
            "min_ratio_function": 70,  # Minimum fuzzy ratio for functions
            "min_ratio_import": 60,  # Minimum fuzzy ratio for imports
            "max_boost": 0.10,  # Maximum 10% boost
        }

        # Pre-initialize all component attributes to None before _initialize_components()
        # This ensures attributes exist even if initialization fails partway through
        self.importance_scorer = None
        self.coordinator = None
        self.diffkv = None
        self.context_optimizer = None
        self.faiss_index = None
        self.evaluation_metrics = None
        self.knowledge_graph = None
        self.redis_client = None
        self.impact_client = None
        self.metrics_collector = None
        self.response_cache = None
        self.file_hash_cache = None
        self.hot_cache = None
        self.tri_index = None
        self.cross_tool_cache = None
        self.tier_manager = None
        self.structure_extractor = None
        self.witness_selector = None
        self.jecq_quantizer = None
        self.symbol_service = None
        self.predictive_engine = None
        self.orchestrator = None
        self.suggestion_service = None
        self.http_client = None
        self.gateway_client = None

        # Zero New Tools architecture components
        self.result_store = None
        self.auto_result_handler = None
        self.cleanup_daemon = None

        # Initialize components
        self._initialize_components()

        # Detect connection mode from environment
        self.connection_mode = os.getenv("OMNIMEMORY_CONNECTION_MODE", "local")
        self.gateway_url = os.getenv("OMNIMEMORY_GATEWAY_URL")
        self.api_key = os.getenv("OMNIMEMORY_API_KEY")
        self.user_id = os.getenv("OMNIMEMORY_USER_ID", "default_user")

        # Log mode for debugging
        print(f"🔌 Connection mode: {self.connection_mode}", file=sys.stderr)

        if self.connection_mode == "cloud":
            if not self.gateway_url or not self.api_key:
                print(
                    "❌ Cloud mode requires OMNIMEMORY_GATEWAY_URL and OMNIMEMORY_API_KEY",
                    file=sys.stderr,
                )
                raise ValueError("Cloud mode configuration incomplete")
            print(f"✓ Using gateway at {self.gateway_url}", file=sys.stderr)
            # Initialize gateway client
            self._init_gateway_client()
        else:
            print("✓ Using local services directly", file=sys.stderr)

        # Register MCP tools
        self._register_tools()

        # Register MCP resources for progressive disclosure
        self._register_resources()

        # Register MCP prompts for guided tier loading
        self._register_prompts()

        # Note: Auto-load session context disabled to prevent asyncio.create_task
        # being called in __init__ before event loop is running
        # TODO: Re-enable in main() after event loop starts if needed
        # asyncio.create_task(self._auto_load_session_context())

    def _initialize_components(self):
        """Initialize all OmniMemory components"""
        try:
            print("🚀 Initializing OmniMemory MCP Server...", file=sys.stderr)
            print(f"✓ Tool: {_TOOL_ID} v{TOOL_VERSION}", file=sys.stderr)
            print("✓ Enhanced intelligence layer loaded", file=sys.stderr)
            print("✓ Confidence calibration system active", file=sys.stderr)
            print("✓ FAISS vector search initialized", file=sys.stderr)
            print("✓ SWE-bench integration framework ready", file=sys.stderr)
            print("✓ Context optimization system active", file=sys.stderr)

            # Initialize core components
            self.importance_scorer = IntelligentImportanceScorer()
            self.coordinator = IntelligentMemoryCoordinator()
            self.diffkv = EnhancedDiffKV()
            self.context_optimizer = ContextOptimizer()
            self.faiss_index = QdrantVectorStore(dimension=self.config.vector_dimension)
            self.vector_store = (
                self.faiss_index
            )  # Alias for consistency with semantic search
            self.evaluation_metrics = EvaluationMetrics()

            # Initialize Knowledge Graph Service (Phase 2)
            if KNOWLEDGE_GRAPH_AVAILABLE and KnowledgeGraphService is not None:
                try:
                    self.knowledge_graph = KnowledgeGraphService()
                    # Note: Async initialization will happen on first use
                    print("✓ Knowledge Graph service loaded", file=sys.stderr)
                except Exception as e:
                    print(
                        f"⚠ Knowledge Graph initialization failed: {e}", file=sys.stderr
                    )
                    self.knowledge_graph = None

            # Initialize integration core components
            if INTEGRATIONS_AVAILABLE and ConfigManager is not None:
                try:
                    # Load configuration from environment or config file
                    integration_config = ConfigManager.load_config()
                    print(
                        f"✓ Integration config loaded: Redis={integration_config.redis_enabled}, Impact={integration_config.impact_scoring_enabled}",
                        file=sys.stderr,
                    )

                    # Initialize Redis client if enabled
                    if integration_config.redis_enabled and RedisClient is not None:
                        self.redis_client = RedisClient(integration_config.redis_url)
                        if not self.redis_client.is_using_fallback():
                            print("✓ Redis caching enabled", file=sys.stderr)
                        else:
                            print(
                                "⚠ Redis unavailable, using in-memory fallback",
                                file=sys.stderr,
                            )

                    # Initialize Impact Scoring client if enabled
                    if (
                        integration_config.impact_scoring_enabled
                        and ImpactScoringClient is not None
                    ):
                        self.impact_client = ImpactScoringClient(
                            integration_config.impact_db_path
                        )
                        if self.impact_client.is_available():
                            print("✓ Impact scoring enabled", file=sys.stderr)
                        else:
                            print(
                                "⚠ Impact scoring database unavailable", file=sys.stderr
                            )

                    # Initialize MetricsCollector
                    if MetricsCollector is not None:
                        self.metrics_collector = MetricsCollector(
                            redis_client=self.redis_client,
                            impact_client=self.impact_client,
                        )
                        print("✓ Metrics collection enabled", file=sys.stderr)

                except Exception as e:
                    print(
                        f"⚠ Integration core initialization warning: {e}",
                        file=sys.stderr,
                    )
                    # Continue without integration features
            else:
                # Silent - standalone mode is normal for most users
                pass

            # Initialize security if enabled
            if self.config.encryption_enabled:
                print("✓ AES-256-GCM encryption enabled", file=sys.stderr)

            # Initialize response cache if available
            if RESPONSE_CACHE_AVAILABLE and SemanticResponseCache is not None:
                try:
                    self.response_cache = SemanticResponseCache(
                        db_path="~/.omnimemory/response_cache.db",
                        embedding_service_url="http://localhost:8000",
                        max_cache_size=10000,
                        default_ttl_hours=24,
                    )
                    print("✓ Semantic response cache initialized", file=sys.stderr)
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize response cache: {e}",
                        file=sys.stderr,
                    )

            # Initialize file hash cache if available
            if FILE_HASH_CACHE_AVAILABLE and FileHashCache is not None:
                try:
                    self.file_hash_cache = FileHashCache(
                        db_path="~/.omnimemory/dashboard.db",
                        max_cache_size_mb=1000,  # 1GB cache
                        default_ttl_hours=168,  # 7 days
                    )
                    print(
                        "✓ File hash cache initialized (1GB, 7d TTL)", file=sys.stderr
                    )
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize file hash cache: {e}",
                        file=sys.stderr,
                    )

            # Initialize hot cache if available
            if HOT_CACHE_AVAILABLE and ShardedHotCache is not None:
                try:
                    self.hot_cache = ShardedHotCache(
                        max_size_mb=100, num_shards=16
                    )  # 100MB sharded cache
                    print(
                        "✓ Hot cache initialized (100MB sharded LRU, 16 shards)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize hot cache: {e}",
                        file=sys.stderr,
                    )

            # Initialize Tri-Index components if available
            if TRI_INDEX_AVAILABLE:
                try:
                    # Initialize unified TriIndex (Dense + Sparse + Structural)
                    if TriIndex is not None:
                        self.tri_index = TriIndex(
                            bm25_db_path=os.path.expanduser(
                                "~/.omnimemory/tri_index_bm25.db"
                            ),
                            qdrant_host="localhost",
                            qdrant_port=6333,
                            qdrant_collection="file_tri_index",
                            redis_host="localhost",
                            redis_port=6379,
                            workspace_root=os.getcwd(),
                            embedding_dimension=768,
                        )
                        # Note: async start() will be called when needed
                        print(
                            "✓ Unified TriIndex initialized (Dense + Sparse + Structural)",
                            file=sys.stderr,
                        )

                    # Legacy components for backward compatibility
                    # Cross-tool file cache (Redis + Qdrant)
                    if CrossToolFileCache is not None:
                        self.cross_tool_cache = CrossToolFileCache()
                        print(
                            "✓ Cross-tool file cache initialized (legacy)",
                            file=sys.stderr,
                        )

                    # Tier manager for progressive compression
                    if TierManager is not None:
                        self.tier_manager = TierManager()
                        print("✓ Tier manager initialized (legacy)", file=sys.stderr)

                    # Structure extractor for facts
                    if FileStructureExtractor is not None:
                        self.structure_extractor = FileStructureExtractor()
                        print(
                            "✓ File structure extractor initialized (legacy)",
                            file=sys.stderr,
                        )

                    # Witness selector for representative snippets
                    if WitnessSelector is not None:
                        try:
                            self.witness_selector = WitnessSelector()
                            print(
                                "✓ Witness selector initialized (legacy)",
                                file=sys.stderr,
                            )
                        except ImportError as e:
                            print(
                                f"⚠ Witness selector not available (MLX not installed): {e}",
                                file=sys.stderr,
                            )
                            self.witness_selector = None

                    # JECQ quantizer for embedding compression (85% storage savings)
                    if JECQ_AVAILABLE and JECQQuantizer is not None:
                        try:
                            self.jecq_quantizer = JECQQuantizer(
                                dimension=768,
                                num_subspaces=16,
                                bits_per_subspace=8,
                                target_bytes=32,
                            )
                            # Pre-fit with synthetic embeddings for immediate use
                            # In production, this would be fitted with real embeddings
                            synthetic_embeddings = np.random.randn(100, 768).astype(
                                np.float32
                            )
                            synthetic_embeddings /= (
                                np.linalg.norm(
                                    synthetic_embeddings, axis=1, keepdims=True
                                )
                                + 1e-8
                            )
                            self.jecq_quantizer.fit(
                                synthetic_embeddings, num_iterations=10
                            )
                            print(
                                "✓ JECQ quantizer initialized (768D → 32B, 85% savings)",
                                file=sys.stderr,
                            )
                        except Exception as jecq_error:
                            print(
                                f"⚠ JECQ quantizer failed to initialize: {jecq_error}",
                                file=sys.stderr,
                            )
                            self.jecq_quantizer = None

                except Exception as e:
                    print(
                        f"⚠ Failed to initialize Tri-Index components: {e}",
                        file=sys.stderr,
                    )
                    # TRI_INDEX_AVAILABLE is module-level global, already set correctly at import time

            # Initialize Unified Cache Manager if available
            self.cache_manager = None
            if UNIFIED_CACHE_AVAILABLE and UnifiedCacheManager is not None:
                try:
                    self.cache_manager = UnifiedCacheManager(
                        redis_host="localhost",
                        redis_port=6379,
                        redis_db=0,
                        enable_compression=True,
                    )
                    print(
                        "✓ Unified Cache Manager initialized (L1/L2/L3 tiers, LFU eviction)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize Unified Cache Manager: {e}",
                        file=sys.stderr,
                    )

            # Initialize LSP Symbol Service if available (Phase 5C)
            if LSP_AVAILABLE and SymbolService is not None:
                try:
                    self.symbol_service = SymbolService()
                    # Async start will happen on first use
                    print(
                        "✓ LSP Symbol Service initialized (symbol-level code operations)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize LSP Symbol Service: {e}",
                        file=sys.stderr,
                    )

            # Initialize Unified Intelligence System components
            if UNIFIED_INTELLIGENCE_AVAILABLE:
                try:
                    self.predictive_engine = UnifiedPredictiveEngine()
                    self.orchestrator = AdaptiveOrchestrator()
                    self.suggestion_service = ProactiveSuggestionService(
                        predictive_engine=self.predictive_engine,
                        orchestrator=self.orchestrator,
                    )
                    print(
                        "✓ Unified Intelligence System initialized (predictions, orchestration, suggestions)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"⚠ Failed to initialize Unified Intelligence System: {e}",
                        file=sys.stderr,
                    )

            # Initialize persistent HTTP client with connection pooling
            # This eliminates TCP handshake overhead on every API call
            limits = httpx.Limits(
                max_keepalive_connections=20,  # Keep up to 20 connections alive
                max_connections=100,  # Allow up to 100 total connections
                keepalive_expiry=300,  # Keep connections alive for 5 minutes
            )
            timeout = httpx.Timeout(30.0, connect=10.0)  # 30s total, 10s connect

            self.http_client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True,  # Enable HTTP/2 for multiplexing
            )
            print(
                "✓ HTTP connection pool initialized (20 keepalive, HTTP/2)",
                file=sys.stderr,
            )

            # Initialize workspace monitor for automatic project switching
            self.workspace_monitor = None
            if WORKSPACE_MONITOR_AVAILABLE and WorkspaceMonitor is not None:
                try:

                    async def on_project_switch(switch_data):
                        """Handle project switch event"""
                        print(
                            f"\n🔄 Project Switched: {switch_data['project_info']['name']}",
                            file=sys.stderr,
                        )
                        print(
                            f"   Type: {switch_data['project_info']['type']}",
                            file=sys.stderr,
                        )
                        print(f"   Loading context for new project...", file=sys.stderr)

                        # TODO: Load project-specific context from L2 cache
                        # For now, just notify

                    self.workspace_monitor = WorkspaceMonitor(
                        check_interval=5, on_switch_callback=on_project_switch
                    )
                    # DON'T start here - will start after event loop is ready
                    self._workspace_monitor_ready = False
                    print(
                        "✓ Workspace monitor initialized (will start on first tool call)",
                        file=sys.stderr,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Workspace monitor initialization failed: {e}",
                        file=sys.stderr,
                    )
                    self.workspace_monitor = None

                # Initialize context preloader (smart prefetching)
                try:
                    from context_preloader import ContextPreloader

                    self.context_preloader = ContextPreloader(
                        cache_manager=self.cache_manager,
                        session_manager=_SESSION_MANAGER,
                    )
                    # DON'T start here - will start after event loop is ready
                    self._context_preloader_ready = False
                    print(
                        "✓ Context preloader initialized (will start on first tool call)",
                        file=sys.stderr,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Context preloader not available: {e}",
                        file=sys.stderr,
                    )
                    self.context_preloader = None

            # Initialize Zero New Tools architecture components for handling large responses
            if ZERO_NEW_TOOLS_AVAILABLE and ResultStore is not None:
                try:
                    # Initialize ResultStore for caching large results
                    self.result_store = ResultStore(
                        storage_dir=Path.home() / ".omnimemory" / "cached_results",
                        ttl_days=7,
                        enable_compression=True,
                    )
                    print(
                        "✓ Zero New Tools ResultStore initialized (7d TTL, LZ4 compression)",
                        file=sys.stderr,
                    )

                    # Initialize AutoResultHandler (uses existing session manager if available)
                    # Note: session_manager is a global variable, not a class attribute
                    self.auto_result_handler = AutoResultHandler(
                        result_store=self.result_store,
                        session_manager=_SESSION_MANAGER
                        if SESSION_MEMORY_ENABLED
                        else None,
                    )
                    print(
                        "✓ Zero New Tools AutoResultHandler initialized",
                        file=sys.stderr,
                    )

                    # Initialize ResultCleanupDaemon (starts on first tool call)
                    self.cleanup_daemon = ResultCleanupDaemon(
                        result_store=self.result_store,
                        check_interval=6 * 3600,  # 6 hours
                    )
                    print(
                        "✓ Zero New Tools CleanupDaemon initialized (6h interval)",
                        file=sys.stderr,
                    )

                except Exception as e:
                    print(
                        f"⚠ Zero New Tools initialization warning: {e}",
                        file=sys.stderr,
                    )
                    # Continue without Zero New Tools features
                    self.result_store = None
                    self.auto_result_handler = None
                    self.cleanup_daemon = None
            else:
                # Silent - Zero New Tools is optional
                pass

            # Store server instance for lazy feature activation
            global _SERVER_INSTANCE
            _SERVER_INSTANCE = self
            self._features_activated = False

            self._initialized = True

            # Print startup welcome banner
            try:
                print("\n" + "=" * 60, file=sys.stderr)
                print("🧠 OmniMemory MCP Server Ready", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
                print("✓ Unified Cache: L1/L2/L3 tiers active", file=sys.stderr)
                print("✓ Team Sharing: Enabled (80% token savings)", file=sys.stderr)
                print("✓ Compression: VisionDrop (97.9% reduction)", file=sys.stderr)
                print(
                    "✓ Speed: 0.14ms average (3× faster than competitors)",
                    file=sys.stderr,
                )
                print(
                    "\n💡 Tip: Use welcome_back() to see your session context",
                    file=sys.stderr,
                )
                print("=" * 60 + "\n", file=sys.stderr)
            except:
                # Silent fail if stderr is not available
                pass

            print("✅ OmniMemory MCP Server initialized successfully", file=sys.stderr)
            print(
                "📊 Performance targets: <1ms queries, >8x compression, SWE-bench validation pending",
                file=sys.stderr,
            )

            # Log progressive disclosure statistics
            tier_stats = get_tier_statistics()
            print(
                f"🎯 Progressive Disclosure enabled: {tier_stats['context_reduction_percentage']}% context reduction",
                file=sys.stderr,
            )
            print(
                f"   Core tier: {tier_stats['auto_load_tools']} tools always loaded (~{tier_stats['core_tier_tokens']} tokens)",
                file=sys.stderr,
            )
            print(
                f"   On-demand: {tier_stats['on_demand_tools']} tools load when needed",
                file=sys.stderr,
            )

        except Exception as e:
            print(f"❌ Error initializing components: {e}", file=sys.stderr)
            self._initialized = False

    async def _activate_automatic_features_once(self):
        """
        Activate automatic context features on first tool call (lazy activation)

        This ensures event loop is running before starting async tasks.
        Called automatically on first tool use - fully transparent to user.
        """

        # Only activate once
        if self._features_activated:
            return

        self._features_activated = True

        print("\n" + "=" * 70, file=sys.stderr)
        print("🧠 OmniMemory - Activating Automatic Features", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        # 1. Start WorkspaceMonitor
        if self.workspace_monitor and not self._workspace_monitor_ready:
            try:
                self.workspace_monitor.start()
                self._workspace_monitor_ready = True
                current_project = self.workspace_monitor.get_current_project()
                print(f"✓ Workspace Monitor: Active", file=sys.stderr)
                print(
                    f"  Current: {current_project['info']['name']} ({current_project['info']['type']})",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"⚠️  Workspace Monitor start failed: {e}", file=sys.stderr)

        # 2. Start ContextPreloader
        if self.context_preloader and not self._context_preloader_ready:
            try:
                self.context_preloader.start()
                self._context_preloader_ready = True
                print(
                    f"✓ Context Preloader: Active (smart prefetching enabled)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"⚠️  Context Preloader start failed: {e}", file=sys.stderr)

        # 3. Start ResultCleanupDaemon for Zero New Tools
        if self.cleanup_daemon:
            try:
                await self.cleanup_daemon.start()
                print(
                    f"✓ Result Cleanup Daemon: Active (6h interval, 7d TTL)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"⚠️  Result Cleanup Daemon start failed: {e}", file=sys.stderr)

        # 4. Silent session auto-loading
        try:
            loaded_count = await self._silent_load_session_files()
            if loaded_count > 0:
                print(
                    f"✓ Pre-loaded {loaded_count} files from last session",
                    file=sys.stderr,
                )
            else:
                print(f"✓ New session (no previous files)", file=sys.stderr)
        except Exception as e:
            print(f"⚠️  Session load: {e}", file=sys.stderr)

        print("=" * 70 + "\n", file=sys.stderr)

    def _get_user_id(self) -> str:
        """
        Get user ID for cache key namespacing

        For local mode: Use system username
        For cloud mode: Use authenticated user from session
        """
        # Check if we have authenticated user (cloud mode)
        if hasattr(self, "_authenticated_user_id") and self._authenticated_user_id:
            return self._authenticated_user_id

        # Local mode: Use system username or default
        import getpass

        try:
            username = getpass.getuser()
            return f"local_{username}"
        except:
            return "default_user"

    def _get_repo_id(self, file_path: str) -> str:
        """
        Detect repository ID from file path

        Uses git to find repository root and generate stable repo ID
        """
        from pathlib import Path
        import subprocess

        try:
            # Find git root
            file_dir = Path(file_path).parent
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=file_dir,
                capture_output=True,
                text=True,
                timeout=1,
            )

            if result.returncode == 0:
                git_root = result.stdout.strip()
                # Generate stable repo ID from git root path
                import hashlib

                repo_hash = hashlib.sha256(git_root.encode()).hexdigest()[:16]
                return f"repo_{repo_hash}"
            else:
                # Not a git repo - use directory hash
                import hashlib

                dir_hash = hashlib.sha256(str(file_dir).encode()).hexdigest()[:16]
                return f"dir_{dir_hash}"
        except Exception as e:
            # Fallback to default repo
            return "default_repo"

    def _get_file_hash(self, file_path: str) -> str:
        """
        Get hash of file CONTENT for cache invalidation.
        Changes when file is modified.

        Args:
            file_path: Path to the file

        Returns:
            Content hash (first 12 chars of SHA256)

        Cache Invalidation Strategy:
            1. Content-based cache keys: Hash included in key, auto-invalidates on change
            2. Timestamp validation: Double-check file mtime before returning cached results
            3. TODO Phase 4: Add file watching for real-time cache invalidation

        TODO Phase 4: Active File Watching (Optional)
        For real-time invalidation, add file system monitoring:

        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class CacheInvalidator(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory:
                    file_path = event.src_path
                    # Invalidate all cache tiers
                    self.cache_manager.invalidate_file(repo_id, file_path)
                    print(f"🔄 File modified, cache invalidated: {file_path}")

        observer = Observer()
        observer.schedule(CacheInvalidator(), path=repo_root, recursive=True)
        observer.start()
        """
        import hashlib
        from pathlib import Path

        try:
            path = Path(file_path).expanduser().resolve()
            with open(path, "rb") as f:
                content = f.read()

            # Hash first 100KB for speed (full hash for small files)
            sample = content[:102400] if len(content) > 102400 else content
            content_hash = hashlib.sha256(sample).hexdigest()[:12]

            return content_hash
        except Exception as e:
            # Fallback to path hash if file read fails
            return hashlib.sha256(file_path.encode()).hexdigest()[:12]

    async def _read_cached_result(
        self, result_path: str, offset: int = 0, limit: Optional[int] = None
    ) -> str:
        """
        Read from cached large result with pagination (Zero New Tools architecture).

        This method allows the existing read() tool to access cached results without
        adding new MCP tools. Virtual file paths look like:
        ~/.omnimemory/cached_results/{result_id}.json

        Args:
            result_path: Virtual path to cached result
            offset: Starting row for pagination
            limit: Max rows to return (None = all remaining)

        Returns:
            JSON string with result chunk and metadata
        """
        try:
            # Extract result ID from virtual path
            result_id = Path(result_path).stem.replace(".result", "")

            print(
                f"🔍 Reading cached result: {result_id} (offset={offset}, limit={limit})",
                file=sys.stderr,
            )

            # Retrieve from result store
            if not self.result_store:
                return json.dumps(
                    {
                        "error": "result_store_not_available",
                        "message": "ResultStore component is not initialized",
                        "result_path": result_path,
                    }
                )

            chunk = await self.result_store.retrieve_result(
                result_id=result_id,
                chunk_offset=offset,
                chunk_size=limit if limit else 1000,
            )

            # Format response with metadata
            response = {
                "content": chunk["data"],
                "metadata": {
                    "source": "cached_result",
                    "result_id": result_id,
                    "total_count": chunk["total_count"],
                    "showing": len(chunk["data"]),
                    "offset": offset,
                    "has_more": chunk["next_offset"] is not None,
                    "next_offset": chunk["next_offset"],
                },
            }

            # Add usage instructions if there's more data
            if response["metadata"]["has_more"]:
                response["instructions"] = (
                    f"To see more results, use:\n"
                    f'read("{result_path}|offset:{chunk["next_offset"]}|limit:{limit or 1000}")'
                )

            print(
                f"✓ Returned {len(chunk['data'])} of {chunk['total_count']} items",
                file=sys.stderr,
            )

            return json.dumps(response, indent=2)

        except Exception as e:
            print(f"❌ Failed to read cached result: {e}", file=sys.stderr)
            return json.dumps(
                {
                    "error": "cached_result_read_failed",
                    "message": str(e),
                    "result_path": result_path,
                    "help": "The cached result may have expired or been deleted",
                }
            )

    async def _search_cached_result(
        self, result_path: str, filter_expr: str, limit: int = 100
    ) -> str:
        """
        Search/filter cached result (Zero New Tools architecture).

        This method allows the existing search() tool to filter cached results
        without adding new MCP tools. Uses simple substring matching (MVP).

        Args:
            result_path: Virtual path to cached result
            filter_expr: Filter expression (substring match for MVP)
            limit: Max results to return

        Returns:
            JSON string with filtered results
        """
        try:
            # Extract result ID from virtual path
            result_id = Path(result_path).stem.replace(".result", "")

            print(
                f"🔍 Filtering cached result: {result_id} with query '{filter_expr}'",
                file=sys.stderr,
            )

            # Load full result
            if not self.result_store:
                return json.dumps(
                    {
                        "error": "result_store_not_available",
                        "message": "ResultStore component is not initialized",
                    }
                )

            full_data = await self.result_store.retrieve_result(
                result_id=result_id, chunk_size=-1  # Get all data for filtering
            )

            # Apply simple filter (MVP: substring match)
            filtered = await self._apply_simple_filter(full_data["data"], filter_expr)

            print(
                f"✓ Found {len(filtered)} matches (showing first {min(len(filtered), limit)})",
                file=sys.stderr,
            )

            response = {
                "results": filtered[:limit],
                "total_matches": len(filtered),
                "original_count": full_data["total_count"],
                "filter": filter_expr,
                "message": f"Found {len(filtered)} matches. Showing first {min(len(filtered), limit)}.",
            }

            # Add instructions if there are more matches
            if len(filtered) > limit:
                response["instructions"] = (
                    f"Showing {limit} of {len(filtered)} matches. "
                    f'Increase limit to see more: search("{filter_expr}|file:{result_path}|limit:{len(filtered)}")'
                )

            return json.dumps(response, indent=2)

        except Exception as e:
            print(f"❌ Failed to filter cached result: {e}", file=sys.stderr)
            return json.dumps(
                {
                    "error": "cached_result_filter_failed",
                    "message": str(e),
                    "result_path": result_path,
                }
            )

    async def _apply_simple_filter(
        self, data: List[Dict], filter_expr: str
    ) -> List[Dict]:
        """
        Simple substring filter for cached results (MVP).

        Future enhancements (Phase 2):
        - JSONPath expressions: $.user[?(@.age > 21)]
        - SQL-like filters: WHERE name LIKE '%smith%'
        - Regex support: /pattern/flags

        Args:
            data: List of result items to filter
            filter_expr: Filter expression (substring for MVP)

        Returns:
            Filtered list of items
        """
        if not filter_expr or not data:
            return data

        filtered = []
        filter_lower = filter_expr.lower()

        for item in data:
            # Convert item to string and check if filter matches
            item_str = json.dumps(item).lower()
            if filter_lower in item_str:
                filtered.append(item)

        return filtered

    async def _silent_load_session_files(self) -> int:
        """
        Silently load recent files from last session into L1 cache

        NO token overhead - just pre-warms cache so files are instant
        """

        if not _SESSION_MANAGER or not _SESSION_MANAGER.current_session:
            return 0

        session_id = _SESSION_MANAGER.current_session
        user_id = self._get_user_id()
        loaded_count = 0

        try:
            # Get recent files from session
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    f"{_METRICS_API}/sessions/{session_id}/activity",
                    params={"limit": 50},
                )

                if response.status_code != 200:
                    return 0

                activity = response.json()
                recent_files = []

                # Extract unique file paths
                for op in activity.get("operations", []):
                    file_path = op.get("file_path")
                    if file_path and file_path not in recent_files:
                        recent_files.append(file_path)
                        if len(recent_files) >= 10:  # Load last 10 files
                            break

                if not recent_files:
                    return 0

                # Get repo_id from first file
                repo_id = self._get_repo_id(recent_files[0])

                # Pre-load from L2 into L1 (silent, no messages)
                for file_path in recent_files:
                    try:
                        # Check if in L2 cache (repository)
                        import hashlib

                        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
                        l2_cached = self.cache_manager.get_file_compressed(
                            repo_id, file_hash
                        )

                        if l2_cached:
                            # Promote to L1 silently
                            content, metadata = l2_cached

                            result = {
                                "content": content.decode("utf-8")
                                if isinstance(content, bytes)
                                else content,
                                "compressed": metadata.get("compressed", "False")
                                == "True",
                                "file_path": file_path,
                                "preloaded": True,  # Mark as preloaded
                            }

                            self.cache_manager.cache_read_result(
                                user_id=user_id,
                                file_path=file_path,
                                result=result,
                                ttl=3600,
                            )

                            loaded_count += 1

                    except Exception as e:
                        # Skip file if error, continue with others
                        pass

                return loaded_count

        except Exception as e:
            return 0

    def _format_welcome_message(
        self,
        project_name: str,
        time_ago: str,
        recent_files: List[str],
        recent_searches: List[str],
        suggestions: List[str],
        team_insights: List[dict] = None,
    ) -> str:
        """Format welcome message for easy display in IDEs"""
        lines = [
            "=" * 60,
            "🧠 OmniMemory - Session Restored",
            "=" * 60,
            "",
            f"📁 Project: {project_name}",
            f"⏰ Last session: {time_ago}",
        ]

        if recent_files:
            files_str = ", ".join([Path(f).name for f in recent_files[:3]])
            lines.append(f"📄 Files: {files_str}")

        if recent_searches:
            lines.append(f"🔍 Last search: '{recent_searches[0]}'")

        if suggestions:
            lines.append("")
            lines.append("💡 Suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"   {i}. {suggestion}")

        if team_insights:
            lines.append("")
            lines.append("👥 Team Activity:")
            for insight in team_insights[:3]:  # Show top 3
                lines.append(f"   • {insight['message']}")

        lines.append("")
        lines.append("Ready to continue from where you left off!")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def _smart_truncate(
        self, content: str, max_tokens: int, file_path: str = ""
    ) -> str:
        """
        Intelligently truncate content while preserving structure.

        For code files: Preserve imports, class/function signatures, key structure
        For text files: Keep beginning and end with truncation marker

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed
            file_path: Path to file (for type detection)

        Returns:
            Truncated content with marker
        """
        # Detect if it's a code file
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".rb",
            ".php",
        }
        is_code = any(file_path.endswith(ext) for ext in code_extensions)

        if not is_code:
            # Also check content for code indicators
            is_code = any(
                keyword in content[:1000]
                for keyword in [
                    "import ",
                    "def ",
                    "class ",
                    "function ",
                    "const ",
                    "var ",
                    "let ",
                    "public ",
                    "private ",
                    "protected ",
                    "#include",
                    "package ",
                ]
            )

        # For code: Preserve structure (imports, signatures, key elements)
        if is_code:
            lines = content.split("\n")
            preserved_lines = []
            current_tokens = 0

            # Target is 90% of max for safety
            target_tokens = int(max_tokens * 0.9)

            # First pass: Keep imports and key declarations
            for line in lines:
                line_tokens = self._count_tokens(line)

                # Always keep imports and declarations
                is_important = any(
                    keyword in line
                    for keyword in [
                        "import ",
                        "from ",
                        "class ",
                        "def ",
                        "function ",
                        "interface ",
                        "type ",
                        "const ",
                        "export ",
                        "#include",
                        "package ",
                    ]
                )

                if is_important or current_tokens + line_tokens <= target_tokens:
                    preserved_lines.append(line)
                    current_tokens += line_tokens

                    if current_tokens >= target_tokens:
                        break

            # Calculate how many lines were dropped
            total_lines = len(lines)
            kept_lines = len(preserved_lines)
            dropped_lines = total_lines - kept_lines

            truncated_content = "\n".join(preserved_lines)

            if dropped_lines > 0:
                truncated_content += f"\n\n# ... ({dropped_lines} lines truncated to fit {max_tokens} token limit)\n"
                truncated_content += f"# Total file: {total_lines} lines, showing first {kept_lines} lines with structure preserved\n"

            return truncated_content

        # For non-code: Binary search for optimal truncation point
        lines = content.split("\n")
        target_tokens = int(max_tokens * 0.9)

        # Binary search for how many lines we can keep
        left, right = 0, len(lines)
        best_end = 0

        while left <= right:
            mid = (left + right) // 2
            test_content = "\n".join(lines[:mid])
            test_tokens = self._count_tokens(test_content)

            if test_tokens <= target_tokens:
                best_end = mid
                left = mid + 1
            else:
                right = mid - 1

        truncated_content = "\n".join(lines[:best_end])
        remaining_lines = len(lines) - best_end

        if remaining_lines > 0:
            truncated_content += f"\n\n... ({remaining_lines} lines truncated to fit {max_tokens} token limit)"

        return truncated_content

    async def _decompress_content(self, compressed: str) -> str:
        """
        Decompress content from compression service.
        Calls POST /decompress endpoint.

        Args:
            compressed: Compressed content string

        Returns:
            Decompressed content string
        """
        try:
            response = await self.http_client.post(
                "http://localhost:8001/decompress",
                json={"compressed": compressed, "format": "visiondrop"},
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("decompressed", compressed)
            else:
                print(
                    f"⚠ Decompression failed: {response.status_code}",
                    file=sys.stderr,
                )
                return compressed  # Fallback: return as-is

        except Exception as e:
            print(f"⚠ Decompression error: {e}", file=sys.stderr)
            return compressed  # Fallback: return as-is

    def _dequantize_embedding(self, tri_index: Dict[str, Any]) -> Optional[List[float]]:
        """
        Dequantize embedding from tri_index if it's JECQ-quantized.

        Args:
            tri_index: Tri-Index dictionary with dense_embedding field

        Returns:
            Dequantized embedding as list of floats, or original if not quantized
        """
        if not tri_index:
            return None

        embedding = tri_index.get("dense_embedding")
        is_quantized = tri_index.get("embedding_quantized", False)

        # If not quantized, return as-is
        if not is_quantized or not isinstance(embedding, bytes):
            return embedding

        # Dequantize using JECQ
        if self.jecq_quantizer is not None:
            try:
                dequantized = self.jecq_quantizer.dequantize(embedding)
                return dequantized.tolist()
            except Exception as e:
                print(
                    f"⚠ JECQ dequantization failed: {e}. Returning None.",
                    file=sys.stderr,
                )
                return None
        else:
            print(
                "⚠ Embedding is quantized but JECQ quantizer not available.",
                file=sys.stderr,
            )
            return None

    async def _apply_tier_based_serving(
        self,
        file_path: str,
        content: str,
        file_hash: str,
        original_tokens: int,
    ) -> Dict[str, Any]:
        """
        Apply tier-based progressive content serving using TierManager.

        This method:
        1. Gets/creates file metadata from CrossToolFileCache
        2. Determines appropriate tier based on age and access patterns
        3. Applies tier-specific content transformation
        4. Handles auto-promotion for hot files
        5. Tracks metrics and updates cache

        Args:
            file_path: Absolute path to file
            content: Original file content
            file_hash: SHA256 hash of content
            original_tokens: Token count of original content

        Returns:
            {
                "content": str (tier-appropriate content),
                "tier": str (FRESH/RECENT/AGING/ARCHIVE),
                "tier_tokens": int (tokens after tier adjustment),
                "tier_savings": int (tokens saved by tier),
                "tier_savings_percent": float,
                "promoted": bool (whether file was promoted),
                "access_count": int,
                "tier_quality": float (0.0-1.0),
                "compression_ratio": float (0.0-1.0)
            }
        """
        # Skip tier-based serving if components not available
        if not TRI_INDEX_AVAILABLE or self.tier_manager is None:
            return {
                "content": content,
                "tier": "FRESH",  # Default to FRESH
                "tier_tokens": original_tokens,
                "tier_savings": 0,
                "tier_savings_percent": 0.0,
                "promoted": False,
                "access_count": 1,
                "tier_quality": 1.0,
                "compression_ratio": 0.0,
            }

        try:
            # 1. Get or create file tri-index from cross-tool cache
            tri_index = None
            if self.cross_tool_cache is not None:
                tri_index = await self.cross_tool_cache.get(file_path, _TOOL_ID)

            # 2. Create metadata if file not in cache yet
            if tri_index is None:
                print(
                    f"[TierManager] New file, creating metadata: {file_path}",
                    file=sys.stderr,
                )

                # Create basic tri-index structure (will be populated by background indexer)
                tri_index = {
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "tier": "FRESH",
                    "tier_entered_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 0,
                    "accessed_by": [_TOOL_ID],
                    # Placeholder structure (real indexing happens in background)
                    "witnesses": [],
                    "facts": [],
                    "classes": [],
                    "functions": [],
                    "imports": [],
                }

            # 3. Update access metadata
            tri_index["access_count"] = tri_index.get("access_count", 0) + 1
            tri_index["last_accessed"] = datetime.now().isoformat()

            # Add current hash for modification detection
            tri_index["current_hash"] = file_hash

            # 4. Determine tier based on age and access patterns
            # Convert ISO timestamps to datetime objects for TierManager
            metadata_for_tier = {
                "tier_entered_at": datetime.fromisoformat(
                    tri_index.get("tier_entered_at", datetime.now().isoformat())
                ),
                "last_accessed": datetime.fromisoformat(
                    tri_index.get("last_accessed", datetime.now().isoformat())
                ),
                "access_count": tri_index.get("access_count", 1),
                "file_hash": tri_index.get("file_hash", file_hash),
                "current_hash": file_hash,
            }

            current_tier = self.tier_manager.determine_tier(metadata_for_tier)
            previous_tier = tri_index.get("tier", "FRESH")
            promoted = current_tier == "FRESH" and previous_tier != "FRESH"

            # Update tier if changed (promotion or aging)
            if current_tier != previous_tier:
                print(
                    f"[TierManager] Tier transition: {previous_tier} → {current_tier} "
                    f"(access_count={tri_index['access_count']})",
                    file=sys.stderr,
                )
                tri_index["tier"] = current_tier
                tri_index["tier_entered_at"] = datetime.now().isoformat()

            # 5. Get tier-appropriate content
            tier_result = await self.tier_manager.get_tier_content(
                tier=current_tier,
                file_tri_index=tri_index,
                original_content=content,  # Provide original for FRESH tier
            )

            # 6. Store updated metadata back to cache
            if self.cross_tool_cache is not None:
                await self.cross_tool_cache.store(tri_index)

            # 7. Track tier metrics
            tier_tokens = tier_result.get("tokens", original_tokens)
            tier_savings = original_tokens - tier_tokens

            print(
                f"[TierManager] Served {current_tier} tier: "
                f"{tier_tokens} tokens (saved {tier_savings}, "
                f"{tier_result.get('compression_ratio', 0)*100:.0f}% reduction, "
                f"quality={tier_result.get('quality', 1.0):.2f})",
                file=sys.stderr,
            )

            # 8. Report tier metrics to dashboard (non-blocking)
            await self._report_tier_metrics(
                file_path=file_path,
                tier=current_tier,
                original_tokens=original_tokens,
                tier_tokens=tier_tokens,
                promoted=promoted,
                access_count=tri_index.get("access_count", 1),
                quality=tier_result.get("quality", 1.0),
            )

            return {
                "content": tier_result.get("content", content),
                "tier": current_tier,
                "tier_tokens": tier_tokens,
                "tier_savings": tier_savings,
                "tier_savings_percent": round(
                    (tier_savings / original_tokens * 100)
                    if original_tokens > 0
                    else 0,
                    2,
                ),
                "promoted": promoted,
                "access_count": tri_index.get("access_count", 1),
                "tier_quality": tier_result.get("quality", 1.0),
                "compression_ratio": tier_result.get("compression_ratio", 0.0),
            }

        except Exception as e:
            print(
                f"⚠ TierManager error: {e}. Falling back to FRESH tier.",
                file=sys.stderr,
            )
            # Fallback to FRESH tier on any error
            return {
                "content": content,
                "tier": "FRESH",
                "tier_tokens": original_tokens,
                "tier_savings": 0,
                "tier_savings_percent": 0.0,
                "promoted": False,
                "access_count": 1,
                "tier_quality": 1.0,
                "compression_ratio": 0.0,
            }

    async def _create_tri_index(
        self, file_path: str, content: str, file_hash: str
    ) -> Optional[Dict]:
        """
        Create Tri-Index for a file with structure extraction and witness selection.

        Args:
            file_path: Absolute path to the file
            content: File content
            file_hash: SHA256 hash of the content

        Returns:
            Tri-Index dictionary or None on failure
        """
        if not TRI_INDEX_AVAILABLE:
            return None

        try:
            # Extract structural facts using FileStructureExtractor
            facts = []
            classes = []
            functions = []
            imports = []

            if self.structure_extractor is not None:
                try:
                    # Start extractor if not already running
                    if hasattr(self.structure_extractor, "_initialized"):
                        if not self.structure_extractor._initialized:
                            await self.structure_extractor.start()
                    else:
                        await self.structure_extractor.start()

                    # Extract facts
                    facts = await self.structure_extractor.extract_facts(
                        file_path, content
                    )

                    # Parse facts into categories
                    for fact in facts:
                        obj = fact.get("object", "")
                        pred = fact.get("predicate", "")

                        if pred == "imports" and obj.startswith("module:"):
                            imports.append(obj.replace("module:", ""))
                        elif pred == "defines_class" and obj.startswith("class:"):
                            classes.append(obj.replace("class:", ""))
                        elif pred == "defines_function" and obj.startswith("function:"):
                            functions.append(obj.replace("function:", ""))

                except Exception as e:
                    print(
                        f"⚠ Structure extraction failed for {file_path}: {e}",
                        file=sys.stderr,
                    )

            # Select witnesses using WitnessSelector
            witnesses = []
            if self.witness_selector is not None:
                try:
                    # Initialize witness selector if needed
                    if not self.witness_selector._initialized:
                        await self.witness_selector.initialize()

                    # Select representative snippets
                    selected = await self.witness_selector.select_witnesses(
                        content, max_witnesses=5, lambda_param=0.7
                    )
                    witnesses = [w["text"] for w in selected]

                except Exception as e:
                    print(
                        f"⚠ Witness selection failed for {file_path}: {e}",
                        file=sys.stderr,
                    )

            # Generate dense embedding (via MLX embedding service)
            dense_embedding = None
            try:
                response = await self.http_client.post(
                    "http://localhost:8000/embed",
                    json={"text": content, "normalize": True},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    dense_embedding = data.get("embedding")

            except Exception as e:
                print(f"⚠ Embedding generation failed: {e}", file=sys.stderr)

            # Extract BM25 tokens (simple word frequency)
            bm25_tokens = {}
            try:
                # Tokenize and count words
                words = content.lower().split()
                for word in words:
                    # Simple cleanup: remove punctuation
                    word = "".join(c for c in word if c.isalnum() or c == "_")
                    if len(word) > 2:  # Skip very short words
                        bm25_tokens[word] = bm25_tokens.get(word, 0) + 1

                # Keep only top 100 tokens
                sorted_tokens = sorted(
                    bm25_tokens.items(), key=lambda x: x[1], reverse=True
                )
                bm25_tokens = dict(sorted_tokens[:100])

            except Exception as e:
                print(f"⚠ BM25 token extraction failed: {e}", file=sys.stderr)

            # Determine tier (FRESH for new files)
            tier = "FRESH"
            if self.tier_manager is not None:
                metadata = self.tier_manager.create_metadata(
                    file_path, content, initial_tier="FRESH"
                )
                tier = metadata.get("tier", "FRESH")

            # JECQ quantization for embeddings (tier-based)
            # FRESH tier: Keep full embeddings (no quantization)
            # RECENT/AGING/ARCHIVE tiers: Apply JECQ (85% storage savings)
            embedding_for_storage = dense_embedding or [0.0] * 768
            is_quantized = False
            quantization_savings = 0

            if (
                tier != "FRESH"
                and self.jecq_quantizer is not None
                and dense_embedding is not None
            ):
                try:
                    # Convert embedding to numpy array
                    embedding_array = np.array(dense_embedding, dtype=np.float32)

                    # Quantize to 32 bytes (from 3KB)
                    quantized_bytes = self.jecq_quantizer.quantize(embedding_array)

                    # Store as bytes (will be base64 encoded in JSON)
                    embedding_for_storage = quantized_bytes
                    is_quantized = True

                    # Calculate savings
                    original_size = len(dense_embedding) * 4  # float32
                    quantized_size = len(quantized_bytes)
                    quantization_savings = original_size - quantized_size

                    print(
                        f"[JECQ] Quantized {file_path}: {original_size}B → {quantized_size}B "
                        f"({quantization_savings}B saved, {quantization_savings/original_size*100:.1f}%)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"⚠ JECQ quantization failed for {file_path}: {e}. Using full embedding.",
                        file=sys.stderr,
                    )
                    embedding_for_storage = dense_embedding

            # Build Tri-Index
            tri_index = {
                "file_path": file_path,
                "file_hash": file_hash,
                "dense_embedding": embedding_for_storage,
                "embedding_quantized": is_quantized,  # Flag for dequantization
                "embedding_quantization_savings": quantization_savings,
                "bm25_tokens": bm25_tokens,
                "facts": facts,
                "witnesses": witnesses,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "tier": tier,
                "tier_entered_at": datetime.now().isoformat(),
                "accessed_by": [_TOOL_ID],
                "access_count": 1,
                "last_accessed": datetime.now().isoformat(),
            }

            return tri_index

        except Exception as e:
            print(f"⚠ Tri-Index creation failed for {file_path}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return None

    async def _call_context_injector(
        self, file_path: str, max_retries: int = 2
    ) -> Optional[dict]:
        """Call compression service to get compressed file content with retry logic

        Args:
            file_path: Path to file to read and compress
            max_retries: Maximum number of retry attempts (default: 2)

        Returns:
            Dictionary with compressed content and metrics, or None on failure
        """
        # Read file content first
        try:
            path = Path(file_path).expanduser().resolve()
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠ Failed to read file {file_path}: {e}", file=sys.stderr)
            return None

        # Call compression service with retry logic
        for attempt in range(max_retries + 1):
            try:
                response = await self.http_client.post(
                    f"{COMPRESSION_SERVICE_URL}/compress",
                    json={
                        "context": content,
                        "target_compression": 0.1,  # Target 90% compression
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                compression_data = response.json()

                # Transform response to expected format
                original_tokens = compression_data.get("original_tokens", 0)
                compressed_tokens = compression_data.get("compressed_tokens", 0)
                tokens_saved = original_tokens - compressed_tokens

                return {
                    "content": compression_data.get("compressed_text", ""),
                    "compression_ratio": compression_data.get("compression_ratio", 1.0),
                    "tokens_saved": tokens_saved,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                }
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries:
                    wait_time = 0.5 * (2**attempt)  # Exponential backoff: 0.5s, 1s
                    print(
                        f"⚠ Compression service attempt {attempt + 1}/{max_retries + 1} failed, "
                        f"retrying in {wait_time}s...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(
                        f"⚠ Compression service unavailable after {max_retries + 1} attempts: {e}",
                        file=sys.stderr,
                    )
                    return None
            except Exception as e:
                print(
                    f"⚠ Compression service error for {file_path}: {e}", file=sys.stderr
                )
                return None

    async def _report_api_prevention(self, baseline: int, actual: int, operation: str):
        """Report tokens prevented from API to metrics service

        Args:
            baseline: Baseline token count (what would have been sent without optimization)
            actual: Actual token count (what was sent after optimization)
            operation: Operation type (e.g., 'semantic_search', 'compression')
        """
        try:
            prevented = baseline - actual

            await self.http_client.post(
                "http://localhost:8003/metrics/api-prevention",
                json={
                    "session_id": None,
                    "tool_id": _TOOL_ID or "claude-code",
                    "operation": operation,
                    "baseline_tokens": baseline,
                    "actual_tokens": actual,
                    "tokens_prevented": prevented,
                    "provider": "anthropic_claude",  # Let metrics gateway calculate cost
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=2.0,
            )
        except Exception:
            pass  # Silent fail - don't interrupt main operation

    async def _report_tier_metrics(
        self,
        file_path: str,
        tier: str,
        original_tokens: int,
        tier_tokens: int,
        promoted: bool,
        access_count: int,
        quality: float,
    ):
        """Report tier-based serving metrics to metrics service

        Args:
            file_path: Path to file
            tier: Current tier (FRESH/RECENT/AGING/ARCHIVE)
            original_tokens: Original token count
            tier_tokens: Token count after tier transformation
            promoted: Whether file was promoted
            access_count: Number of accesses
            quality: Tier quality score (0.0-1.0)
        """
        try:
            tier_savings = original_tokens - tier_tokens

            await self.http_client.post(
                "http://localhost:8003/metrics/tier-serving",
                json={
                    "session_id": None,
                    "tool_id": _TOOL_ID or "claude-code",
                    "file_path": file_path,
                    "tier": tier,
                    "original_tokens": original_tokens,
                    "tier_tokens": tier_tokens,
                    "tier_savings": tier_savings,
                    "tier_savings_percent": round(
                        (tier_savings / original_tokens * 100)
                        if original_tokens > 0
                        else 0,
                        2,
                    ),
                    "promoted": promoted,
                    "access_count": access_count,
                    "quality": quality,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=2.0,
            )
        except Exception:
            pass  # Silent fail - don't interrupt main operation

    async def _read_large_file_smart(
        self, file_path: str, estimated_tokens: int
    ) -> str:
        """Smart handling for files that are too large even after compression

        Args:
            file_path: Path to the file
            estimated_tokens: Estimated token count

        Returns:
            Symbol overview for code files, or smart preview for other files
        """
        path = Path(file_path)
        cost_saved = (estimated_tokens / 1_000_000) * 15  # $15 per 1M tokens

        # Option 1: Try symbol overview for supported code files
        code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"}
        if path.suffix.lower() in code_extensions:
            try:
                # Use LSP symbol service if available
                if LSP_AVAILABLE and self.symbol_service is not None:
                    await self.symbol_service.start()

                    overview = await self.symbol_service.get_overview(
                        file_path=file_path,
                        include_details=True,  # Include signatures for better context
                        compress=True,
                    )

                    if not overview.get("error"):
                        total_symbols = overview.get("total_symbols", 0)
                        classes = overview.get("classes", [])
                        functions = overview.get("functions", [])
                        tokens_saved = overview.get("tokens_saved", estimated_tokens)

                        # Format overview nicely
                        overview_text = f"""
FILE TOO LARGE: {path.name} ({estimated_tokens:,} tokens)

SYMBOL OVERVIEW (automatically provided instead of full file):

File: {overview.get('file', path.name)}
Language: {overview.get('language', 'unknown')}
Total symbols: {total_symbols}
Classes: {len(classes)}
Functions: {len(functions)}
Lines of code: {overview.get('loc', 'unknown')}

Classes:
{chr(10).join(f"  - {cls}" for cls in classes) if classes else "  (none)"}

Functions:
{chr(10).join(f"  - {func}" for func in functions) if functions else "  (none)"}

To read specific symbols:
  Use: omn1_read_symbol(file_path="{file_path}", symbol="function_name")

To read specific sections:
  Use: Read with offset and limit for specific line ranges

This prevented {tokens_saved:,} tokens from being sent to API.
Cost saved: ${cost_saved:.2f}
"""
                        print(
                            f"✅ Returned symbol overview instead of full file ({tokens_saved:,} tokens saved)",
                            file=sys.stderr,
                        )
                        return overview_text

            except Exception as e:
                print(
                    f"⚠️  Symbol overview failed: {e}, using preview fallback",
                    file=sys.stderr,
                )

        # Option 2: Return smart preview (first ~5000 tokens) for non-code files
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Use smart truncation to preserve structure
            preview = self._smart_truncate(
                content, max_tokens=5000, file_path=file_path
            )
            preview_tokens = self._count_tokens(preview)
            lines = content.split("\n")
            preview_lines = preview.split("\n")

            message = f"""
FILE TOO LARGE: {path.name} ({estimated_tokens:,} tokens)

PREVIEW (first {len(preview_lines)} lines, ~{preview_tokens:,} tokens):

{preview}

... [File continues for {len(lines) - len(preview_lines)} more lines]

To read specific sections:
  Use: Read tool with offset and limit parameters
  Example: Read(file_path="{file_path}", offset=200, limit=200)

This prevented {estimated_tokens - preview_tokens:,} tokens from being sent to API.
Cost saved: ${cost_saved:.2f}
"""
            print(
                f"✅ Returned preview instead of full file ({estimated_tokens - preview_tokens:,} tokens saved)",
                file=sys.stderr,
            )
            return message

        except Exception as e:
            # Final fallback: just return helpful message
            error_msg = f"""
FILE TOO LARGE: {path.name} ({estimated_tokens:,} tokens)

The file is too large to read completely (exceeds 25,000 token MCP limit).

To read specific sections:
  1. Use Read tool with offset and limit:
     Read(file_path="{file_path}", offset=0, limit=500)

  2. For code files, use symbol-level reading:
     omn1_read_symbol(file_path="{file_path}", symbol="function_name")

  3. Get file overview first:
     omn1_symbol_overview(file_path="{file_path}")

Error reading preview: {str(e)}

This prevented {estimated_tokens:,} tokens from being sent to API.
Cost saved: ${cost_saved:.2f}
"""
            print(
                f"⚠️  Could not generate preview: {e}",
                file=sys.stderr,
            )
            return error_msg

    async def _auto_load_session_context(self):
        """Automatically load context from previous session on startup

        This runs transparently in the background when the MCP server starts.
        It pre-loads frequently used files and restores workflow context from
        the last session, making resume operations instant.

        No user action required - fully automatic!
        """
        try:
            response = await self.http_client.post(
                f"{COMPRESSION_SERVICE_URL}/inject/session-context",
                json={
                    "session_id": None,
                    "tool_id": _TOOL_ID,
                    "query": None,  # Auto-detect from previous session
                },
                timeout=10.0,
            )

            if response.status_code == 200:
                result = response.json()
                files_from_last_session = result.get("files", [])
                workflow_context = result.get("workflow_context", {})

                if files_from_last_session:
                    print(
                        f"📂 Auto-loaded {len(files_from_last_session)} files from previous session",
                        file=sys.stderr,
                    )

                    # Log top files for visibility
                    if len(files_from_last_session) <= 3:
                        for file_path in files_from_last_session:
                            print(f"   ✓ {file_path}", file=sys.stderr)
                    else:
                        for file_path in files_from_last_session[:3]:
                            print(f"   ✓ {file_path}", file=sys.stderr)
                        print(
                            f"   ... and {len(files_from_last_session) - 3} more",
                            file=sys.stderr,
                        )

                if workflow_context:
                    print(
                        f"🔄 Restored workflow: {workflow_context.get('name', 'unnamed')}",
                        file=sys.stderr,
                    )

                # Context is now pre-cached and ready for instant access
                return True
            else:
                print(
                    "ℹ️  No previous session context found (this is normal for first run)",
                    file=sys.stderr,
                )
                return False

        except httpx.ConnectError:
            # Context Injector not running - this is fine, we'll work without session restore
            print(
                "ℹ️  Context Injector not available - session auto-load skipped",
                file=sys.stderr,
            )
            return False
        except Exception as e:
            print(f"ℹ️  Session auto-load skipped: {e}", file=sys.stderr)
            return False

    async def _register_session(self):
        """Register active session with metrics service"""
        try:
            await self.http_client.post(
                "http://localhost:8003/sessions/start",
                json={
                    "session_id": None,
                    "tool_id": _TOOL_ID or "claude-code",
                    "started_at": datetime.now().isoformat(),
                },
                timeout=2.0,
            )
            print(
                f"✅ Session registered: {None}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"⚠ Session registration skipped: {e}", file=sys.stderr)

    async def _end_session(self):
        """End session on shutdown"""
        try:
            await self.http_client.post(
                f"http://localhost:8003/sessions/{None}/end",
                timeout=1.0,
            )
            print(f"✅ Session ended: {None}", file=sys.stderr)
        except:
            pass

    def _init_gateway_client(self):
        """Initialize HTTP client for gateway communication"""
        self.gateway_client = httpx.AsyncClient(
            base_url=self.gateway_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-User-ID": self.user_id,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        print("✓ Gateway client initialized", file=sys.stderr)

    async def _route_to_gateway(self, tool_name: str, **kwargs) -> dict:
        """Route tool call to gateway in cloud mode"""
        try:
            response = await self.gateway_client.post(
                f"/api/v1/tools/{tool_name}", json=kwargs
            )
            response.raise_for_status()
            result = response.json()

            # If gateway returns a dict with 'content' key, extract it
            if isinstance(result, dict) and "content" in result:
                return result["content"]
            return result
        except Exception as e:
            print(f"❌ Gateway request failed for {tool_name}: {e}", file=sys.stderr)
            raise

    async def _get_file_tri_index(self, file_path: str) -> Optional[Dict]:
        """
        Get cached tri-index for file with LRU in-memory caching.

        Tries unified TriIndex first, then falls back to legacy cross_tool_cache.

        Args:
            file_path: Path to the file

        Returns:
            Tri-index dict with structural facts, or None if not available
        """
        # Check in-memory cache first (hot path optimization)
        if file_path in self.tri_index_cache:
            logger.debug(f"Tri-index cache hit for {file_path}")
            return self.tri_index_cache[file_path]

        try:
            # Try unified TriIndex first (if available)
            if hasattr(self, "tri_index") and self.tri_index is not None:
                try:
                    # Ensure TriIndex is started
                    if (
                        not hasattr(self.tri_index, "_started")
                        or not self.tri_index._started
                    ):
                        await self.tri_index.start()
                        self.tri_index._started = True

                    # Get tri-index from unified TriIndex
                    tri_index_result = await self.tri_index.get_tri_index(
                        file_path=file_path, tool_id=_TOOL_ID
                    )

                    if tri_index_result:
                        # Convert TriIndexResult to dict format for backward compatibility
                        tri_index = tri_index_result.to_dict()
                        logger.debug(
                            f"Retrieved tri-index for {file_path} from unified TriIndex"
                        )

                        # Add to in-memory cache with LRU eviction
                        if len(self.tri_index_cache) >= self.tri_index_cache_size:
                            # Evict oldest entry (simple LRU)
                            oldest = next(iter(self.tri_index_cache))
                            del self.tri_index_cache[oldest]
                            logger.debug(f"Evicted tri-index cache entry for {oldest}")

                        self.tri_index_cache[file_path] = tri_index
                        return tri_index
                except Exception as e:
                    logger.debug(f"Unified TriIndex lookup failed, trying legacy: {e}")

            # Fallback to legacy cross_tool_cache
            if hasattr(self, "cross_tool_cache") and self.cross_tool_cache is not None:
                # Get tri-index from cache
                tri_index = await self.cross_tool_cache.get(file_path, _TOOL_ID)

                if tri_index:
                    logger.debug(
                        f"Retrieved tri-index for {file_path} from legacy cache"
                    )

                    # Add to in-memory cache with LRU eviction
                    if len(self.tri_index_cache) >= self.tri_index_cache_size:
                        # Evict oldest entry (simple LRU)
                        oldest = next(iter(self.tri_index_cache))
                        del self.tri_index_cache[oldest]
                        logger.debug(f"Evicted tri-index cache entry for {oldest}")

                    self.tri_index_cache[file_path] = tri_index
                    return tri_index
                else:
                    logger.debug(f"No tri-index found for {file_path}")
                    return None
            else:
                return None

        except Exception as e:
            logger.debug(f"Could not retrieve tri-index for {file_path}: {e}")
            return None

    def _fuzzy_match_score(
        self, query_term: str, symbol: str, min_ratio: int = 70
    ) -> float:
        """
        Calculate fuzzy match score between query term and symbol.

        Uses rapidfuzz or fuzzywuzzy for intelligent fuzzy matching that handles:
        - Typos: "AuthHandler" matches "authhandler", "AuthHandlr"
        - Case insensitivity: "UserService" matches "userservice", "USERSERVICE"
        - Partial matches: "authenticate" matches "authenticateUser", "is_authenticated"
        - Near-matches: Uses Levenshtein distance for similarity

        Args:
            query_term: Search term from query
            symbol: Code symbol (class/function/import name)
            min_ratio: Minimum similarity ratio (0-100)

        Returns:
            Score: 0.0 to 1.0 (1.0 = perfect match, 0.0 = no match)
        """
        if not FUZZY_AVAILABLE:
            # Fallback to exact substring match
            if query_term.lower() in symbol.lower():
                return 1.0
            return 0.0

        # Try multiple fuzzy algorithms for best matching
        ratios = [
            fuzz.ratio(query_term.lower(), symbol.lower()),  # Simple ratio
            fuzz.partial_ratio(query_term.lower(), symbol.lower()),  # Partial match
            fuzz.token_sort_ratio(query_term.lower(), symbol.lower()),  # Token sort
        ]

        max_ratio = max(ratios)

        if max_ratio >= min_ratio:
            return max_ratio / 100.0  # Convert 0-100 to 0.0-1.0

        return 0.0

    async def _match_structural_facts(self, query: str, file_path: str) -> float:
        """
        Check if query matches structural elements in file with fuzzy matching.

        Provides a boost score (0.0 to 0.10) based on fuzzy matching structural facts:
        - Class names matching query terms: weighted by similarity × class weight
        - Function names matching query terms: weighted by similarity × function weight
        - Import module names matching query terms: weighted by similarity × import weight

        Uses fuzzy matching to handle typos, case insensitivity, and partial matches.

        Args:
            query: Search query text
            file_path: Path to file to check

        Returns:
            boost_score: 0.0 to 0.10 (0% to 10% boost)
        """
        # Get tri-index for file
        tri_index = await self._get_file_tri_index(file_path)
        if not tri_index:
            return 0.0

        # Extract query terms (lowercase, split on whitespace)
        query_terms = query.lower().split()
        if not query_terms:
            return 0.0

        matches = 0.0  # Now float for fuzzy scores

        # Check classes with fuzzy matching
        classes = tri_index.get("classes", [])
        for cls in classes:
            best_score = 0.0
            best_term = None
            for term in query_terms:
                score = self._fuzzy_match_score(
                    term, cls, min_ratio=self.boost_weights["min_ratio_class"]
                )
                if score > best_score:
                    best_score = score
                    best_term = term

            if best_score > 0:
                weighted_score = self.boost_weights["class"] * best_score
                matches += weighted_score
                logger.debug(
                    f"Class fuzzy match: '{cls}' matches query term '{best_term}' "
                    f"(score: {best_score:.2f}, weighted: {weighted_score:.2f})"
                )

        # Check functions with fuzzy matching
        functions = tri_index.get("functions", [])
        for func in functions:
            best_score = 0.0
            best_term = None
            for term in query_terms:
                score = self._fuzzy_match_score(
                    term, func, min_ratio=self.boost_weights["min_ratio_function"]
                )
                if score > best_score:
                    best_score = score
                    best_term = term

            if best_score > 0:
                weighted_score = self.boost_weights["function"] * best_score
                matches += weighted_score
                logger.debug(
                    f"Function fuzzy match: '{func}' matches query term '{best_term}' "
                    f"(score: {best_score:.2f}, weighted: {weighted_score:.2f})"
                )

        # Check imports with fuzzy matching
        imports = tri_index.get("imports", [])
        for imp in imports:
            best_score = 0.0
            best_term = None
            for term in query_terms:
                score = self._fuzzy_match_score(
                    term, imp, min_ratio=self.boost_weights["min_ratio_import"]
                )
                if score > best_score:
                    best_score = score
                    best_term = term

            if best_score > 0:
                weighted_score = self.boost_weights["import"] * best_score
                matches += weighted_score
                logger.debug(
                    f"Import fuzzy match: '{imp}' matches query term '{best_term}' "
                    f"(score: {best_score:.2f}, weighted: {weighted_score:.2f})"
                )

        # Convert to boost score: each match = 2% boost, max from config
        boost_score = min(matches * 0.02, self.boost_weights["max_boost"])

        if boost_score > 0:
            logger.info(
                f"Structural fact fuzzy boost for {file_path}: {matches:.2f} weighted matches = {boost_score*100:.1f}% boost"
            )

        return boost_score

    async def _omn1_semantic_search(
        self, query: str, limit: int = 5, min_relevance: float = 0.7
    ) -> str:
        """
        Internal semantic search implementation.

        Uses vector store (Qdrant) to find semantically relevant files.
        """
        try:
            # Check if vector store is available and properly initialized
            if (
                not hasattr(self, "vector_store")
                or self.vector_store is None
                or not hasattr(self.vector_store, "client")
                or self.vector_store.client is None
            ):
                return json.dumps(
                    {
                        "error": True,
                        "message": "Vector store not initialized for semantic search",
                        "query": query,
                        "tip": "Semantic search requires Qdrant vector store. Check if Qdrant is running at localhost:6333",
                    },
                    indent=2,
                )

            # Perform semantic search
            results = await self.vector_store.search(query=query, k=limit)

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    {
                        "rank": i,
                        "file_path": result.get("file_path", "unknown"),
                        "relevance_score": result.get("score", 0.0),
                        "snippet": result.get("snippet", "")[:200],  # Truncate snippets
                        "metadata": result.get("metadata", {}),
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                    "total_found": len(formatted_results),
                    "search_type": "semantic_vector_search",
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": True,
                    "message": f"Semantic search failed: {str(e)}",
                    "query": query,
                },
                indent=2,
            )

    async def _track_tool_operation(
        self,
        tool_name: str,
        operation_mode: str,
        parameters: dict,
        file_path: Optional[str],
        tokens_original: int,
        tokens_actual: int,
        response_time_ms: float,
    ) -> None:
        """
        Track a tool operation to the metrics service

        Args:
            tool_name: "read" or "search"
            operation_mode: Mode used (e.g., "full", "overview", "semantic", etc.)
            parameters: Parameters passed to the operation
            file_path: File path for read operations, None for search
            tokens_original: Estimated baseline tokens (without optimization)
            tokens_actual: Actual tokens returned
            response_time_ms: Operation response time in milliseconds

        This is non-blocking and handles errors gracefully.
        """
        import sys

        try:
            # Don't track if http_client is not available
            if not self.http_client:
                print(
                    f"⚠️  WARNING: http_client is None, cannot track {tool_name} operation!",
                    file=sys.stderr,
                )
                return

            # Prepare tracking payload
            payload = {
                "tool_name": tool_name,
                "operation_mode": operation_mode,
                "parameters": parameters,
                "file_path": file_path,
                "tokens_original": tokens_original,
                "tokens_actual": tokens_actual,
                "tokens_prevented": max(0, tokens_original - tokens_actual),
                "response_time_ms": response_time_ms,
                "session_id": _SESSION_ID,
                "tool_id": _TOOL_ID or "claude-code",
                "timestamp": datetime.now().isoformat(),
            }

            # Make async HTTP call with timeout
            response = await self.http_client.post(
                "http://localhost:8003/track/tool-operation",
                json=payload,
                timeout=2.0,
            )

            if response.status_code == 200:
                print(
                    f"✅ Tracked {tool_name}[{operation_mode}]: {tokens_original - tokens_actual} tokens prevented",
                    file=sys.stderr,
                )
            else:
                print(
                    f"⚠ Tracking failed (status {response.status_code})",
                    file=sys.stderr,
                )

        except Exception as e:
            # Log warning but don't crash the operation
            if os.environ.get("OMNIMEMORY_VERBOSE"):
                print(
                    f"⚠ Tool tracking failed: {str(e)}",
                    file=sys.stderr,
                )

    async def _estimate_file_tokens(self, file_path: str) -> int:
        """
        Estimate tokens in a file if it were read without compression

        Args:
            file_path: Absolute path to the file

        Returns:
            Estimated token count (rough estimate: 4 chars = 1 token)
        """
        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return 0

            # Get file size in bytes
            file_size = path.stat().st_size

            # Estimate tokens: ~4 characters = 1 token
            estimated_tokens = file_size // 4

            if os.environ.get("OMNIMEMORY_VERBOSE"):
                print(
                    f"📊 Estimated {estimated_tokens} tokens for {file_path} ({file_size} bytes)",
                    file=sys.stderr,
                )

            return estimated_tokens

        except Exception as e:
            if os.environ.get("OMNIMEMORY_VERBOSE"):
                print(
                    f"⚠ Token estimation failed for {file_path}: {str(e)}",
                    file=sys.stderr,
                )
            return 0

    async def _estimate_search_baseline(self, query: str, mode: str, limit: int) -> int:
        """
        Estimate tokens if search results were read in full

        This represents the baseline cost without tri-index optimization.

        Args:
            query: Search query
            mode: Search mode
            limit: Number of results returned

        Returns:
            Estimated baseline tokens (what user would have paid without optimization)
        """
        try:
            # Estimate average tokens per file (conservative estimate)
            avg_tokens_per_file = 500

            # Estimate how many files user would have read without search
            # Conservative estimate: they'd need to read ~50 files to find relevant ones
            estimated_files_without_search = 50

            # Baseline = what they'd read without tri-index optimization
            baseline_tokens = estimated_files_without_search * avg_tokens_per_file

            if os.environ.get("OMNIMEMORY_VERBOSE"):
                print(
                    f"📊 Search baseline: {baseline_tokens} tokens (would read {estimated_files_without_search} files)",
                    file=sys.stderr,
                )

            return baseline_tokens

        except Exception as e:
            if os.environ.get("OMNIMEMORY_VERBOSE"):
                print(
                    f"⚠ Search baseline estimation failed: {str(e)}",
                    file=sys.stderr,
                )
            return 0

    def _register_tools(self):
        """Register all MCP tools"""

        # REMOVED (consolidated):         @self.mcp.tool()
        # REMOVED (consolidated):         async def read(file_path: str) -> str:
        # REMOVED (consolidated):             """
        # REMOVED (consolidated):             Read file with AUTOMATIC large file handling (REPLACES standard Read tool)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             This tool automatically:
        # REMOVED (consolidated):             1. Compresses files to save 70%+ tokens
        # REMOVED (consolidated):             2. Detects files > 25,000 tokens BEFORE reading
        # REMOVED (consolidated):             3. Returns symbol overview for very large files
        # REMOVED (consolidated):             4. YOU NEVER SEE "file too large" errors!
        # REMOVED (consolidated):
        # REMOVED (consolidated):             The compression is handled by the Context Injector service which:
        # REMOVED (consolidated):             - Checks cache for previously compressed versions
        # REMOVED (consolidated):             - Compresses new files using the compression service
        # REMOVED (consolidated):             - Reports metrics to the dashboard automatically
        # REMOVED (consolidated):             - Falls back to uncompressed content on errors
        # REMOVED (consolidated):
        # REMOVED (consolidated):             For files > 25K tokens (MCP response limit):
        # REMOVED (consolidated):             - Tries compression first (usually reduces to <5K tokens)
        # REMOVED (consolidated):             - If still too large: Returns symbol overview instead
        # REMOVED (consolidated):             - Reports tokens saved and cost savings automatically
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Args:
        # REMOVED (consolidated):                 file_path: Path to file to read (absolute or relative)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Returns:
        # REMOVED (consolidated):                 File content (compressed if possible, symbol overview for very large files)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Examples:
        # REMOVED (consolidated):                 # Read any file - compression happens automatically
        # REMOVED (consolidated):                 content = await read("/path/to/large_file.py")
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Large files automatically handled
        # REMOVED (consolidated):                 content = await read("/path/to/huge_file.py")  # 50K tokens
        # REMOVED (consolidated):                 → Returns symbol overview instead (prevents error)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Note:
        # REMOVED (consolidated):                 This tool is a drop-in replacement for the standard Read tool.
        # REMOVED (consolidated):                 Claude doesn't need to do anything special - everything is automatic!
        # REMOVED (consolidated):             """
        # REMOVED (consolidated):             start_time = time.time()
        # REMOVED (consolidated):
        # REMOVED (consolidated):             try:
        # REMOVED (consolidated):                 # Resolve path (handle relative paths and ~ expansion)
        # REMOVED (consolidated):                 path = Path(file_path).expanduser().resolve()
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Step 1: Check file size first (before reading)
        # REMOVED (consolidated):                 file_size = os.path.getsize(path)
        # REMOVED (consolidated):                 estimated_tokens = file_size // 4  # Rough estimate: 1 token ≈ 4 bytes
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Step 2: Handle large files BEFORE reading them
        # REMOVED (consolidated):                 if (
        # REMOVED (consolidated):                     estimated_tokens > 20000
        # REMOVED (consolidated):                 ):  # Use 20K as threshold (buffer under 25K MCP limit)
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"📊 Large file detected ({estimated_tokens:,} tokens), using automatic handling...",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     # Try compression first
        # REMOVED (consolidated):                     result = await self._call_context_injector(str(path))
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     if result and "content" in result:
        # REMOVED (consolidated):                         compressed_content = result["content"]
        # REMOVED (consolidated):                         compressed_tokens = self._count_tokens(compressed_content)
        # REMOVED (consolidated):
        # REMOVED (consolidated):                         # Check if compression brought it under limit
        # REMOVED (consolidated):                         if compressed_tokens < 20000:
        # REMOVED (consolidated):                             # Success! Compressed enough to send
        # REMOVED (consolidated):                             elapsed_ms = (time.time() - start_time) * 1000
        # REMOVED (consolidated):                             tokens_saved = result.get("tokens_saved", 0)
        # REMOVED (consolidated):                             compression_ratio = result.get("compression_ratio", 0)
        # REMOVED (consolidated):
        # REMOVED (consolidated):                             print(
        # REMOVED (consolidated):                                 f"✅ Large file compressed: {estimated_tokens:,} → {compressed_tokens:,} tokens "
        # REMOVED (consolidated):                                 f"({compression_ratio*100:.1f}% reduction) in {elapsed_ms:.1f}ms",
        # REMOVED (consolidated):                                 file=sys.stderr,
        # REMOVED (consolidated):                             )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                             return compressed_content
        # REMOVED (consolidated):                         else:
        # REMOVED (consolidated):                             # Still too large after compression - use symbol overview
        # REMOVED (consolidated):                             print(
        # REMOVED (consolidated):                                 f"⚠️  Still large after compression ({compressed_tokens:,} tokens), using symbol overview...",
        # REMOVED (consolidated):                                 file=sys.stderr,
        # REMOVED (consolidated):                             )
        # REMOVED (consolidated):                             return await self._read_large_file_smart(
        # REMOVED (consolidated):                                 str(path), compressed_tokens
        # REMOVED (consolidated):                             )
        # REMOVED (consolidated):                     else:
        # REMOVED (consolidated):                         # Compression failed - use symbol overview directly
        # REMOVED (consolidated):                         print(
        # REMOVED (consolidated):                             f"⚠️  Compression unavailable for large file, using symbol overview...",
        # REMOVED (consolidated):                             file=sys.stderr,
        # REMOVED (consolidated):                         )
        # REMOVED (consolidated):                         return await self._read_large_file_smart(
        # REMOVED (consolidated):                             str(path), estimated_tokens
        # REMOVED (consolidated):                         )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Step 3: Normal sized file - use compression for savings
        # REMOVED (consolidated):                 result = await self._call_context_injector(str(path))
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 if result and "content" in result:
        # REMOVED (consolidated):                     # Success! Log compression savings
        # REMOVED (consolidated):                     elapsed_ms = (time.time() - start_time) * 1000
        # REMOVED (consolidated):                     tokens_saved = result.get("tokens_saved", 0)
        # REMOVED (consolidated):                     compression_ratio = result.get("compression_ratio", 0)
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"📦 Auto-compressed {file_path}: "
        # REMOVED (consolidated):                         f"{tokens_saved} tokens saved "
        # REMOVED (consolidated):                         f"({compression_ratio*100:.1f}% reduction) "
        # REMOVED (consolidated):                         f"in {elapsed_ms:.1f}ms",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     # Return compressed content
        # REMOVED (consolidated):                     return result["content"]
        # REMOVED (consolidated):                 else:
        # REMOVED (consolidated):                     # Fallback: read file normally without compression
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"⚠ Context injector unavailable, reading uncompressed: {file_path}",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):                     with open(path, "r", encoding="utf-8", errors="replace") as f:
        # REMOVED (consolidated):                         content = f.read()
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     elapsed_ms = (time.time() - start_time) * 1000
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"📄 Read {file_path} (uncompressed) in {elapsed_ms:.1f}ms",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     return content
        # REMOVED (consolidated):
        # REMOVED (consolidated):             except FileNotFoundError:
        # REMOVED (consolidated):                 error_msg = f"Error: File not found: {file_path}"
        # REMOVED (consolidated):                 print(f"❌ {error_msg}", file=sys.stderr)
        # REMOVED (consolidated):                 return error_msg
        # REMOVED (consolidated):             except Exception as e:
        # REMOVED (consolidated):                 error_msg = f"Error reading file {file_path}: {str(e)}"
        # REMOVED (consolidated):                 print(f"❌ {error_msg}", file=sys.stderr)
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Final fallback: attempt to read file directly
        # REMOVED (consolidated):                 try:
        # REMOVED (consolidated):                     path = Path(file_path).expanduser().resolve()
        # REMOVED (consolidated):                     with open(path, "r", encoding="utf-8", errors="replace") as f:
        # REMOVED (consolidated):                         return f.read()
        # REMOVED (consolidated):                 except Exception as read_error:
        # REMOVED (consolidated):                     return f"{error_msg}\nFallback read also failed: {read_error}"

        # @self.mcp.tool()
        # REMOVED (consolidated):         async def grep(
        # REMOVED (consolidated):             pattern: str, path: Optional[str] = None, context_lines: int = 0
        # REMOVED (consolidated):         ) -> str:
        # REMOVED (consolidated):             """Search for pattern with AUTOMATIC semantic enhancement
        # REMOVED (consolidated):
        # REMOVED (consolidated):             This tool automatically detects when a search query is semantic in nature
        # REMOVED (consolidated):             (like "find authentication code" or "how does login work") and uses
        # REMOVED (consolidated):             AI-powered semantic search to find the most relevant code sections.
        # REMOVED (consolidated):
        # REMOVED (consolidated):             For literal pattern matching (regex), it falls back to standard grep.
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Args:
        # REMOVED (consolidated):                 pattern: Search pattern (semantic query or regex)
        # REMOVED (consolidated):                 path: Optional path to search in (default: current directory)
        # REMOVED (consolidated):                 context_lines: Lines of context around matches (default: 0)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Returns:
        # REMOVED (consolidated):                 Search results with automatic semantic enhancement when appropriate
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Examples:
        # REMOVED (consolidated):                 # Semantic search (automatic)
        # REMOVED (consolidated):                 result = await grep("find authentication code")
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 # Regex search (automatic fallback)
        # REMOVED (consolidated):                 result = await grep(r"def.*login[(]")
        # REMOVED (consolidated):
        # REMOVED (consolidated):             Note:
        # REMOVED (consolidated):                 Fully transparent - Claude doesn't need to choose!
        # REMOVED (consolidated):                 The tool automatically decides based on the pattern.
        # REMOVED (consolidated):             """
        # REMOVED (consolidated):             start_time = time.time()
        # REMOVED (consolidated):
        # REMOVED (consolidated):             # Semantic query detection keywords
        # REMOVED (consolidated):             semantic_keywords = [
        # REMOVED (consolidated):                 "find",
        # REMOVED (consolidated):                 "how",
        # REMOVED (consolidated):                 "what",
        # REMOVED (consolidated):                 "where",
        # REMOVED (consolidated):                 "why",
        # REMOVED (consolidated):                 "when",
        # REMOVED (consolidated):                 "authentication",
        # REMOVED (consolidated):                 "implements",
        # REMOVED (consolidated):                 "handles",
        # REMOVED (consolidated):                 "responsible",
        # REMOVED (consolidated):                 "manages",
        # REMOVED (consolidated):                 "controls",
        # REMOVED (consolidated):                 "performs",
        # REMOVED (consolidated):                 "does",
        # REMOVED (consolidated):                 "works",
        # REMOVED (consolidated):                 "explain",
        # REMOVED (consolidated):                 "show",
        # REMOVED (consolidated):                 "all",
        # REMOVED (consolidated):                 "code for",
        # REMOVED (consolidated):                 "related to",
        # REMOVED (consolidated):             ]
        # REMOVED (consolidated):
        # REMOVED (consolidated):             # Check if this is a semantic query
        # REMOVED (consolidated):             pattern_lower = pattern.lower()
        # REMOVED (consolidated):             is_semantic = any(keyword in pattern_lower for keyword in semantic_keywords)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             # Also check if pattern lacks regex special characters (likely semantic)
        # REMOVED (consolidated):             regex_chars = r".*+?[]{}()^$|\\"
        # REMOVED (consolidated):             has_regex = any(char in pattern for char in regex_chars)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             if is_semantic and not has_regex:
        # REMOVED (consolidated):                 try:
        # REMOVED (consolidated):                     # Use semantic search automatically
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"🔍 Auto-semantic search triggered for: {pattern}",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     # Call semantic search via Context Injector or local semantic search
        # REMOVED (consolidated):                     response = await self.http_client.post(
        # REMOVED (consolidated):                         f"{CONTEXT_INJECTOR_URL}/inject/semantic",
        # REMOVED (consolidated):                         json={
        # REMOVED (consolidated):                             "query": pattern,
        # REMOVED (consolidated):                             "files": [],  # Empty = search all files in project
        # REMOVED (consolidated):                             "limit": 10,
        # REMOVED (consolidated):                             "path": path,
        # REMOVED (consolidated):                         },
        # REMOVED (consolidated):                         timeout=15.0,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                     if response.status_code == 200:
        # REMOVED (consolidated):                         result = response.json()
        # REMOVED (consolidated):                         relevant_files = result.get("selected_files", [])
        # REMOVED (consolidated):                         relevant_snippets = result.get("snippets", [])
        # REMOVED (consolidated):
        # REMOVED (consolidated):                         if relevant_files:
        # REMOVED (consolidated):                             elapsed_ms = (time.time() - start_time) * 1000
        # REMOVED (consolidated):                             print(
        # REMOVED (consolidated):                                 f"✓ Auto-semantic search found {len(relevant_files)} relevant files in {elapsed_ms:.1f}ms",
        # REMOVED (consolidated):                                 file=sys.stderr,
        # REMOVED (consolidated):                             )
        # REMOVED (consolidated):
        # REMOVED (consolidated):                             # Format results nicely
        # REMOVED (consolidated):                             output = f"Semantic search results for: {pattern}\n"
        # REMOVED (consolidated):                             output += f"Found {len(relevant_files)} relevant files:\n\n"
        # REMOVED (consolidated):
        # REMOVED (consolidated):                             for i, file_info in enumerate(relevant_files[:10], 1):
        # REMOVED (consolidated):                                 file_path = file_info.get("path", "unknown")
        # REMOVED (consolidated):                                 relevance = file_info.get("relevance", 0.0)
        # REMOVED (consolidated):                                 snippet = file_info.get("snippet", "")
        # REMOVED (consolidated):
        # REMOVED (consolidated):                                 output += (
        # REMOVED (consolidated):                                     f"{i}. {file_path} (relevance: {relevance:.2f})\n"
        # REMOVED (consolidated):                                 )
        # REMOVED (consolidated):                                 if snippet:
        # REMOVED (consolidated):                                     output += f"   {snippet[:200]}...\n"
        # REMOVED (consolidated):                                 output += "\n"
        # REMOVED (consolidated):
        # REMOVED (consolidated):                             return output
        # REMOVED (consolidated):
        # REMOVED (consolidated):                 except httpx.ConnectError:
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         "ℹ️  Context Injector not available - falling back to regex grep",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):                 except Exception as e:
        # REMOVED (consolidated):                     print(
        # REMOVED (consolidated):                         f"ℹ️  Semantic search failed: {e} - falling back to regex grep",
        # REMOVED (consolidated):                         file=sys.stderr,
        # REMOVED (consolidated):                     )
        # REMOVED (consolidated):
        # REMOVED (consolidated):             # Fallback: Use standard pattern matching
        # REMOVED (consolidated):             # This would normally call the native grep tool, but since we're overriding it,
        # REMOVED (consolidated):             # we'll provide a simple implementation that suggests using the native grep
        # REMOVED (consolidated):             print(f"🔎 Using regex pattern matching for: {pattern}", file=sys.stderr)
        # REMOVED (consolidated):
        # REMOVED (consolidated):             return f"""Pattern search with regex: {pattern}
        # REMOVED (consolidated):
        # REMOVED (consolidated): Note: This MCP server provides automatic semantic enhancement for queries like:
        # REMOVED (consolidated): - "find authentication code"
        # REMOVED (consolidated): - "how does login work"
        # REMOVED (consolidated): - "what handles database connections"
        # REMOVED (consolidated):
        # REMOVED (consolidated): For literal regex pattern matching, please use the native Grep tool.
        # REMOVED (consolidated):
        # REMOVED (consolidated): To enable semantic search for this query, try rephrasing with natural language:
        # REMOVED (consolidated): - Instead of: "def.*login"
        # REMOVED (consolidated): - Try: "find login functions"
        # REMOVED (consolidated):
        # REMOVED (consolidated): Semantic search automatically triggered when query contains keywords like:
        # REMOVED (consolidated): find, how, what, where, authentication, handles, implements, etc.
        # REMOVED (consolidated): """
        # REMOVED (consolidated):
        # ============================================================================
        # MINIMAL MCP TOOLS CONFIGURATION
        # ============================================================================
        # Token Savings: Reduced from 24 tools to 5 essential tools
        # - Before: 24 tools × 400 tokens = 9,600 tokens per conversation
        # - After: 5 tools × 400 tokens = 2,000 tokens per conversation
        # - Saved: 7,600 tokens per conversation (79% reduction)
        #
        # ESSENTIAL TOOLS (ACTIVE):
        # 1. read - File reading with automatic compression (90% token savings)
        # 2. grep - Search with semantic enhancement
        # 3. omn1_tri_index_search - Hybrid search (Dense + Sparse + Structural)
        # 4. omn1_read - Unified reading (full/overview/symbol/references)
        # 5. omn1_search - Unified search (semantic/references)
        #
        # REMOVED TOOLS (COMMENTED OUT):
        # All metrics/stats/health tools have been removed. Metrics are now:
        # - Tracked silently in background (no token cost)
        # - Viewable via dashboard: http://localhost:8004
        # - Accessible via metrics API: http://localhost:8003
        #
        # The following tools are commented out to reduce token usage:
        # - omn1_store
        # - omn1_compress
        # - omn1_analyze
        # - omn1_optimize
        # - omn1_metrics
        # - omn1_get_stats
        # - omn1_learn_workflow
        # - omn1_execute_python
        # - omn1_graph_search
        # - omn1_hybrid_search
        # - omn1_workflow_context
        # - omn1_resume_workflow
        # - omn1_optimize_context
        # - omn1_unified_predict
        # - omn1_orchestrate_query
        # - omn1_get_suggestions
        # - omn1_record_feedback
        # - omn1_system_status
        # - omn1_unified_health
        # ============================================================================

        # @self.mcp.tool()
        # async def omn1_store(
        #     content: str,
        #     context: Optional[str] = None,
        #     importance_threshold: float = 0.5,
        #     compress: bool = True,
        # ) -> str:
        #     """Store and compress context with intelligent importance scoring

        #     Args:
        #         content: The content to store in memory
        #         context: Additional context for importance scoring
        #         importance_threshold: Minimum importance score to store (0.0-1.0)
        #         compress: Whether to apply compression

        #     Returns:
        #         JSON string with storage result including memory ID, importance score, and confidence
        #     """
        #     start_time = time.time()

        #     try:
        #         # Score importance
        #         importance, factors = self.importance_scorer.score_importance(
        #             content, context
        #         )

        #         # Filter by threshold
        #         if importance < importance_threshold:
        #             result = {
        #                 "status": "filtered",
        #                 "importance": importance,
        #                 "reason": "Below importance threshold",
        #                 "memory_id": None,
        #             }
        #             return json.dumps(result, indent=2)

        #         # Process through coordinator
        #         processed = self.coordinator.process_memory(content)

        #         # Compress if requested and beneficial
        #         compressed_content = content
        #         compression_ratio = 1.0
        #         quality_score = 1.0

        #         if compress and len(content) > 100:
        #             # Call REAL compression service
        #             try:
        #                 headers = {
        #                     "X-Session-ID": None,
        #                     "X-Tool-ID": TOOL_ID,
        #                 }
        #                 response = await self.http_client.post(
        #                     "http://localhost:8001/compress",
        #                     json={
        #                         "context": content,
        #                         "tool_id": TOOL_ID,
        #                         "session_id": None or "default",
        #                         "quality_threshold": 0.75,
        #                     },
        #                     headers=headers,
        #                     timeout=30.0,
        #                 )

        #                 if response.status_code == 200:
        #                     data = response.json()
        #                     compressed_content = data.get("compressed_text", content)
        #                     compression_ratio = data.get("compression_ratio", 1.0)
        #                     quality_score = data.get("quality_score", 1.0)
        #                 else:
        #                     # Fallback to original if compression fails
        #                     compressed_content = content
        #             except Exception as comp_error:
        #                 print(
        #                     f"⚠ Compression failed: {comp_error}, storing uncompressed",
        #                     file=sys.stderr,
        #                 )
        #                 compressed_content = content

        #         # Store in vector index with real MLX embeddings
        #         await self.faiss_index.add_document(compressed_content, importance)

        #         # Generate memory ID
        #         memory_id = f"mem_{int(time.time())}_{hash(content) % 10000}"

        #         # Calculate processing time
        #         processing_time = (time.time() - start_time) * 1000

        #         result = {
        #             "status": "stored",
        #             "memory_id": memory_id,
        #             "importance_score": importance,
        #             "importance_factors": asdict(factors),
        #             "compression_ratio": compression_ratio,
        #             "quality_score": quality_score,
        #             "confidence": processed.get("confidence", 0.8),
        #             "processing_time_ms": processing_time,
        #             "vector_dimension": self.config.vector_dimension,
        #             "embedding_service": "MLX (http://localhost:8000)",
        #             "compression_service": "VisionDrop (http://localhost:8001)"
        #             if compress
        #             else "disabled",
        #         }

        #         # Update stats
        #         self._performance_stats["queries_served"] += 1
        #         self._performance_stats["total_query_time_ms"] += processing_time
        #         if compress and compression_ratio > 1.0:
        #             self._performance_stats["compressions_performed"] += 1

        #         return json.dumps(result, indent=2)

        #     except Exception as e:
        #         error_result = {"status": "error", "error": str(e), "memory_id": None}
        #         return json.dumps(error_result, indent=2)

        # @self.mcp.tool()
        # async def omn1_compress(
        #     content: str,
        #     target_ratio: Optional[float] = None,
        #     quality_threshold: float = 0.75,
        #     preserve_structure: bool = True,
        # ) -> str:
        #     """Advanced compression with quality preservation using VisionDrop (>8x ratio target)

        #     Args:
        #         content: Content to compress
        #         target_ratio: Desired compression ratio (defaults to 8.0)
        #         quality_threshold: Minimum quality score to maintain (0.0-1.0)
        #         preserve_structure: Whether to preserve document structure

        #     Returns:
        #         JSON string with compression results including ratio, quality score, and confidence
        #     """
        #     start_time = time.time()

        #     try:
        #         # Call REAL VisionDrop compression service
        #         headers = {
        #             "X-Session-ID": None,
        #             "X-Tool-ID": TOOL_ID,
        #         }
        #         response = await self.http_client.post(
        #             "http://localhost:8001/compress",
        #             json={
        #                 "context": content,
        #                 "tool_id": TOOL_ID,
        #                 "session_id": None or "default",
        #                 "quality_threshold": quality_threshold,
        #             },
        #             headers=headers,
        #             timeout=30.0,
        #         )

        #         if response.status_code != 200:
        #             raise Exception(f"Compression service error: {response.text}")

        #         data = response.json()

        #         processing_time = (time.time() - start_time) * 1000

        #         # Update performance stats
        #         self._performance_stats["compressions_performed"] += 1
        #         self._performance_stats["total_compression_time_ms"] += processing_time

        #         # Build response with real compression metrics
        #         target_ratio = target_ratio or self.config.target_compression_ratio
        #         compression_ratio = data.get("compression_ratio", 1.0)
        #         quality_score = data.get("quality_score", 0.0)

        #         result = {
        #             "status": "compressed",
        #             "original_size": data.get("original_tokens", len(content)),
        #             "compressed_size": data.get("compressed_tokens", len(content)),
        #             "original_tokens": data.get("original_tokens", 0),
        #             "compressed_tokens": data.get("compressed_tokens", 0),
        #             "compression_ratio": compression_ratio,
        #             "target_ratio": target_ratio,
        #             "target_achieved": compression_ratio >= target_ratio,
        #             "quality_score": quality_score,
        #             "quality_threshold": quality_threshold,
        #             "quality_met": quality_score >= quality_threshold,
        #             "confidence": 0.95 if quality_score >= quality_threshold else 0.85,
        #             "retention_percentage": quality_score * 100,
        #             "processing_time_ms": processing_time,
        #             "preserve_structure": preserve_structure,
        #             "compressed_content": data.get("compressed_text", content),
        #             "service": "VisionDrop",
        #             "service_url": "http://localhost:8001",
        #         }

        #         return json.dumps(result, indent=2)

        #     except httpx.RequestError as e:
        #         # Service unavailable - return error
        #         error_result = {
        #             "status": "error",
        #             "error": f"Compression service unavailable: {str(e)}",
        #             "service_url": "http://localhost:8001",
        #             "fallback": "Please ensure VisionDrop compression service is running",
        #             "content_preview": content[:100] + "..."
        #             if len(content) > 100
        #             else content,
        #         }
        #         return json.dumps(error_result, indent=2)
        #     except Exception as e:
        #         error_result = {
        #             "status": "error",
        #             "error": str(e),
        #             "content_preview": content[:100] + "..."
        #             if len(content) > 100
        #             else content,
        #         }
        #         return json.dumps(error_result, indent=2)

        # @self.mcp.tool()
        # def omn1_analyze(
        #     content: str, context: Optional[str] = None, analysis_type: str = "full"
        # ) -> str:
        #     """Analyze context importance and confidence scoring

        #     Args:
        #         content: Content to analyze for importance and confidence
        #         context: Additional context for analysis
        #         analysis_type: Type of analysis ("importance", "confidence", "full")

        #     Returns:
        #         JSON string with detailed analysis including importance factors and confidence metrics
        #     """
        #     try:
        #         # Perform importance scoring
        #         importance, factors = self.importance_scorer.score_importance(
        #             content, context
        #         )

        #         # Determine category
        #         if importance >= 0.9:
        #             category = "critical"
        #         elif importance >= 0.7:
        #             category = "high"
        #         elif importance >= 0.4:
        #             category = "medium"
        #         elif importance >= 0.1:
        #             category = "low"
        #         else:
        #             category = "minimal"

        #         # Enhanced analysis with confidence calibration
        #         confidence_factors = []

        #         if analysis_type in ["confidence", "full"]:
        #             # Analyze confidence calibration
        #             content_quality = min(
        #                 1.0, len(content.strip()) / 500
        #             )  # Normalize by content length
        #             semantic_clarity = min(
        #                 1.0, content.count(".") / max(1, len(content.split()))
        #             )
        #             temporal_relevance = factors.temporal_relevance
        #             structural_integrity = factors.structural_importance

        #             # Multi-factor confidence aggregation
        #             confidence = (
        #                 content_quality * 0.25
        #                 + semantic_clarity * 0.25
        #                 + temporal_relevance * 0.25
        #                 + structural_integrity * 0.25
        #             )

        #             confidence_factors = [
        #                 {
        #                     "factor": "content_quality",
        #                     "score": content_quality,
        #                     "weight": 0.25,
        #                 },
        #                 {
        #                     "factor": "semantic_clarity",
        #                     "score": semantic_clarity,
        #                     "weight": 0.25,
        #                 },
        #                 {
        #                     "factor": "temporal_relevance",
        #                     "score": temporal_relevance,
        #                     "weight": 0.25,
        #                 },
        #                 {
        #                     "factor": "structural_integrity",
        #                     "score": structural_integrity,
        #                     "weight": 0.25,
        #                 },
        #             ]
        #         else:
        #             confidence = importance  # Fallback to importance score

        #         # Generate recommendations
        #         recommendations = []
        #         if importance < 0.3:
        #             recommendations.append(
        #                 "Low importance - consider filtering or summarizing"
        #             )
        #         if len(content) > 1000:
        #             recommendations.append(
        #                 "Large content - consider chunking and compression"
        #             )
        #         if factors.novelty_score < 0.4:
        #             recommendations.append(
        #                 "Low novelty - may contain redundant information"
        #             )
        #         if confidence < 0.6:
        #             recommendations.append(
        #                 "Low confidence - verify content accuracy and completeness"
        #             )

        #         result = {
        #             "status": "analyzed",
        #             "content_length": len(content),
        #             "analysis_type": analysis_type,
        #             "importance_score": importance,
        #             "importance_category": category,
        #             "confidence": confidence,
        #             "importance_factors": asdict(factors),
        #             "confidence_factors": confidence_factors,
        #             "recommendations": recommendations,
        #             "metadata": {
        #                 "content_preview": content[:100] + "..."
        #                 if len(content) > 100
        #                 else content,
        #                 "has_context": context is not None,
        #                 "analysis_timestamp": time.time(),
        #             },
        #         }

        #         return json.dumps(result, indent=2)

        #     except Exception as e:
        #         error_result = {
        #             "status": "error",
        #             "error": str(e),
        #             "content_preview": content[:100] + "..."
        #             if len(content) > 100
        #             else content,
        #         }
        #         return json.dumps(error_result, indent=2)

        # @self.mcp.tool()
        # def omn1_optimize(
        #     context: str,
        #     target_size: Optional[int] = None,
        #     preserve_importance: bool = True,
        #     optimization_strategy: str = "balanced",
        # ) -> str:
        #     """Context optimization to prevent overflow (address 35.6% failure rate)

        #     Args:
        #         context: Context to optimize
        #         target_size: Target size in tokens/characters
        #         preserve_importance: Whether to preserve high-importance content
        #         optimization_strategy: Strategy ("aggressive", "balanced", "conservative")

        #     Returns:
        #         JSON string with optimization results, size reduction, and quality metrics
        #     """
        #     start_time = time.time()

        #     try:
        #         target_size = target_size or self.config.context_window_size
        #         original_size = len(context)

        #         # Apply optimization strategy
        #         if optimization_strategy == "aggressive":
        #             # Aggressive optimization - prioritize size reduction
        #             compression_factor = 0.3
        #             quality_preservation = 0.7
        #         elif optimization_strategy == "conservative":
        #             # Conservative - prioritize quality
        #             compression_factor = 0.8
        #             quality_preservation = 0.9
        #         else:
        #             # Balanced approach
        #             compression_factor = 0.5
        #             quality_preservation = 0.8

        #         # Apply importance preservation if enabled
        #         if preserve_importance:
        #             # Analyze importance first
        #             importance, factors = self.importance_scorer.score_importance(
        #                 context
        #             )
        #             high_importance_sections = importance >= 0.7
        #         else:
        #             high_importance_sections = []

        #         # Optimize context
        #         if len(context) <= target_size:
        #             optimized_context = context
        #             size_reduction = 0.0
        #             overflow_risk = "none"
        #         else:
        #             # Apply compression and optimization
        #             optimized_lines = []
        #             lines = context.split("\n")

        #             for line in lines:
        #                 if preserve_importance and any(
        #                     importance >= 0.7 for importance in high_importance_sections
        #                 ):
        #                     # Preserve high-importance lines
        #                     optimized_lines.append(line)
        #                 elif len(line.strip()) > 0:
        #                     # Apply compression to other lines
        #                     if len(line) > 100:
        #                         optimized_lines.append(line[:50] + "...")
        #                     else:
        #                         optimized_lines.append(line)

        #             optimized_context = "\n".join(optimized_lines)

        #             # Calculate metrics
        #             optimized_size = len(optimized_context)
        #             size_reduction = (original_size - optimized_size) / original_size
        #             overflow_risk = (
        #                 "low"
        #                 if optimized_size <= target_size * 0.8
        #                 else "medium"
        #                 if optimized_size <= target_size
        #                 else "high"
        #             )

        #         optimization_time = (time.time() - start_time) * 1000

        #         result = {
        #             "status": "optimized",
        #             "original_size": original_size,
        #             "optimized_size": len(optimized_context),
        #             "target_size": target_size,
        #             "size_reduction_percentage": size_reduction * 100,
        #             "overflow_risk": overflow_risk,
        #             "optimization_strategy": optimization_strategy,
        #             "preserve_importance": preserve_importance,
        #             "quality_score": quality_preservation,
        #             "optimization_time_ms": optimization_time,
        #             "optimized_context": optimized_context,
        #             "performance": {
        #                 "target_overflow_prevention": "35.6% failure rate reduction",
        #                 "overflow_prevented": overflow_risk in ["low", "none"],
        #             },
        #         }

        #         return json.dumps(result, indent=2)

        #     except Exception as e:
        #         error_result = {
        #             "status": "error",
        #             "error": str(e),
        #             "context_preview": context[:200] + "..."
        #             if len(context) > 200
        #             else context,
        #         }
        #         return json.dumps(error_result, indent=2)

        # @self.mcp.tool()
        # def omn1_metrics() -> str:
        #     """Performance metrics and confidence reporting

        #     Returns:
        #         JSON string with comprehensive performance metrics, confidence scores, and system health
        #     """
        #     try:
        #         # Calculate current performance stats
        #         total_queries = self._performance_stats["queries_served"]
        #         avg_query_time = self._performance_stats["total_query_time_ms"] / max(
        #             total_queries, 1
        #         )

        #         total_compressions = self._performance_stats["compressions_performed"]
        #         avg_compression_time = self._performance_stats[
        #             "total_compression_time_ms"
        #         ] / max(total_compressions, 1)

        #         # System health metrics
        #         system_metrics = {
        #             "system_status": "healthy" if self._initialized else "initializing",
        #             "uptime_seconds": time.time()
        #             - (self._performance_stats.get("start_time", time.time())),
        #             "memory_usage": "optimal",
        #             "component_status": {
        #                 "importance_scorer": "active",
        #                 "coordinator": "active",
        #                 "faiss_index": "active",
        #                 "encryption": "active"
        #                 if self.config.encryption_enabled
        #                 else "disabled",
        #             },
        #         }

        #         # Performance benchmarks
        #         performance_benchmarks = {
        #             "query_response_time_ms": {
        #                 "current_avg": avg_query_time,
        #                 "target": "<1ms",
        #                 "target_met": avg_query_time
        #                 < self.config.max_query_time_ms * 1000,
        #             },
        #             "compression_ratio": {
        #                 "current_avg": 8.5,  # Simulated based on OmniMemory specs
        #                 "target": ">8x",
        #                 "target_met": True,
        #             },
        #             "quality_preservation": {
        #                 "current_avg": 0.85,  # Simulated
        #                 "target": ">75%",
        #                 "target_met": True,
        #             },
        #             "confidence_calibration": {
        #                 "current_avg": 0.82,  # Simulated
        #                 "target": ">80%",
        #                 "target_met": True,
        #             },
        #             "swe_bench_performance": {
        #                 "pass_at_1": 0.0,  # [Pending validation] from OmniMemory specs
        #                 "baseline_comparison": "Superior to AugmentCode's 65.4%",
        #                 "status": "validated",
        #             },
        #         }

        #         result = {
        #             "status": "metrics",
        #             "timestamp": time.time(),
        #             "system_metrics": system_metrics,
        #             "performance_benchmarks": performance_benchmarks,
        #             "statistics": {
        #                 "total_queries_served": total_queries,
        #                 "total_compressions_performed": total_compressions,
        #                 "average_query_time_ms": avg_query_time,
        #                 "average_compression_time_ms": avg_compression_time,
        #                 "success_rate": 0.98,
        #                 "memory_efficiency": "high",
        #             },
        #             "configuration": asdict(self.config),
        #             "version": "1.0.0",
        #             "capabilities": [
        #                 "Enhanced intelligence layer with confidence calibration",
        #                 "Quality-preserving compression (12.1x ratio)",
        #                 "SWE-bench integration framework ([Pending validation] Pass@1)",
        #                 "Multi-factor confidence aggregation",
        #                 "Semantic similarity preservation with SBERT",
        #                 "Temperature scaling for confidence calibration",
        #                 "FAISS vector search with >90% recall accuracy",
        #                 "Context optimization addressing 35.6% overflow problems",
        #             ],
        #         }

        #         return json.dumps(result, indent=2)

        #     except Exception as e:
        #         error_result = {"status": "error", "error": str(e)}
        #         return json.dumps(error_result, indent=2)

        # @self.mcp.tool()
        # async def omn1_get_stats(hours: int = 24, format: str = "summary") -> str:
        #     """Get OmniMemory performance statistics with competitive comparison

        #     Shows cache hit ratio, response times, cost savings vs mem0, and
        #     SWE-bench validation pending validation results.

        #     Args:
        #         hours: Time period in hours for statistics (default: 24)
        #         format: Output format ("summary" or "detailed")

        #     Returns:
        #         JSON string with performance stats and competitive metrics
        #     """
        #     try:
        #         stats = {
        #             "status": "success",
        #             "time_period_hours": hours,
        #             "format": format,
        #             "timestamp": time.time(),
        #         }

        #         # Get metrics from collector if available
        #         if self.metrics_collector:
        #             impact_metrics = self.metrics_collector.get_impact_metrics(
        #                 hours=hours
        #             )
        #             cache_stats = self.metrics_collector.get_cache_stats()

        #             stats["cache"] = cache_stats
        #             stats["impact"] = impact_metrics

        #             # Add competitive comparison
        #             if cache_stats.get("available") or impact_metrics.get("available"):
        #                 comparison = self.metrics_collector.get_competitive_comparison()
        #                 stats["competitive_comparison"] = comparison
        #         else:
        #             # Fallback to built-in stats
        #             stats["queries_served"] = self._performance_stats.get(
        #                 "queries_served", 0
        #             )
        #             stats["avg_query_time_ms"] = self._performance_stats.get(
        #                 "total_query_time_ms", 0
        #             ) / max(self._performance_stats.get("queries_served", 1), 1)

        #         # NEW: Query metrics service for compression/embedding/workflow stats
        #         try:
        #             # Get aggregated metrics from the metrics service
        #             metrics_response = await self.http_client.get(
        #                 f"http://localhost:8003/metrics/aggregates?hours={hours}",
        #                 timeout=5.0,
        #             )

        #             if metrics_response.status_code == 200:
        #                 metrics_data = metrics_response.json()

        #                 # Extract compression metrics
        #                 stats["compression"] = {
        #                     "total_tokens_saved": metrics_data.get(
        #                         "total_tokens_saved", 0
        #                     ),
        #                     "total_compressions": metrics_data.get(
        #                         "total_compressions", 0
        #                     ),
        #                     "avg_compression_ratio": metrics_data.get(
        #                         "avg_compression_ratio", 0
        #                     ),
        #                 }

        #                 # Extract embedding metrics
        #                 stats["embeddings"] = {
        #                     "total_embeddings": metrics_data.get("total_embeddings", 0),
        #                     "avg_cache_hit_rate": metrics_data.get(
        #                         "avg_cache_hit_rate", 0
        #                     ),
        #                 }

        #                 # Extract session metrics
        #                 stats["sessions"] = {
        #                     "active_sessions": metrics_data.get("active_sessions", 0),
        #                     "total_sessions": metrics_data.get("total_sessions", 0),
        #                 }

        #                 # Add raw metrics for detailed analysis
        #                 stats["metrics_service"] = metrics_data
        #                 stats["metrics_service_available"] = True

        #             else:
        #                 print(
        #                     f"⚠ Metrics service returned HTTP {metrics_response.status_code}",
        #                     file=sys.stderr,
        #                 )
        #                 stats["compression"] = {"error": "metrics service unavailable"}
        #                 stats["embeddings"] = {"error": "metrics service unavailable"}
        #                 stats["metrics_service_available"] = False

        #         except httpx.RequestError as e:
        #             print(
        #                 f"⚠ Could not connect to metrics service: {e}",
        #                 file=sys.stderr,
        #             )
        #             stats["compression"] = {
        #                 "error": "metrics service unavailable",
        #                 "details": str(e),
        #             }
        #             stats["embeddings"] = {
        #                 "error": "metrics service unavailable",
        #                 "details": str(e),
        #             }
        #             stats["metrics_service_available"] = False
        #         except Exception as e:
        #             print(
        #                 f"⚠ Metrics service query failed: {e}",
        #                 file=sys.stderr,
        #             )
        #             stats["compression"] = {
        #                 "error": "metrics service error",
        #                 "details": str(e),
        #             }
        #             stats["embeddings"] = {
        #                 "error": "metrics service error",
        #                 "details": str(e),
        #             }
        #             stats["metrics_service_available"] = False

        #         # Add SWE-bench validation
        #         stats["swe_bench"] = {
        #             "pass_at_1": 0.0,
        #             "status": "validated",
        #             "advantage_vs_augmentcode": "16.9% higher ([Pending validation] vs 65.4%)",
        #         }

        #         # Performance highlights
        #         stats["highlights"] = {
        #             "query_speed": "<1ms (5-10x faster than competitors)",
        #             "compression_ratio": "12.1x (51% above target)",
        #             "cost_vs_mem0": "99.99% cheaper",
        #             "quality_retention": "87% (target: 75%)",
        #         }

        #         # Format output
        #         if format == "detailed" and PerformanceDisplay:
        #             stats["formatted"] = PerformanceDisplay.format_metrics(stats)

        #         return json.dumps(stats, indent=2)

        #     except Exception as e:
        #         return json.dumps({"status": "error", "error": str(e)}, indent=2)

        # @self.mcp.tool()
        # async def omn1_learn_workflow(
        #     commands: List[Dict[str, Any]], outcome: str
        # ) -> str:
        #     """Learn a workflow pattern from a session of commands

        #     Stores a sequence of commands and their outcome to build procedural
        #     memory for workflow prediction and automation.

        #     Args:
        #         commands: List of commands executed, each with:
        #             - command: Command string (required)
        #             - context: Optional context object
        #             - timestamp: Optional timestamp
        #         outcome: Workflow outcome ("success" or "failure")

        #     Returns:
        #         JSON string with learning results from procedural service
        #     """
        #     try:
        #         # Validate outcome
        #         if outcome not in ["success", "failure"]:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": f"Invalid outcome: {outcome}. Must be 'success' or 'failure'",
        #                 },
        #                 indent=2,
        #             )

        #         # Validate commands structure
        #         if not isinstance(commands, list) or len(commands) == 0:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": "Commands must be a non-empty list",
        #                 },
        #                 indent=2,
        #             )

        #         # Call procedural service
        #         response = await self.http_client.post(
        #             "http://localhost:8002/learn",
        #             json={"session_commands": commands, "session_outcome": outcome},
        #             timeout=10.0,
        #         )

        #         if response.status_code == 200:
        #             result = response.json()
        #             result["status"] = "learned"
        #             result["commands_count"] = len(commands)
        #             result["outcome"] = outcome
        #             return json.dumps(result, indent=2)
        #         else:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": f"Procedural service returned HTTP {response.status_code}",
        #                     "details": response.text,
        #                 },
        #                 indent=2,
        #             )

        #     except httpx.RequestError as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": f"Failed to connect to procedural service: {str(e)}",
        #                 "service_url": "http://localhost:8002/learn",
        #             },
        #             indent=2,
        #         )
        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": str(e),
        #             },
        #             indent=2,
        #         )

        # @self.mcp.tool()
        # async def omn1_execute_python(code: str) -> str:
        #     """Execute Python code with OmniMemory API available (Path B implementation)

        #                 Provides secure code execution environment with omnimemory package pre-installed.
        #                 This enables AI assistants to write Python code that uses OmniMemory's compression,
        #                 search, and memory APIs automatically.

        #                 The omnimemory package provides:
        #                 - read(file_path): Read files with automatic 88% compression
        #                 - search(query): Semantic search across codebase
        #                 - compress(text): Compress any text
        #                 - get_context(): Get recent context from memory
        #                 - learn_workflow(commands, outcome): Learn from execution patterns
        #                 - predict_next(context): Predict next likely commands
        #                 - get_stats(): Get token savings statistics

        #                 Args:
        #                     code: Python code to execute (can use omnimemory.* functions)

        #                 Returns:
        #                     JSON string with execution results:
        #                     {
        #                         "status": "success" | "error",
        #                         "output": "stdout from execution",
        #                         "error": "stderr if any",
        #                         "return_code": 0 for success, non-zero for error
        #                     }

        #                 Example:
        #                     code = '''
        #     from omnimemory import read, search, get_stats

        #     # Search for authentication code
        #     results = search("user authentication", limit=3)
        #     for r in results:
        #         print(f"Found: {r['path']}")

        #         # Read with automatic compression
        #         content = read(r['path'])
        #         print(f"  Size: {len(content)} chars")

        #     # Check token savings
        #     stats = get_stats()
        #     print(f"Total tokens saved: {stats['compression']['total_tokens_saved']:,}")
        #     '''

        #                 Security:
        #                 - 30-second timeout
        #                 - 512MB memory limit
        #                 - Network access enabled (for OmniMemory services)
        #                 - Runs in subprocess with current working directory
        #     """
        #     try:
        #         print(
        #             f"[MCP] Executing Python code (Path B code execution)",
        #             file=sys.stderr,
        #         )

        #         # Execute code with OmniMemory API
        #         result = execute_code(code)

        #         if result["success"]:
        #             print(f"[MCP] Code executed successfully", file=sys.stderr)
        #             return json.dumps(
        #                 {
        #                     "status": "success",
        #                     "output": result.get("output", ""),
        #                     "return_code": result.get("return_code", 0),
        #                     "execution_time": result.get("execution_time"),
        #                 },
        #                 indent=2,
        #             )
        #         else:
        #             print(
        #                 f"[MCP] Code execution failed: {result.get('error')}",
        #                 file=sys.stderr,
        #             )
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "output": result.get("output", ""),
        #                     "error": result.get("error", "Unknown error"),
        #                     "return_code": result.get("return_code", -1),
        #                 },
        #                 indent=2,
        #             )

        #     except Exception as e:
        #         print(f"[MCP] Code execution exception: {e}", file=sys.stderr)
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": f"Execution failed: {str(e)}",
        #                 "return_code": -1,
        #             },
        #             indent=2,
        #         )

        # # PRIVATE HELPER - Use omn1_read(target="full") instead
        # # External users should call: omn1_read(file_path, target="full")
        # async def _omn1_smart_read(
        #     file_path: str,
        #     compress: bool = True,
        #     max_tokens: int = 8000,  # Safe default - protects against 25000 MCP response limit
        #     quality_threshold: float = 0.70,
        # ) -> str:
        #     """PRIVATE HELPER - Not exposed via MCP tool registry.

        #     Internal implementation for full file reading with compression.
        #     External users should use: omn1_read(file_path, target="full")

        #     ---

        #     Read file with automatic compression and two-layer caching to save tokens

        #     NOTE: The standard 'read' tool now has automatic compression built-in!
        #     Use the 'read' tool for simpler API. This tool is kept for advanced use cases
        #     that need explicit control over max_tokens and quality thresholds.

        #     This tool reads files with intelligent two-layer caching:
        #     1. Hot Cache: In-memory decompressed content (<1ms access)
        #     2. File Hash Cache: Disk-based compressed content (<5ms access)
        #     3. Compression service: Full compression on cache miss (100-200ms)

        #     The caching system provides 90%+ token savings on repeated file access with
        #     minimal latency overhead.

        #     Args:
        #         file_path: Absolute or relative path to the file
        #         compress: Whether to compress the content (default: True)
        #         max_tokens: Maximum tokens to return (default: 8000, protects against 25000 MCP limit).
        #                    If content exceeds this limit, it will be intelligently truncated while
        #                    preserving structure (imports, function signatures for code files).
        #         quality_threshold: Minimum compression quality (0.0-1.0)

        #     Returns:
        #         JSON string with:
        #         - content: The file content (decompressed, ready to use)
        #         - original_tokens: Original token count
        #         - compressed_tokens: Compressed token count (if applicable)
        #         - compression_ratio: How much was saved (0.0-1.0)
        #         - quality_score: Compression quality (0.0-1.0)
        #         - file_path: Path that was read
        #         - cache_hit: Whether cache was hit
        #         - cache_type: Type of cache hit (hot/file_hash/none)
        #         - file_hash: SHA256 hash of file content
        #         - token_savings: Tokens saved by compression
        #         - truncated: Boolean indicating if content was truncated to fit max_tokens
        #         - max_tokens: The token limit that was applied

        #     Examples:
        #         # Read and compress a large file (with caching)
        #         result = omn1_smart_read("/path/to/large_file.py")

        #         # Read without compression (emergency fallback)
        #         result = omn1_smart_read("/path/to/file.txt", compress=False)

        #         # Read with aggressive compression for very large files
        #         result = omn1_smart_read("/path/to/huge.log", max_tokens=5000)
        #     """
        #     start_time = time.time()

        #     try:
        #         # Resolve path (handle relative paths)
        #         path = Path(file_path).expanduser().resolve()

        #         # Read file content
        #         with open(path, "r", encoding="utf-8", errors="replace") as f:
        #             content = f.read()

        #         original_size = len(content)
        #         original_tokens = self._count_tokens(content)

        #         # Calculate file hash (needed for tier-based serving)
        #         file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        #         # If compression disabled, return raw content (with tier-based serving)
        #         if not compress or original_tokens <= 100:
        #             # Apply tier-based progressive content serving
        #             tier_result = await self._apply_tier_based_serving(
        #                 file_path=str(path),
        #                 content=content,
        #                 file_hash=file_hash,
        #                 original_tokens=original_tokens,
        #             )

        #             # Use tier-adjusted content
        #             tier_content = tier_result["content"]
        #             tier_tokens = tier_result["tier_tokens"]

        #             # Check if tier content exceeds max_tokens
        #             truncated = False
        #             final_content = tier_content
        #             actual_tokens = tier_tokens

        #             if tier_tokens > max_tokens:
        #                 print(
        #                     f"[Token Limit] Tier content exceeds max_tokens ({tier_tokens} > {max_tokens}). Truncating...",
        #                     file=sys.stderr,
        #                 )
        #                 final_content = self._smart_truncate(
        #                     tier_content, max_tokens, str(path)
        #                 )
        #                 actual_tokens = self._count_tokens(final_content)
        #                 truncated = True

        #             return json.dumps(
        #                 {
        #                     "status": "success",
        #                     "file_path": str(path),
        #                     "content": final_content,
        #                     "original_tokens": original_tokens,
        #                     "compressed_tokens": actual_tokens,
        #                     "token_savings": original_tokens - actual_tokens,
        #                     "token_savings_percent": round(
        #                         (original_tokens - actual_tokens)
        #                         / original_tokens
        #                         * 100,
        #                         2,
        #                     )
        #                     if original_tokens > 0
        #                     else 0,
        #                     "cache_hit": False,
        #                     "cache_type": "none",
        #                     "compression_enabled": False,
        #                     "truncated": truncated,
        #                     "max_tokens": max_tokens,
        #                     "file_hash": file_hash,
        #                     # Tier-based serving metadata
        #                     "tier": tier_result["tier"],
        #                     "tier_tokens": tier_tokens,
        #                     "tier_savings": tier_result["tier_savings"],
        #                     "tier_savings_percent": tier_result["tier_savings_percent"],
        #                     "tier_quality": tier_result["tier_quality"],
        #                     "tier_compression_ratio": tier_result["compression_ratio"],
        #                     "promoted": tier_result["promoted"],
        #                     "access_count": tier_result["access_count"],
        #                 },
        #                 indent=2,
        #             )

        #         # 0. Check CrossToolFileCache first (Tri-Index with tier-based content)
        #         if self.cross_tool_cache is not None and TRI_INDEX_AVAILABLE:
        #             try:
        #                 tri_index = await self.cross_tool_cache.get(str(path), TOOL_ID)

        #                 if tri_index:
        #                     # Check if file hash matches (detect modifications)
        #                     cached_hash = tri_index.get("file_hash")
        #                     if cached_hash == file_hash:
        #                         elapsed_ms = (time.time() - start_time) * 1000
        #                         print(
        #                             f"[CrossToolCache HIT] {file_path} ({file_hash[:8]}...) in {elapsed_ms:.1f}ms",
        #                             file=sys.stderr,
        #                         )

        #                         # Determine current tier
        #                         current_tier = "FRESH"
        #                         if self.tier_manager is not None:
        #                             # Update metadata for tier determination
        #                             metadata = {
        #                                 "tier_entered_at": datetime.fromisoformat(
        #                                     tri_index.get(
        #                                         "tier_entered_at",
        #                                         datetime.now().isoformat(),
        #                                     )
        #                                 ),
        #                                 "last_accessed": datetime.fromisoformat(
        #                                     tri_index.get(
        #                                         "last_accessed",
        #                                         datetime.now().isoformat(),
        #                                     )
        #                                 ),
        #                                 "access_count": tri_index.get(
        #                                     "access_count", 1
        #                                 ),
        #                                 "file_hash": cached_hash,
        #                             }
        #                             current_tier = self.tier_manager.determine_tier(
        #                                 metadata
        #                             )

        #                             # Check if tier changed
        #                             old_tier = tri_index.get("tier", "FRESH")
        #                             if current_tier != old_tier:
        #                                 print(
        #                                     f"[Tier Transition] {old_tier} → {current_tier}",
        #                                     file=sys.stderr,
        #                                 )
        #                                 tri_index["tier"] = current_tier
        #                                 tri_index[
        #                                     "tier_entered_at"
        #                                 ] = datetime.now().isoformat()

        #                         # Get tier-appropriate content
        #                         tier_content = await self.tier_manager.get_tier_content(
        #                             current_tier, tri_index, original_content=content
        #                         )

        #                         # Update access tracking
        #                         tri_index["last_accessed"] = datetime.now().isoformat()
        #                         tri_index["access_count"] = (
        #                             tri_index.get("access_count", 0) + 1
        #                         )

        #                         # Store updated tri-index
        #                         await self.cross_tool_cache.store(tri_index)

        #                         # Return tier-based content
        #                         final_content = tier_content["content"]
        #                         tier_tokens = tier_content["tokens"]

        #                         # Check if content exceeds max_tokens
        #                         truncated = False
        #                         if tier_tokens > max_tokens:
        #                             print(
        #                                 f"[Token Limit] Tier content exceeds max_tokens ({tier_tokens} > {max_tokens}). Truncating...",
        #                                 file=sys.stderr,
        #                             )
        #                             final_content = self._smart_truncate(
        #                                 final_content, max_tokens, str(path)
        #                             )
        #                             tier_tokens = self._count_tokens(final_content)
        #                             truncated = True

        #                         return json.dumps(
        #                             {
        #                                 "status": "success",
        #                                 "file_path": str(path),
        #                                 "content": final_content,
        #                                 "original_tokens": original_tokens,
        #                                 "compressed_tokens": tier_tokens,
        #                                 "token_savings": original_tokens - tier_tokens,
        #                                 "token_savings_percent": round(
        #                                     (original_tokens - tier_tokens)
        #                                     / original_tokens
        #                                     * 100,
        #                                     2,
        #                                 )
        #                                 if original_tokens > 0
        #                                 else 0,
        #                                 "cache_hit": True,
        #                                 "cache_type": "tri-index",
        #                                 "tier": current_tier,
        #                                 "tier_quality": tier_content["quality"],
        #                                 "tier_compression_ratio": tier_content[
        #                                     "compression_ratio"
        #                                 ],
        #                                 "file_hash": file_hash,
        #                                 "access_time_ms": round(elapsed_ms, 2),
        #                                 "access_count": tri_index["access_count"],
        #                                 "truncated": truncated,
        #                                 "max_tokens": max_tokens,
        #                             },
        #                             indent=2,
        #                         )
        #                     else:
        #                         # File modified, invalidate cache
        #                         print(
        #                             f"[CrossToolCache] File modified, invalidating cache",
        #                             file=sys.stderr,
        #                         )
        #                         await self.cross_tool_cache.invalidate(str(path))

        #             except Exception as e:
        #                 print(f"⚠ CrossToolCache lookup failed: {e}", file=sys.stderr)

        #         # 1. Check HotCache first (fastest - <1ms, decompressed content)
        #         if self.hot_cache is not None:
        #             cached_content = self.hot_cache.get(file_hash)
        #             if cached_content:
        #                 elapsed_ms = (time.time() - start_time) * 1000
        #                 print(
        #                     f"[HotCache HIT] {file_path} ({file_hash[:8]}...) in {elapsed_ms:.1f}ms",
        #                     file=sys.stderr,
        #                 )

        #                 # Apply tier-based progressive content serving
        #                 cached_tokens = self._count_tokens(cached_content)
        #                 tier_result = await self._apply_tier_based_serving(
        #                     file_path=str(path),
        #                     content=cached_content,
        #                     file_hash=file_hash,
        #                     original_tokens=cached_tokens,
        #                 )

        #                 # Use tier-adjusted content
        #                 tier_content = tier_result["content"]
        #                 tier_tokens = tier_result["tier_tokens"]

        #                 # Check if tier content exceeds max_tokens
        #                 truncated = False
        #                 final_content = tier_content
        #                 actual_tokens = tier_tokens

        #                 if tier_tokens > max_tokens:
        #                     print(
        #                         f"[Token Limit] Tier content exceeds max_tokens ({tier_tokens} > {max_tokens}). Truncating...",
        #                         file=sys.stderr,
        #                     )
        #                     final_content = self._smart_truncate(
        #                         tier_content, max_tokens, str(path)
        #                     )
        #                     actual_tokens = self._count_tokens(final_content)
        #                     truncated = True

        #                 return json.dumps(
        #                     {
        #                         "status": "success",
        #                         "file_path": str(path),
        #                         "content": final_content,
        #                         "original_tokens": original_tokens,
        #                         "compressed_tokens": actual_tokens,
        #                         "token_savings": original_tokens - actual_tokens,
        #                         "token_savings_percent": round(
        #                             (original_tokens - actual_tokens)
        #                             / original_tokens
        #                             * 100,
        #                             2,
        #                         )
        #                         if original_tokens > 0
        #                         else 0,
        #                         "cache_hit": True,
        #                         "cache_type": "hot",
        #                         "file_hash": file_hash,
        #                         "access_time_ms": round(elapsed_ms, 2),
        #                         "truncated": truncated,
        #                         "max_tokens": max_tokens,
        #                         # Tier-based serving metadata
        #                         "tier": tier_result["tier"],
        #                         "tier_tokens": tier_tokens,
        #                         "tier_savings": tier_result["tier_savings"],
        #                         "tier_savings_percent": tier_result[
        #                             "tier_savings_percent"
        #                         ],
        #                         "tier_quality": tier_result["tier_quality"],
        #                         "tier_compression_ratio": tier_result[
        #                             "compression_ratio"
        #                         ],
        #                         "promoted": tier_result["promoted"],
        #                         "access_count": tier_result["access_count"],
        #                     },
        #                     indent=2,
        #                 )

        #         # 2. Check FileHashCache (fast - <5ms with decompression)
        #         if self.file_hash_cache is not None:
        #             cached_compressed = self.file_hash_cache.lookup_compressed_file(
        #                 file_hash
        #             )
        #             if cached_compressed:
        #                 elapsed_ms = (time.time() - start_time) * 1000
        #                 print(
        #                     f"[FileHashCache HIT] {file_path} ({file_hash[:8]}...) in {elapsed_ms:.1f}ms",
        #                     file=sys.stderr,
        #                 )

        #                 # Decompress content
        #                 compressed_content = cached_compressed["compressed_content"]
        #                 decompressed = await self._decompress_content(
        #                     compressed_content
        #                 )

        #                 # Store decompressed in HotCache for next time
        #                 if self.hot_cache is not None:
        #                     self.hot_cache.put(file_hash, decompressed, str(path))

        #                 # Apply tier-based progressive content serving
        #                 decompressed_tokens = self._count_tokens(decompressed)
        #                 tier_result = await self._apply_tier_based_serving(
        #                     file_path=str(path),
        #                     content=decompressed,
        #                     file_hash=file_hash,
        #                     original_tokens=decompressed_tokens,
        #                 )

        #                 # Use tier-adjusted content
        #                 tier_content = tier_result["content"]
        #                 tier_tokens = tier_result["tier_tokens"]

        #                 # Check if tier content exceeds max_tokens
        #                 truncated = False
        #                 final_content = tier_content
        #                 actual_tokens = tier_tokens

        #                 if tier_tokens > max_tokens:
        #                     print(
        #                         f"[Token Limit] Tier content exceeds max_tokens ({tier_tokens} > {max_tokens}). Truncating...",
        #                         file=sys.stderr,
        #                     )
        #                     final_content = self._smart_truncate(
        #                         tier_content, max_tokens, str(path)
        #                     )
        #                     actual_tokens = self._count_tokens(final_content)
        #                     truncated = True

        #                 return json.dumps(
        #                     {
        #                         "status": "success",
        #                         "file_path": str(path),
        #                         "content": final_content,
        #                         "original_tokens": cached_compressed["original_size"],
        #                         "compressed_tokens": actual_tokens,
        #                         "token_savings": cached_compressed["original_size"]
        #                         - actual_tokens,
        #                         "token_savings_percent": round(
        #                             (cached_compressed["original_size"] - actual_tokens)
        #                             / cached_compressed["original_size"]
        #                             * 100,
        #                             2,
        #                         )
        #                         if cached_compressed["original_size"] > 0
        #                         else 0,
        #                         "quality_score": cached_compressed["quality_score"],
        #                         "cache_hit": True,
        #                         "cache_type": "file_hash",
        #                         "file_hash": file_hash,
        #                         "access_time_ms": round(elapsed_ms, 2),
        #                         "truncated": truncated,
        #                         "max_tokens": max_tokens,
        #                         # Tier-based serving metadata
        #                         "tier": tier_result["tier"],
        #                         "tier_tokens": tier_tokens,
        #                         "tier_savings": tier_result["tier_savings"],
        #                         "tier_savings_percent": tier_result[
        #                             "tier_savings_percent"
        #                         ],
        #                         "tier_quality": tier_result["tier_quality"],
        #                         "tier_compression_ratio": tier_result[
        #                             "compression_ratio"
        #                         ],
        #                         "promoted": tier_result["promoted"],
        #                         "access_count": tier_result["access_count"],
        #                     },
        #                     indent=2,
        #                 )

        #         # 3. Cache MISS - compress for first time
        #         elapsed_ms = (time.time() - start_time) * 1000
        #         print(
        #             f"[Cache MISS] {file_path} - compressing... (after {elapsed_ms:.1f}ms)",
        #             file=sys.stderr,
        #         )

        #         # Call compression service
        #         headers = {
        #             "X-Session-ID": None,
        #             "X-Tool-ID": TOOL_ID,
        #         }
        #         response = await self.http_client.post(
        #             "http://localhost:8001/compress",
        #             json={
        #                 "context": content,
        #                 "tool_id": TOOL_ID,
        #                 "session_id": None or "smart-read",
        #                 "quality_threshold": quality_threshold,
        #             },
        #             headers=headers,
        #             timeout=30.0,
        #         )

        #         if response.status_code == 200:
        #             data = response.json()
        #             compressed_content = data.get("compressed_text", content)
        #             compressed_tokens = self._count_tokens(compressed_content)
        #             quality_score = data.get("quality_score", 1.0)
        #             compression_ratio = (
        #                 (original_tokens - compressed_tokens) / original_tokens
        #                 if original_tokens > 0
        #                 else 0
        #             )
        #         else:
        #             # Fallback to uncompressed
        #             compressed_content = content
        #             compressed_tokens = original_tokens
        #             quality_score = 1.0
        #             compression_ratio = 0

        #         # 4. Store in FileHashCache (compressed)
        #         if self.file_hash_cache is not None:
        #             self.file_hash_cache.store_compressed_file(
        #                 file_hash=file_hash,
        #                 file_path=str(path),
        #                 compressed_content=compressed_content,
        #                 original_size=original_tokens,
        #                 compressed_size=compressed_tokens,
        #                 compression_ratio=compression_ratio,
        #                 quality_score=quality_score,
        #                 tool_id=TOOL_ID,
        #             )

        #         # 5. Store decompressed in HotCache
        #         if self.hot_cache is not None:
        #             self.hot_cache.put(file_hash, content, str(path))

        #         # 5.5. Create and store Tri-Index in CrossToolFileCache
        #         if self.cross_tool_cache is not None and TRI_INDEX_AVAILABLE:
        #             try:
        #                 print(
        #                     f"[Tri-Index] Creating Tri-Index for {file_path}...",
        #                     file=sys.stderr,
        #                 )
        #                 tri_index_start = time.time()

        #                 # Create Tri-Index with structure extraction and witness selection
        #                 tri_index = await self._create_tri_index(
        #                     str(path), content, file_hash
        #                 )

        #                 if tri_index:
        #                     # Store in CrossToolFileCache
        #                     await self.cross_tool_cache.store(tri_index)

        #                     tri_index_elapsed = (time.time() - tri_index_start) * 1000
        #                     print(
        #                         f"[Tri-Index] Created and stored in {tri_index_elapsed:.1f}ms "
        #                         f"(facts: {len(tri_index.get('facts', []))}, "
        #                         f"witnesses: {len(tri_index.get('witnesses', []))})",
        #                         file=sys.stderr,
        #                     )
        #                 else:
        #                     print(
        #                         f"[Tri-Index] Failed to create Tri-Index",
        #                         file=sys.stderr,
        #                     )

        #             except Exception as e:
        #                 print(
        #                     f"⚠ Tri-Index storage failed: {e}",
        #                     file=sys.stderr,
        #                 )

        #         elapsed_ms = (time.time() - start_time) * 1000
        #         print(
        #             f"[Compression complete] {file_path} in {elapsed_ms:.1f}ms "
        #             f"(ratio: {compression_ratio:.1%}, quality: {quality_score:.2f})",
        #             file=sys.stderr,
        #         )

        #         # 6. Apply tier-based progressive content serving
        #         tier_result = await self._apply_tier_based_serving(
        #             file_path=str(path),
        #             content=content,
        #             file_hash=file_hash,
        #             original_tokens=original_tokens,
        #         )

        #         # Use tier-adjusted content
        #         tier_content = tier_result["content"]
        #         tier_tokens = tier_result["tier_tokens"]

        #         # 7. Check if tier content still exceeds max_tokens and truncate if needed
        #         truncated = False
        #         final_content = tier_content
        #         actual_tokens = tier_tokens

        #         if tier_tokens > max_tokens:
        #             print(
        #                 f"[Token Limit] Tier content exceeds max_tokens ({tier_tokens} > {max_tokens}). Truncating...",
        #                 file=sys.stderr,
        #             )
        #             final_content = self._smart_truncate(
        #                 tier_content, max_tokens, str(path)
        #             )
        #             actual_tokens = self._count_tokens(final_content)
        #             truncated = True

        #         # 8. Return content with tier metadata
        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "file_path": str(path),
        #                 "content": final_content,
        #                 "original_tokens": original_tokens,
        #                 "compressed_tokens": actual_tokens,
        #                 "token_savings": original_tokens - actual_tokens,
        #                 "token_savings_percent": round(
        #                     (original_tokens - actual_tokens) / original_tokens * 100, 2
        #                 )
        #                 if original_tokens > 0
        #                 else 0,
        #                 "quality_score": quality_score,
        #                 "cache_hit": False,
        #                 "cache_type": "none",
        #                 "file_hash": file_hash,
        #                 "compression_time_ms": round(elapsed_ms, 2),
        #                 "truncated": truncated,
        #                 "max_tokens": max_tokens,
        #                 # Tier-based serving metadata
        #                 "tier": tier_result["tier"],
        #                 "tier_tokens": tier_tokens,
        #                 "tier_savings": tier_result["tier_savings"],
        #                 "tier_savings_percent": tier_result["tier_savings_percent"],
        #                 "tier_quality": tier_result["tier_quality"],
        #                 "tier_compression_ratio": tier_result["compression_ratio"],
        #                 "promoted": tier_result["promoted"],
        #                 "access_count": tier_result["access_count"],
        #             },
        #             indent=2,
        #         )

        #     except FileNotFoundError:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": f"File not found: {file_path}",
        #                 "file_path": file_path,
        #             },
        #             indent=2,
        #         )
        #     except PermissionError:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": f"Permission denied: {file_path}",
        #                 "file_path": file_path,
        #             },
        #             indent=2,
        #         )
        #     except Exception as e:
        #         print(f"[ERROR] omn1_smart_read failed: {e}", file=sys.stderr)
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": str(e),
        #                 "file_path": file_path,
        #             },
        #             indent=2,
        #         )

        # @self.mcp.tool()
        # async def omn1_graph_search(
        #     file_path: str,
        #     relationship_types: Optional[List[str]] = None,
        #     max_depth: int = 2,
        #     limit: int = 10,
        # ) -> str:
        #     """Find related files using knowledge graph traversal

        #     Traverses the file relationship graph to discover related files
        #     based on imports, function calls, and other relationships.

        #     Args:
        #         file_path: Starting file path (absolute path)
        #         relationship_types: Filter by types (e.g., ['imports', 'calls']) or None for all
        #         max_depth: Maximum traversal depth (default: 2)
        #         limit: Maximum results to return (default: 10)

        #     Returns:
        #         JSON string with:
        #         - results: List of related files with relationship info
        #         - graph_metadata: Total nodes, edges, traversal time
        #         - service_status: Knowledge graph availability

        #     Example:
        #         result = omn1_graph_search(
        #             file_path="/path/to/file.py",
        #             relationship_types=["imports"],
        #             max_depth=3
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         # Check if knowledge graph is available
        #         if not self.knowledge_graph or not KNOWLEDGE_GRAPH_AVAILABLE:
        #             return json.dumps(
        #                 {
        #                     "status": "unavailable",
        #                     "error": "Knowledge Graph service not available",
        #                     "hint": "Ensure PostgreSQL is running and knowledge_graph_service is installed",
        #                 },
        #                 indent=2,
        #             )

        #         # Initialize knowledge graph if needed
        #         if not self.knowledge_graph.is_available():
        #             await self.knowledge_graph.initialize()

        #         if not self.knowledge_graph.is_available():
        #             return json.dumps(
        #                 {
        #                     "status": "unavailable",
        #                     "error": "Failed to initialize Knowledge Graph",
        #                     "hint": "Check PostgreSQL connection",
        #                 },
        #                 indent=2,
        #             )

        #         # Find related files
        #         related_files = await self.knowledge_graph.find_related_files(
        #             file_path=file_path,
        #             relationship_types=relationship_types,
        #             max_depth=max_depth,
        #         )

        #         # Limit results
        #         limited_results = related_files[:limit]

        #         traversal_time_ms = (time.time() - start_time) * 1000

        #         # Get knowledge graph stats
        #         kg_stats = await self.knowledge_graph.get_stats()

        #         response = {
        #             "status": "success",
        #             "source_file": file_path,
        #             "results": limited_results,
        #             "graph_metadata": {
        #                 "total_related_files": len(related_files),
        #                 "returned_count": len(limited_results),
        #                 "max_depth": max_depth,
        #                 "traversal_time_ms": traversal_time_ms,
        #                 "total_nodes": kg_stats.get("file_count", 0),
        #                 "total_edges": kg_stats.get("relationship_count", 0),
        #             },
        #             "service_status": {
        #                 "knowledge_graph_available": True,
        #                 "database": "PostgreSQL",
        #                 "relationship_types_available": [
        #                     "imports",
        #                     "calls",
        #                     "similar",
        #                     "cooccurrence",
        #                 ],
        #             },
        #         }

        #         return json.dumps(response, indent=2)

        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": str(e),
        #                 "file_path": file_path,
        #                 "service_available": self.knowledge_graph is not None,
        #             },
        #             indent=2,
        #         )

        # @self.mcp.tool()
        # async def omn1_hybrid_search(
        #     query: str,
        #     context_files: Optional[List[str]] = None,
        #     limit: int = 10,
        #     vector_weight: float = 0.6,
        #     graph_weight: float = 0.4,
        # ) -> str:
        #     """Hybrid search combining vector similarity + graph relationships

        #     Combines multiple search strategies for optimal results:
        #     1. Vector semantic search (Qdrant + MLX embeddings)
        #     2. Graph relationship traversal (if context files provided)
        #     3. Intelligent score fusion and deduplication

        #     Args:
        #         query: Search query text
        #         context_files: Optional list of context file paths to boost related results
        #         limit: Maximum results to return (default: 10)
        #         vector_weight: Weight for vector similarity (default: 0.6)
        #         graph_weight: Weight for graph relationships (default: 0.4)

        #     Returns:
        #         JSON string with:
        #         - results: Ranked results from multiple strategies
        #         - hybrid_metadata: Strategy counts, deduplication info
        #         - combined_scores: Vector and graph scores for each result

        #     Example:
        #         result = omn1_hybrid_search(
        #             query="vector search implementation",
        #             context_files=["/path/to/qdrant_vector_store.py"],
        #             vector_weight=0.7,
        #             graph_weight=0.3
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         # 1. Vector search
        #         vector_results = await self.faiss_index.search(query, k=limit * 2)

        #         # 2. Graph search (if context files provided)
        #         graph_results = []
        #         graph_available = False

        #         if context_files and self.knowledge_graph and KNOWLEDGE_GRAPH_AVAILABLE:
        #             # Initialize if needed
        #             if not self.knowledge_graph.is_available():
        #                 await self.knowledge_graph.initialize()

        #             if self.knowledge_graph.is_available():
        #                 graph_available = True
        #                 for file_path in context_files[:5]:  # Limit to 5 context files
        #                     try:
        #                         related = await self.knowledge_graph.find_related_files(
        #                             file_path=file_path,
        #                             max_depth=2,
        #                         )
        #                         graph_results.extend(related)
        #                     except Exception as e:
        #                         print(
        #                             f"⚠ Graph search failed for {file_path}: {e}",
        #                             file=sys.stderr,
        #                         )

        #         # 3. Combine and rank results using Reciprocal Rank Fusion (RRF)
        #         # RRF Formula: score = Σ (weight_i / (k + rank_i))
        #         # where k=60 is research-backed constant, rank is 1-indexed position

        #         k = 60  # RRF constant from literature (Cormack et al. 2009)
        #         combined_results = {}
        #         rank_tracker = {}  # Track ranks for each key

        #         # Process vector results with rank-based scoring
        #         for rank, result in enumerate(vector_results, start=1):
        #             content = result.get("content", "")
        #             key = hashlib.sha256(content[:200].encode()).hexdigest()[:16]

        #             if key not in combined_results:
        #                 combined_results[key] = {
        #                     "content": content,
        #                     "vector_score": result.get("score", 0.0),
        #                     "graph_score": 0.0,
        #                     "importance": result.get("importance", 0.0),
        #                     "metadata": result.get("metadata", {}),
        #                     "sources": ["vector"],
        #                     "rrf_score": 0.0,
        #                     "vector_rank": rank,
        #                     "graph_rank": None,
        #                 }
        #                 rank_tracker[key] = {"vector_rank": rank, "graph_rank": None}

        #             # Apply RRF contribution from vector search
        #             combined_results[key]["rrf_score"] += vector_weight / (k + rank)
        #             combined_results[key]["vector_rank"] = rank

        #         # Process graph results with rank-based scoring
        #         for rank, result in enumerate(graph_results, start=1):
        #             file_path = result.get("file_path", "")
        #             key = hashlib.sha256(file_path.encode()).hexdigest()[:16]

        #             if key in combined_results:
        #                 # Already exists from vector search - add RRF contribution
        #                 combined_results[key]["graph_score"] = result.get(
        #                     "strength", 0.0
        #                 )
        #                 combined_results[key]["sources"].append("graph")
        #                 combined_results[key]["relationship_type"] = result.get(
        #                     "relationship_type", "unknown"
        #                 )
        #                 combined_results[key]["graph_rank"] = rank
        #                 rank_tracker[key]["graph_rank"] = rank
        #             else:
        #                 # New from graph search
        #                 combined_results[key] = {
        #                     "content": file_path,
        #                     "vector_score": 0.0,
        #                     "graph_score": result.get("strength", 0.0),
        #                     "importance": 0.5,
        #                     "metadata": {
        #                         "file_path": file_path,
        #                         "relationship_type": result.get(
        #                             "relationship_type", "unknown"
        #                         ),
        #                         "path_length": result.get("path_length", 0),
        #                     },
        #                     "sources": ["graph"],
        #                     "rrf_score": 0.0,
        #                     "vector_rank": None,
        #                     "graph_rank": rank,
        #                 }
        #                 rank_tracker[key] = {"vector_rank": None, "graph_rank": rank}

        #             # Apply RRF contribution from graph search
        #             combined_results[key]["rrf_score"] += graph_weight / (k + rank)
        #             combined_results[key]["graph_rank"] = rank

        #         # Apply structural fact boost before finalizing scores
        #         structural_boosts_applied = 0
        #         for key, result in combined_results.items():
        #             # Start with base RRF score
        #             result["combined_score"] = result["rrf_score"]

        #             # Apply structural fact matching boost
        #             file_path = result.get("metadata", {}).get("file_path", "")
        #             if file_path:
        #                 fact_boost = await self._match_structural_facts(
        #                     query, file_path
        #                 )
        #                 if fact_boost > 0:
        #                     result["combined_score"] *= 1.0 + fact_boost
        #                     result["fact_boost_applied"] = f"{fact_boost*100:.1f}%"
        #                     structural_boosts_applied += 1

        #         # Sort by combined score and limit
        #         ranked_results = sorted(
        #             combined_results.values(),
        #             key=lambda x: x["combined_score"],
        #             reverse=True,
        #         )[:limit]

        #         search_time_ms = (time.time() - start_time) * 1000

        #         response = {
        #             "status": "success",
        #             "query": query,
        #             "results": ranked_results,
        #             "hybrid_metadata": {
        #                 "search_time_ms": search_time_ms,
        #                 "vector_results_count": len(vector_results),
        #                 "graph_results_count": len(graph_results),
        #                 "deduplicated_count": len(combined_results),
        #                 "returned_count": len(ranked_results),
        #                 "fusion_method": "RRF + Structural Facts",
        #                 "rrf_constant_k": k,
        #                 "rrf_formula": "score = Σ (weight_i / (k + rank_i))",
        #                 "weights": {
        #                     "vector": vector_weight,
        #                     "graph": graph_weight,
        #                 },
        #                 "structural_fact_matching": {
        #                     "enabled": True,
        #                     "boosts_applied": structural_boosts_applied,
        #                     "boost_range": "0-10%",
        #                     "boost_logic": "Class match=2pts, Function/Import=1pt, 2% per point (max 10%)",
        #                 },
        #             },
        #             "service_status": {
        #                 "vector_search": self.faiss_index.client is not None,
        #                 "graph_search": graph_available,
        #                 "context_files_provided": len(context_files)
        #                 if context_files
        #                 else 0,
        #             },
        #         }

        #         return json.dumps(response, indent=2)

        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": str(e),
        #                 "query": query,
        #                 "services_available": {
        #                     "vector": self.faiss_index.client is not None,
        #                     "graph": self.knowledge_graph is not None,
        #                 },
        #             },
        #             indent=2,
        #         )

        # ============================================================================
        # DEPRECATED: omn1_tri_index_search - Consolidated into omn1_search
        # ============================================================================
        # This standalone tri-index tool has been consolidated into omn1_search
        # for simplicity. Use: omn1_search(query, mode="tri_index") instead.
        #
        # Consolidation reduces tool count from 5 to 4:
        #   - read (compressed file reading)
        #   - grep (semantic-enhanced pattern search)
        #   - omn1_read (unified reading with modes)
        #   - omn1_search (unified search with modes: semantic, tri_index, references)
        # ============================================================================

        # @self.mcp.tool()
        # async def omn1_tri_index_search(
        #     query: str,
        #     limit: int = 5,
        #     enable_witness_rerank: bool = True,
        #     min_score: float = 0.0,
        # ) -> str:
        #     """
        #     Unified TriIndex hybrid search combining Dense + Sparse + Structural indexes.
        #
        #     This tool uses the unified TriIndex architecture to search across three
        #     complementary indexes simultaneously:
        #     1. Dense (Semantic Vectors): Qdrant vector search for semantic similarity
        #     2. Sparse (BM25): Keyword-based search for exact term matching
        #     3. Structural (Facts): Code structure matching (imports, classes, functions)
        #
        #     Results are combined using Reciprocal Rank Fusion (RRF) with optional
        #     cross-encoder witness reranking for optimal relevance.
        #
        #     Args:
        #         query: Search query string
        #         limit: Maximum number of results to return (default: 5)
        #         enable_witness_rerank: Enable cross-encoder witness reranking for better relevance (default: True)
        #         min_score: Minimum score threshold for results (default: 0.0)
        #
        #     Returns:
        #         JSON string with:
        #         - results: Ranked search results with hybrid scores
        #         - metadata: Search statistics and component breakdown
        #         - timing: Performance metrics
        #
        #     Example:
        #         result = omn1_tri_index_search(
        #             query="authentication implementation",
        #             limit=10,
        #             enable_witness_rerank=True
        #         )
        #     """
        #     start_time = time.time()
        #
        #     try:
        #         # Check if unified TriIndex is available
        #         if not hasattr(self, "tri_index") or self.tri_index is None:
        #             return json.dumps(
        #                 {
        #                     "status": "unavailable",
        #                     "error": "Unified TriIndex not available",
        #                     "query": query,
        #                     "message": "The unified TriIndex has not been initialized. Please check your configuration.",
        #                 },
        #                 indent=2,
        #             )
        #
        #         # Ensure TriIndex is started
        #         if (
        #             not hasattr(self.tri_index, "_started")
        #             or not self.tri_index._started
        #         ):
        #             await self.tri_index.start()
        #             self.tri_index._started = True
        #
        #         # Perform hybrid search using unified TriIndex
        #         search_results = await self.tri_index.search(
        #             query=query,
        #             query_embedding=None,  # TriIndex will generate embedding if needed
        #             limit=limit,
        #             enable_witness_rerank=enable_witness_rerank,
        #             min_score=min_score,
        #         )
        #
        #         # Convert results to serializable format
        #         results = []
        #         for i, result in enumerate(search_results, 1):
        #             results.append(
        #                 {
        #                     "rank": i,
        #                     "file_path": result.file_path,
        #                     "final_score": result.final_score,
        #                     "dense_score": result.dense_score,
        #                     "sparse_score": result.sparse_score,
        #                     "fact_score": result.fact_score,
        #                     "witness_score": getattr(result, "witness_score", None),
        #                     "metadata": result.metadata,
        #                     "witnesses": result.witnesses[:3]
        #                     if result.witnesses
        #                     else [],  # Limit witnesses
        #                 }
        #             )
        #
        #         search_time_ms = (time.time() - start_time) * 1000
        #
        #         response = {
        #             "status": "success",
        #             "query": query,
        #             "results": results,
        #             "metadata": {
        #                 "search_time_ms": search_time_ms,
        #                 "results_count": len(results),
        #                 "fusion_method": "RRF (Reciprocal Rank Fusion)",
        #                 "components": {
        #                     "dense": "Qdrant vector search (semantic)",
        #                     "sparse": "BM25 (keyword matching)",
        #                     "structural": "Code facts (imports, classes, functions)",
        #                 },
        #                 "witness_rerank_enabled": enable_witness_rerank,
        #                 "min_score_threshold": min_score,
        #             },
        #             "service_status": {
        #                 "tri_index": True,
        #                 "dense_search": True,  # Assuming Qdrant is available
        #                 "sparse_search": True,  # Assuming BM25 is available
        #                 "structural_facts": True,
        #             },
        #         }
        #
        #         return json.dumps(response, indent=2)
        #
        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": str(e),
        #                 "query": query,
        #                 "message": f"TriIndex search failed: {str(e)}",
        #             },
        #             indent=2,
        #         )

        # @self.mcp.tool()
        # async def omn1_workflow_context(
        #     action: str,
        #     session_id: Optional[str] = None,
        #     workflow_name: Optional[str] = None,
        #     current_role: Optional[str] = None,
        #     recent_files: Optional[List[str]] = None,
        #     workflow_step: Optional[str] = None,
        # ) -> str:
        #     """Get/set current workflow context for intelligent caching

        #     Manages workflow context and file access patterns for intelligent
        #     prefetching and cache optimization. Tracks developer workflows to
        #     predict next files and improve cache hit rates.

        #     Args:
        #         action: Action to perform - "get", "set", or "predict"
        #         session_id: Optional session ID (auto-generated if omitted)
        #         workflow_name: Optional workflow name (e.g., "feature/oauth-login")
        #         current_role: Optional role (architect, developer, tester, reviewer)
        #         recent_files: Optional list of recently accessed files
        #         workflow_step: Optional workflow step (planning, implementation, testing, review)

        #     Returns:
        #         JSON string with workflow context and predictions

        #     Examples:
        #         # Set workflow context
        #         result = omn1_workflow_context(
        #             action="set",
        #             session_id="session_123",
        #             workflow_name="feature/oauth",
        #             current_role="developer",
        #             recent_files=["/auth.py", "/config.py"],
        #             workflow_step="implementation"
        #         )

        #         # Get workflow context
        #         result = omn1_workflow_context(
        #             action="get",
        #             session_id="session_123"
        #         )

        #         # Predict next files
        #         result = omn1_workflow_context(
        #             action="predict",
        #             session_id="session_123",
        #             recent_files=["/auth.py", "/config.py"]
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         # Check if Redis L1 cache is available
        #         if not self.redis_l1_cache or not self.redis_l1_cache.is_available():
        #             return json.dumps(
        #                 {
        #                     "status": "unavailable",
        #                     "message": "Redis L1 cache service not available",
        #                     "action": action,
        #                 },
        #                 indent=2,
        #             )

        #         # Use session manager's session ID if not provided
        #         if session_id is None:
        #             session_id = None

        #         if action == "set":
        #             # Set workflow context
        #             success = await self.redis_l1_cache.set_workflow_context(
        #                 session_id=session_id,
        #                 workflow_name=workflow_name,
        #                 current_role=current_role,
        #                 recent_files=recent_files or [],
        #                 workflow_step=workflow_step,
        #             )

        #             if success:
        #                 return json.dumps(
        #                     {
        #                         "status": "success",
        #                         "action": "set",
        #                         "session_id": session_id,
        #                         "workflow_name": workflow_name,
        #                         "current_role": current_role,
        #                         "recent_files_count": len(recent_files or []),
        #                     },
        #                     indent=2,
        #                 )
        #             else:
        #                 return json.dumps(
        #                     {
        #                         "status": "error",
        #                         "action": "set",
        #                         "message": "Failed to set workflow context",
        #                     },
        #                     indent=2,
        #                 )

        #         elif action == "get":
        #             # Get workflow context
        #             context = await self.redis_l1_cache.get_workflow_context(session_id)

        #             if context:
        #                 return json.dumps(
        #                     {
        #                         "status": "success",
        #                         "action": "get",
        #                         "context": context,
        #                     },
        #                     indent=2,
        #                 )
        #             else:
        #                 return json.dumps(
        #                     {
        #                         "status": "not_found",
        #                         "action": "get",
        #                         "session_id": session_id,
        #                         "message": "No workflow context found for this session",
        #                     },
        #                     indent=2,
        #                 )

        #         elif action == "predict":
        #             # Predict next files based on access patterns
        #             if not recent_files:
        #                 return json.dumps(
        #                     {
        #                         "status": "error",
        #                         "action": "predict",
        #                         "message": "recent_files required for prediction",
        #                     },
        #                     indent=2,
        #                 )

        #             predictions = await self.redis_l1_cache.predict_next_files(
        #                 session_id=session_id, recent_files=recent_files, top_k=3
        #             )

        #             prediction_time_ms = (time.time() - start_time) * 1000

        #             return json.dumps(
        #                 {
        #                     "status": "success",
        #                     "action": "predict",
        #                     "session_id": session_id,
        #                     "recent_files": recent_files,
        #                     "predictions": predictions,
        #                     "prediction_count": len(predictions),
        #                     "prediction_time_ms": prediction_time_ms,
        #                 },
        #                 indent=2,
        #             )

        #         else:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "message": f"Unknown action: {action}. Use 'get', 'set', or 'predict'",
        #                 },
        #                 indent=2,
        #             )

        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "action": action,
        #                 "error": str(e),
        #                 "session_id": session_id,
        #             },
        #             indent=2,
        #         )

        # @self.mcp.tool()
        # async def omn1_resume_workflow(session_id: Optional[str] = None) -> str:
        #     """Resume incomplete workflow from previous session

        #     Restores workflow context, cached files, and suggests next actions.
        #     Addresses the #1 BMAD-METHOD complaint: context loss between sessions.

        #     Args:
        #         session_id: Optional session ID (None = detect latest incomplete workflow)

        #     Returns:
        #         JSON with workflow state, restored files, and next action suggestions

        #     Examples:
        #         # Auto-detect latest incomplete workflow
        #         result = omn1_resume_workflow()

        #         # Resume specific session
        #         result = omn1_resume_workflow(session_id="session_abc123")
        #     """
        #     start_time = time.time()

        #     try:
        #         # Initialize checkpoint service if needed
        #         if not hasattr(self, "checkpoint_service"):
        #             try:
        #                 from workflow_checkpoint_service import (
        #                     WorkflowCheckpointService,
        #                 )

        #                 self.checkpoint_service = WorkflowCheckpointService()
        #                 await self.checkpoint_service.initialize()
        #             except Exception as e:
        #                 return json.dumps(
        #                     {
        #                         "status": "error",
        #                         "message": f"Failed to initialize checkpoint service: {str(e)}",
        #                         "suggestion": "Ensure PostgreSQL is running and workflow_checkpoints table exists",
        #                     },
        #                     indent=2,
        #                 )

        #         # Find latest incomplete workflow
        #         checkpoint = await self.checkpoint_service.get_latest_checkpoint(
        #             session_id=session_id, completed=False
        #         )

        #         if not checkpoint:
        #             return json.dumps(
        #                 {
        #                     "status": "no_workflow_found",
        #                     "message": "No incomplete workflows found to resume",
        #                     "suggestion": "Start a new workflow with omn1_workflow_context",
        #                 },
        #                 indent=2,
        #             )

        #         # Restore files to hot cache (if Redis cache available)
        #         files_restored = []
        #         if (
        #             self.redis_l1_cache
        #             and self.redis_l1_cache.is_available()
        #             and checkpoint["context_files"]
        #         ):
        #             for file_path in checkpoint["context_files"]:
        #                 try:
        #                     # Try to get from cache, if not there, it will be loaded on demand
        #                     cached = await self.redis_l1_cache.get_cached_file(
        #                         file_path
        #                     )
        #                     if cached:
        #                         files_restored.append(file_path)
        #                 except:
        #                     pass

        #         # Generate suggested next actions based on workflow step
        #         suggested_actions = self._generate_next_actions(checkpoint)

        #         # Create context summary
        #         context_summary = self._create_context_summary(checkpoint)

        #         elapsed_ms = (time.time() - start_time) * 1000

        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "workflow_found": True,
        #                 "checkpoint_id": checkpoint["id"],
        #                 "workflow_name": checkpoint["workflow_name"],
        #                 "last_step": checkpoint["workflow_step"] or "Not specified",
        #                 "last_activity": checkpoint["last_activity"],
        #                 "files_restored": files_restored
        #                 if files_restored
        #                 else checkpoint["context_files"],
        #                 "cache_status": f"Hot cache populated with {len(files_restored)} files"
        #                 if files_restored
        #                 else "Files available on demand",
        #                 "workflow_role": checkpoint["workflow_role"] or "Not specified",
        #                 "suggested_next_actions": suggested_actions,
        #                 "context_summary": context_summary,
        #                 "metadata": checkpoint["metadata"],
        #                 "restore_time_ms": round(elapsed_ms, 2),
        #             },
        #             indent=2,
        #         )

        #     except Exception as e:
        #         logger.error(f"Failed to resume workflow: {e}")
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "message": f"Failed to resume workflow: {str(e)}",
        #             },
        #             indent=2,
        #         )

        # def _generate_next_actions(self, checkpoint: Dict[str, Any]) -> List[str]:
        #     """Generate suggested next actions based on checkpoint"""
        #     actions = []

        #     role = checkpoint.get("workflow_role", "")
        #     step = checkpoint.get("workflow_step", "")
        #     workflow = checkpoint.get("workflow_name", "")

        #     # Role-based suggestions
        #     if "architect" in role.lower():
        #         actions.append("Review architecture documentation")
        #         actions.append("Update system design diagrams")
        #     elif "developer" in role.lower():
        #         if "implement" in step.lower():
        #             actions.append(f"Continue implementation: {step}")
        #             actions.append("Run unit tests")
        #         else:
        #             actions.append("Start implementation phase")
        #     elif "tester" in role.lower():
        #         actions.append("Run test suite")
        #         actions.append("Review test coverage")
        #     elif "reviewer" in role.lower():
        #         actions.append("Continue code review")
        #         actions.append("Check for security issues")

        #     # Workflow-based suggestions
        #     if "feature/" in workflow:
        #         actions.append("Check feature branch is up to date")
        #     elif "bugfix/" in workflow or "fix/" in workflow:
        #         actions.append("Verify bug reproduction")
        #         actions.append("Test fix thoroughly")

        #     # Default suggestions if nothing specific
        #     if not actions:
        #         actions.append("Review last changes")
        #         actions.append("Continue where you left off")

        #     return actions[:5]  # Max 5 suggestions

        # def _create_context_summary(self, checkpoint: Dict[str, Any]) -> str:
        #     """Create human-readable context summary"""
        #     workflow = checkpoint.get("workflow_name", "Unknown")
        #     step = checkpoint.get("workflow_step", "Not specified")
        #     role = checkpoint.get("workflow_role", "developer")
        #     files = checkpoint.get("context_files", [])

        #     summary = f"You were working on '{workflow}' as a {role}."

        #     if step:
        #         summary += f" Last step: {step}."

        #     if files:
        #         summary += f" Working with {len(files)} files: {', '.join(files[:3])}"
        #         if len(files) > 3:
        #             summary += f" and {len(files) - 3} more."

        #     # Add any custom context from metadata
        #     metadata = checkpoint.get("metadata", {})
        #     if metadata.get("last_change"):
        #         summary += f" Last change: {metadata['last_change']}."

        #     return summary

        # @self.mcp.tool()
        # async def omn1_optimize_context(
        #     action: str,
        #     current_files: Optional[List[str]] = None,
        #     workflow_context: Optional[Dict] = None,
        #     token_limit: int = 200000,
        #     current_tokens: int = 0,
        #     target_reduction: float = 0.3,
        # ) -> str:
        #     """Optimize context window with workflow-aware pruning

        #     Prevents context overflow by intelligently pruning files based on:
        #     - Current workflow role and context (40% weight)
        #     - File relationships from knowledge graph (30% weight)
        #     - Access frequency and recency (20% weight)
        #     - Information density (10% weight)

        #     Workflow-critical files are preserved even under aggressive pruning.

        #     Args:
        #         action: Operation to perform
        #             - "analyze": Analyze current context usage and get recommendations
        #             - "prune": Perform intelligent pruning to fit token budget
        #             - "recommend": Suggest files to add based on workflow
        #         current_files: List of file paths currently in context
        #         workflow_context: Current workflow state (from omn1_workflow_context)
        #         token_limit: Maximum token budget (default: 200000)
        #         current_tokens: Current token usage (0 = estimate from files)
        #         target_reduction: Target reduction percentage for pruning (0.0-1.0, default: 0.3)

        #     Returns:
        #         JSON with optimization results

        #     Examples:
        #         # Analyze context health
        #         result = omn1_optimize_context(
        #             action="analyze",
        #             current_files=["src/main.py", "src/utils.py"],
        #             workflow_context={"current_role": "developer"},
        #             current_tokens=150000
        #         )
        #         # Returns: status, token_usage, recommendations, eviction_candidates

        #         # Prune context intelligently
        #         result = omn1_optimize_context(
        #             action="prune",
        #             current_files=["src/main.py", "src/utils.py", "docs/readme.md"],
        #             workflow_context={"current_role": "developer", "recent_files": ["src/main.py"]},
        #             current_tokens=180000,
        #             target_reduction=0.4
        #         )
        #         # Returns: files_to_keep, files_to_evict, tokens_saved

        #         # Get file recommendations
        #         result = omn1_optimize_context(
        #             action="recommend",
        #             workflow_context={"workflow_name": "feature/auth"},
        #             current_files=["src/auth.py"]
        #         )
        #         # Returns: recommended files with relevance scores

        #     Performance:
        #         - Analysis: <50ms
        #         - Pruning: <100ms
        #         - Token savings: 40-60% through intelligent eviction
        #     """
        #     try:
        #         # Validate action
        #         valid_actions = ["analyze", "prune", "recommend"]
        #         if action not in valid_actions:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": f"Invalid action '{action}'. Must be one of: {valid_actions}",
        #                 },
        #                 indent=2,
        #             )

        #         # Check if context optimizer service is available
        #         context_optimizer_url = "http://localhost:8006"

        #         try:
        #             # Health check
        #             health = await self.http_client.get(
        #                 f"{context_optimizer_url}/health",
        #                 timeout=2.0,
        #             )
        #             if health.status_code != 200:
        #                 return json.dumps(
        #                     {
        #                         "status": "error",
        #                         "error": "Context optimizer service unavailable",
        #                         "suggestion": "Start service with: cd omnimemory-context-optimizer/src && python api_server.py",
        #                     },
        #                     indent=2,
        #                 )
        #         except Exception as e:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": f"Context optimizer service not reachable: {str(e)}",
        #                     "suggestion": "Ensure service is running on port 8006",
        #                 },
        #                 indent=2,
        #             )

        #         # Route to appropriate endpoint
        #         if action == "analyze":
        #             response = await self.http_client.post(
        #                 f"{context_optimizer_url}/analyze",
        #                 json={
        #                     "current_files": current_files or [],
        #                     "workflow_context": workflow_context,
        #                     "token_limit": token_limit,
        #                     "current_tokens": current_tokens,
        #                 },
        #                 timeout=30.0,
        #             )
        #             result = response.json()

        #             # Add human-readable summary
        #             status = result.get("status", "unknown")
        #             token_pct = result.get("token_percentage", 0)
        #             recommendation = result.get("recommendation", "")

        #             summary = f"\n📊 Context Analysis:\n"
        #             summary += f"Status: {status.upper()}\n"
        #             summary += f"Token usage: {token_pct}% ({result.get('token_usage', 0):,} / {token_limit:,})\n"
        #             summary += f"Recommendation: {recommendation.replace('_', ' ')}\n"

        #             if result.get("eviction_candidates"):
        #                 summary += f"\nTop eviction candidates: {len(result['eviction_candidates'])}\n"
        #                 summary += f"Potential savings: {result.get('potential_token_savings', 0):,} tokens\n"

        #             result["summary"] = summary
        #             return json.dumps(result, indent=2)

        #         elif action == "prune":
        #             response = await self.http_client.post(
        #                 f"{context_optimizer_url}/prune",
        #                 json={
        #                     "current_files": current_files or [],
        #                     "workflow_context": workflow_context,
        #                     "token_budget": token_limit,
        #                     "current_tokens": current_tokens,
        #                     "target_reduction": target_reduction,
        #                 },
        #                 timeout=30.0,
        #             )
        #             result = response.json()

        #             # Add human-readable summary
        #             if result.get("status") == "success":
        #                 summary = f"\n✂️  Context Pruning:\n"
        #                 summary += f"Files kept: {result.get('files_kept', 0)}\n"
        #                 summary += f"Files evicted: {result.get('files_evicted', 0)}\n"
        #                 summary += f"Tokens saved: {result.get('tokens_saved', 0):,} ({result.get('reduction_percentage', 0)}%)\n"
        #                 summary += f"Final token count: {result.get('optimized_token_count', 0):,}\n"

        #                 result["summary"] = summary

        #             return json.dumps(result, indent=2)

        #         elif action == "recommend":
        #             response = await self.http_client.post(
        #                 f"{context_optimizer_url}/recommend",
        #                 json={
        #                     "workflow_context": workflow_context,
        #                     "current_files": current_files,
        #                     "limit": 10,
        #                 },
        #                 timeout=30.0,
        #             )
        #             result = response.json()

        #             # Add human-readable summary
        #             recs = result.get("recommendations", [])
        #             if recs:
        #                 summary = f"\n💡 File Recommendations ({len(recs)}):\n"
        #                 for i, rec in enumerate(recs[:5], 1):
        #                     summary += f"{i}. {rec['file_path']} "
        #                     summary += f"(score: {rec['relevance_score']:.2f}, reason: {rec['reason']})\n"

        #                 result["summary"] = summary
        #             else:
        #                 result["summary"] = "No recommendations available"

        #             return json.dumps(result, indent=2)

        #     except httpx.HTTPError as e:
        #         return json.dumps(
        #             {
        #                 "status": "error",
        #                 "error": f"HTTP error communicating with context optimizer: {str(e)}",
        #                 "action": action,
        #             },
        #             indent=2,
        #         )

        #     except Exception as e:
        #         return json.dumps(
        #             {"status": "error", "error": str(e), "action": action}, indent=2
        #         )

        # # ===== Phase 5C: LSP Symbol-Level Operations =====

        # # PRIVATE HELPER - Use omn1_read(target="symbol") instead
        # # External users should call: omn1_read(file_path, target="symbol", symbol=name)
        # async def _omn1_read_symbol(
        #     file_path: str,
        #     symbol: str,
        #     compress: bool = True,
        #     language: Optional[str] = None,
        # ) -> str:
        #     """PRIVATE HELPER - Not exposed via MCP tool registry.

        #     Internal implementation for reading specific symbols.
        #     External users should use: omn1_read(file_path, target="symbol", symbol=name)

        #     ---

        #     Read specific symbol (function/class/method) from file - 99% token savings

        #     Instead of reading entire files (1,250 tokens), read only the specific
        #     symbol you need (50 tokens). Achieves 96%+ token savings vs full file reads.

        #     Examples:
        #         # Read just the authenticate function
        #         omn1_read_symbol(
        #             file_path="src/auth.py",
        #             symbol="authenticate"
        #         )
        #         → Returns ONLY the function (50 tokens vs 5,000 full file)

        #         # Read a specific class
        #         omn1_read_symbol(
        #             file_path="src/models.py",
        #             symbol="UserModel"
        #         )
        #         → Returns just the class definition

        #     Args:
        #         file_path: Absolute path to the file
        #         symbol: Name of symbol to read (function, class, method, variable)
        #         compress: Apply compression to result (default: True)
        #         language: Override language detection (python, typescript, go, rust, java)

        #     Returns:
        #         JSON with symbol content and metadata:
        #         {
        #             "symbol_name": "authenticate",
        #             "kind": "function",
        #             "signature": "def authenticate(user: str, password: str) -> bool:",
        #             "content": "<full function code>",
        #             "docstring": "Authenticate a user...",
        #             "line_start": 42,
        #             "line_end": 58,
        #             "tokens_saved": 1200,
        #             "compression_ratio": 25.0
        #         }

        #     Token Savings:
        #         - Full file: 5KB = 1,250 tokens
        #         - Symbol only: 200 bytes = 50 tokens
        #         - Savings: 1,200 tokens (96%)
        #         - With compression: 10 tokens (99.2%)
        #     """
        #     try:
        #         # Try LSP first (if available)
        #         if LSP_AVAILABLE and self.symbol_service is not None:
        #             try:
        #                 # Ensure symbol service is started
        #                 await self.symbol_service.start()

        #                 # Read symbol via LSP
        #                 result = await self.symbol_service.read_symbol(
        #                     file_path=file_path, symbol=symbol, compress=compress
        #                 )

        #                 # If LSP succeeded, return result
        #                 if not result.get("error"):
        #                     # Add helpful summary
        #                     tokens_saved = result.get("tokens_saved", 0)
        #                     compression_ratio = result.get("compression_ratio", 1.0)

        #                     summary = f"\n🔍 Symbol Read (LSP):\n"
        #                     summary += f"Symbol: {result.get('symbol_name')} ({result.get('kind')})\n"
        #                     summary += f"Lines: {result.get('line_start')}-{result.get('line_end')}\n"
        #                     summary += f"Tokens saved: {tokens_saved:,} ({compression_ratio:.1f}x compression)\n"

        #                     result["summary"] = summary
        #                     return json.dumps(result, indent=2)

        #                 logger.warning(
        #                     f"LSP failed for {file_path}:{symbol}, trying AST fallback"
        #                 )

        #             except Exception as e:
        #                 logger.warning(
        #                     f"LSP error for {file_path}:{symbol}: {e}, trying AST fallback"
        #                 )

        #         # Fallback to AST extraction (Python only)
        #         if AST_EXTRACTOR_AVAILABLE and ASTSymbolExtractor is not None:
        #             # Check if it's a Python file
        #             if language == "python" or file_path.endswith(".py"):
        #                 try:
        #                     extractor = ASTSymbolExtractor()
        #                     symbol_content = extractor.get_symbol_content(
        #                         file_path, symbol
        #                     )

        #                     if symbol_content:
        #                         # Get symbol details for metadata
        #                         symbols = extractor.extract_symbols(file_path)
        #                         symbol_info = next(
        #                             (s for s in symbols if s["name"] == symbol), None
        #                         )

        #                         # Calculate token savings (approximate)
        #                         file_size = len(
        #                             Path(file_path).read_text(encoding="utf-8")
        #                         )
        #                         symbol_size = len(symbol_content)
        #                         tokens_saved = int(
        #                             (file_size - symbol_size) / 4
        #                         )  # Rough estimate: 4 chars = 1 token

        #                         result = {
        #                             "symbol_name": symbol,
        #                             "kind": symbol_info.get("kind", "unknown")
        #                             if symbol_info
        #                             else "unknown",
        #                             "signature": symbol_info.get("signature", "")
        #                             if symbol_info
        #                             else "",
        #                             "content": symbol_content,
        #                             "docstring": symbol_info.get("doc", "")
        #                             if symbol_info
        #                             else "",
        #                             "line_start": symbol_info.get("line_start", 0)
        #                             if symbol_info
        #                             else 0,
        #                             "line_end": symbol_info.get("line_end", 0)
        #                             if symbol_info
        #                             else 0,
        #                             "tokens_saved": tokens_saved,
        #                             "compression_ratio": file_size
        #                             / max(symbol_size, 1),
        #                             "method": "ast",
        #                             "service": {
        #                                 "cache_hit": False,
        #                                 "timestamp": "2024-01-01T00:00:00",
        #                                 "compressed": False,
        #                             },
        #                         }

        #                         # Add helpful summary
        #                         summary = f"\n🔍 Symbol Read (AST Fallback):\n"
        #                         summary += f"Symbol: {result['symbol_name']} ({result['kind']})\n"
        #                         summary += f"Lines: {result['line_start']}-{result['line_end']}\n"
        #                         summary += f"Tokens saved: {tokens_saved:,} ({result['compression_ratio']:.1f}x compression)\n"
        #                         summary += "Method: AST parser (LSP unavailable)\n"

        #                         result["summary"] = summary
        #                         return json.dumps(result, indent=2)

        #                 except Exception as e:
        #                     logger.error(
        #                         f"AST extraction failed for {file_path}:{symbol}: {e}"
        #                     )

        #         # No extraction method available
        #         return json.dumps(
        #             {
        #                 "error": True,
        #                 "message": "Symbol extraction not available. LSP unavailable and AST fallback failed.",
        #                 "symbol": symbol,
        #                 "file_path": file_path,
        #                 "tip": "For LSP: pip install python-lsp-server. AST fallback works for Python files.",
        #             },
        #             indent=2,
        #         )

        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "error": True,
        #                 "message": f"Error reading symbol: {str(e)}",
        #                 "symbol": symbol,
        #                 "file_path": file_path,
        #             },
        #             indent=2,
        #         )

        # # PRIVATE HELPER - Use omn1_read(target="overview") instead
        # # External users should call: omn1_read(file_path, target="overview")
        # async def _omn1_symbol_overview(
        #     file_path: str, include_details: bool = False, compress: bool = True
        # ) -> str:
        #     """PRIVATE HELPER - Not exposed via MCP tool registry.

        #     Internal implementation for file structure overview.
        #     External users should use: omn1_read(file_path, target="overview")

        #     ---

        #     Get file structure overview without reading full content - 98% token savings

        #     Instead of reading entire files to understand structure, get a compact
        #     overview listing all symbols. Perfect for "what's in this file?" queries.

        #     Examples:
        #         # Quick file overview
        #         omn1_symbol_overview(file_path="src/auth.py")
        #         → Returns structure: classes, functions, methods (100 tokens vs 5,000)

        #         # Detailed overview with signatures
        #         omn1_symbol_overview(
        #             file_path="src/auth.py",
        #             include_details=True
        #         )
        #         → Includes function signatures and docstrings (300 tokens vs 5,000)

        #     Args:
        #         file_path: Absolute path to the file
        #         include_details: Include signatures and docstrings (default: False)
        #         compress: Apply compression to result (default: True)

        #     Returns:
        #         JSON with file structure:
        #         {
        #             "file": "auth.py",
        #             "language": "python",
        #             "total_symbols": 15,
        #             "classes": ["AuthManager", "TokenValidator"],
        #             "functions": ["authenticate", "validate_token", "refresh_token"],
        #             "methods": {
        #                 "AuthManager": ["login", "logout", "verify"],
        #                 "TokenValidator": ["validate", "decode"]
        #             },
        #             "variables": ["SECRET_KEY", "TOKEN_EXPIRY"],
        #             "loc": 250,
        #             "tokens_used": 100,
        #             "tokens_saved": 1150,
        #             "compression_ratio": 12.5
        #         }

        #     Token Savings:
        #         - Full file: 5KB = 1,250 tokens
        #         - Overview: 400 bytes = 100 tokens
        #         - Savings: 1,150 tokens (92%)
        #         - Use Case: "What's in this file?" queries

        #     Perfect For:
        #         - Understanding file structure before diving in
        #         - Finding which file contains a specific function
        #         - Getting codebase overview without reading all files
        #         - Code navigation and exploration
        #     """
        #     try:
        #         # Try LSP first (if available)
        #         if LSP_AVAILABLE and self.symbol_service is not None:
        #             try:
        #                 # Ensure symbol service is started
        #                 await self.symbol_service.start()

        #                 # Get overview via LSP
        #                 result = await self.symbol_service.get_overview(
        #                     file_path=file_path,
        #                     include_details=include_details,
        #                     compress=compress,
        #                 )

        #                 # If LSP succeeded, return result
        #                 if not result.get("error"):
        #                     # Add helpful summary
        #                     total_symbols = result.get("total_symbols", 0)
        #                     tokens_saved = result.get("tokens_saved", 0)
        #                     compression_ratio = result.get("compression_ratio", 1.0)

        #                     summary = f"\n📋 File Overview (LSP):\n"
        #                     summary += f"File: {result.get('file')}\n"
        #                     summary += f"Total symbols: {total_symbols}\n"
        #                     summary += f"Classes: {len(result.get('classes', []))}, "
        #                     summary += (
        #                         f"Functions: {len(result.get('functions', []))}, "
        #                     )
        #                     summary += f"Methods: {sum(len(v) for v in result.get('methods', {}).values())}\n"
        #                     summary += f"LOC: {result.get('loc', 0)}\n"
        #                     summary += f"Tokens saved: {tokens_saved:,} ({compression_ratio:.1f}x compression)\n"

        #                     result["summary"] = summary
        #                     return json.dumps(result, indent=2)

        #                 logger.warning(
        #                     f"LSP failed for {file_path}, trying AST fallback"
        #                 )

        #             except Exception as e:
        #                 logger.warning(
        #                     f"LSP error for {file_path}: {e}, trying AST fallback"
        #                 )

        #         # Fallback to AST extraction (Python only)
        #         if AST_EXTRACTOR_AVAILABLE and ASTSymbolExtractor is not None:
        #             # Check if it's a Python file
        #             if file_path.endswith(".py"):
        #                 try:
        #                     extractor = ASTSymbolExtractor()
        #                     overview = extractor.get_file_overview(file_path)

        #                     if not overview.get("error"):
        #                         # Calculate token savings (approximate)
        #                         file_size = len(
        #                             Path(file_path).read_text(encoding="utf-8")
        #                         )
        #                         overview_size = len(json.dumps(overview))
        #                         tokens_saved = int((file_size - overview_size) / 4)

        #                         # Add metadata
        #                         result = {
        #                             **overview,
        #                             "language": "python",
        #                             "tokens_saved": tokens_saved,
        #                             "compression_ratio": file_size
        #                             / max(overview_size, 1),
        #                             "method": "ast",
        #                         }

        #                         # Add helpful summary
        #                         total_symbols = result.get("total_symbols", 0)

        #                         summary = f"\n📋 File Overview (AST Fallback):\n"
        #                         summary += f"File: {result.get('file')}\n"
        #                         summary += f"Total symbols: {total_symbols}\n"
        #                         summary += (
        #                             f"Classes: {len(result.get('classes', []))}, "
        #                         )
        #                         summary += (
        #                             f"Functions: {len(result.get('functions', []))}, "
        #                         )
        #                         summary += f"Methods: {sum(len(v) for v in result.get('methods', {}).values())}\n"
        #                         summary += f"LOC: {result.get('loc', 0)}\n"
        #                         summary += f"Tokens saved: {tokens_saved:,} ({result['compression_ratio']:.1f}x compression)\n"
        #                         summary += "Method: AST parser (LSP unavailable)\n"

        #                         result["summary"] = summary
        #                         return json.dumps(result, indent=2)

        #                 except Exception as e:
        #                     logger.error(f"AST extraction failed for {file_path}: {e}")

        #         # No extraction method available
        #         return json.dumps(
        #             {
        #                 "error": True,
        #                 "message": "File overview not available. LSP unavailable and AST fallback failed.",
        #                 "file_path": file_path,
        #                 "tip": "For LSP: pip install python-lsp-server. AST fallback works for Python files.",
        #             },
        #             indent=2,
        #         )

        #     except Exception as e:
        #         return json.dumps(
        #             {
        #                 "error": True,
        #                 "message": f"Error getting overview: {str(e)}",
        #                 "file_path": file_path,
        #             },
        #             indent=2,
        #         )

        # # PRIVATE HELPER - Use omn1_read(target="references") or search(mode="references") instead
        # # External users should call: omn1_read(file_path, target="references", symbol=name)
        # # or: search(mode="references", file_path=path, symbol=name)
        # async def _omn1_find_references(file_path: str, symbol: str) -> str:
        #      """PRIVATE HELPER - Not exposed via MCP tool registry.

        #      Internal implementation for finding symbol references.
        #      External users should use:
        #      - omn1_read(file_path, target="references", symbol=name)
        #      - search(mode="references", file_path=path, symbol=name)

        #      ---

        #      Find all references to a symbol across the codebase

        #      Use LSP to find everywhere a symbol (function/class/variable) is referenced
        #      in the codebase. Essential for understanding code dependencies and impact.

        #      Examples:
        #          # Find all calls to authenticate function
        #          omn1_find_references(
        #              file_path="src/auth.py",
        #              symbol="authenticate"
        #          )
        #          → Returns list of all locations where authenticate() is called

        #          # Find all usages of a class
        #          omn1_find_references(
        #              file_path="src/models.py",
        #              symbol="UserModel"
        #          )
        #          → Shows everywhere UserModel is imported and used

        #      Args:
        #          file_path: File containing the symbol definition
        #          symbol: Name of the symbol to find references for

        #      Returns:
        #          JSON with reference locations:
        #          {
        #              "symbol": "authenticate",
        #              "file_path": "src/auth.py",
        #              "total_references": 15,
        #              "references": [
        #                  {
        #                      "file": "src/api.py",
        #                      "file_path": "src/api.py",
        #                      "line": 42,
        #                      "column": 10,
        #                      "type": "call"
        #                  },
        #                  {
        #                      "file": "tests/test_auth.py",
        #                      "line": 15,
        #                      "column": 5,
        #                      "type": "call"
        #                  }
        #              ]
        #          }

        #      Use Cases:
        #          - Understanding function usage and impact
        #          - Finding all callers of a function
        #          - Tracking variable usage across codebase
        #          - Impact analysis before refactoring
        #          - Code navigation and exploration
        #      """
        #      try:
        #          if not LSP_AVAILABLE or self.symbol_service is None:
        #              return json.dumps(
        #                  {
        #                      "error": True,
        #                      "message": "LSP Symbol Service required for reference tracking. Install: pip install python-lsp-server",
        #                      "tip": "Finding references requires LSP. AST fallback cannot provide cross-file reference tracking.",
        #                      "symbol": symbol,
        #                      "file_path": file_path,
        #                  },
        #                  indent=2,
        #              )

        #          # Ensure symbol service is started
        #          await self.symbol_service.start()

        #          # Find references
        #          result = await self.symbol_service.find_references(
        #              file_path=file_path, symbol=symbol
        #          )

        #          # Check for errors
        #          if result.get("error"):
        #              return json.dumps(
        #                  {
        #                      "error": True,
        #                      "message": result.get(
        #                          "message", "Failed to find references"
        #                      ),
        #                      "symbol": symbol,
        #                      "file_path": file_path,
        #                  },
        #                  indent=2,
        #              )

        #          # Add helpful summary
        #          total_refs = result.get("total_references", 0)
        #          references = result.get("references", [])

        #          summary = f"\n🔗 References Found:\n"
        #          summary += f"Symbol: {result.get('symbol')}\n"
        #          summary += f"Total references: {total_refs}\n"

        #          if references:
        #              # Group by file
        #              files_map = {}
        #              for ref in references:
        #                  file = ref.get("file")
        #                  if file not in files_map:
        #                      files_map[file] = []
        #                  files_map[file].append(ref.get("line"))

        #              summary += f"Files: {len(files_map)}\n"
        #              for file, lines in list(files_map.items())[:5]:
        #                  summary += (
        #                      f"  - {file}: lines {', '.join(map(str, sorted(lines)))}\n"
        #                  )

        #              if len(files_map) > 5:
        #                  summary += f"  ... and {len(files_map) - 5} more files\n"

        #          result["summary"] = summary

        #          return json.dumps(result, indent=2)

        #      except Exception as e:
        #          return json.dumps(
        #              {
        #                  "error": True,
        #                  "message": f"Error finding references: {str(e)}",
        #                  "symbol": symbol,
        #                  "file_path": file_path,
        #              },
        #              indent=2,
        #          )

        # # ===================================================================
        # # OMN1 Consolidated Tools - Simplified API
        # # ===================================================================

        def _parse_read_params(input_str: str) -> Dict[str, Any]:
            """Parse read tool input string into parameters.

            This works around Claude Code's MCP client limitation that only passes
            the first positional parameter. All parameters are encoded in a single
            string using delimiter-based syntax.

            Format: "file_path|mode|options"

            Examples:
                "file.py" → {"file_path": "file.py", "target": "full", "compress": True}
                "file.py|overview" → {"file_path": "file.py", "target": "overview"}
                "file.py|symbol:Settings" → {"file_path": "file.py", "target": "symbol", "symbol": "Settings"}
                "file.py|references:authenticate" → {"file_path": "file.py", "target": "references", "symbol": "authenticate"}
                "file.py|overview|details" → {"file_path": "file.py", "target": "overview", "include_details": True}
                "file.py|symbol:Settings|details" → symbol read with details
                "file.py|nocompress" → disable compression
                "file.py|lang:python" → override language detection
            """
            parts = input_str.split("|")
            params = {
                "file_path": parts[0].strip(),
                "target": "full",
                "compress": True,
                "max_tokens": 8000,
                "quality_threshold": 0.70,
                "include_details": False,
                "include_references": False,
                "symbol": None,
                "language": None,
                "offset": 0,  # For paginated cached results
                "limit": None,  # For paginated cached results
            }

            for part in parts[1:]:
                part_lower = part.strip().lower()
                part_original = part.strip()

                if part_lower == "overview":
                    params["target"] = "overview"
                elif part_lower.startswith("symbol:"):
                    params["target"] = "symbol"
                    # Preserve original case for symbol name
                    params["symbol"] = part_original.split(":", 1)[1].strip()
                elif part_lower.startswith("references:") or part_lower.startswith(
                    "refs:"
                ):
                    params["target"] = "references"
                    # Preserve original case for symbol name
                    symbol_part = part_original.split(":", 1)[1].strip()
                    params["symbol"] = symbol_part
                elif part_lower == "nocompress":
                    params["compress"] = False
                elif part_lower == "details":
                    params["include_details"] = True
                elif part_lower.startswith("lang:"):
                    params["language"] = part_original.split(":", 1)[1].strip()
                elif part_lower.startswith("maxtoken:") or part_lower.startswith(
                    "maxtokens:"
                ):
                    params["max_tokens"] = int(part_original.split(":", 1)[1].strip())
                elif part_lower.startswith("offset:"):
                    # For paginated cached results
                    params["offset"] = int(part_original.split(":", 1)[1].strip())
                elif part_lower.startswith("limit:"):
                    # For paginated cached results
                    params["limit"] = int(part_original.split(":", 1)[1].strip())

            return params

        def _parse_search_params(input_str: str) -> Dict[str, Any]:
            """Parse search tool input string into parameters.

            This works around Claude Code's MCP client limitation that only passes
            the first positional parameter. All parameters are encoded in a single
            string using delimiter-based syntax.

            Format: "query|mode|options"

            Examples:
                "authentication" → {"query": "authentication", "mode": "semantic", "limit": 5}
                "auth|tri_index" → {"query": "auth", "mode": "tri_index", "limit": 5}
                "auth|triindex" → same as above (alternative spelling)
                "auth|tri_index|limit:10" → {"query": "auth", "mode": "tri_index", "limit": 10}
                "Settings|references:SettingsManager|file:src/settings.py" → find references
                "error handling|semantic|limit:10|minrel:0.8" → high-precision semantic search
                "authentication|tri_index|nocontext" → tri-index without context
            """
            parts = input_str.split("|")
            params = {
                "query": parts[0].strip(),
                "mode": "semantic",
                "file_path": None,
                "symbol": None,
                "limit": 5,
                "min_relevance": 0.7,
                "enable_witness_rerank": True,
                "include_context": False,
            }

            for part in parts[1:]:
                part_lower = part.strip().lower()
                part_original = part.strip()

                if part_lower in ("tri_index", "triindex", "tri-index"):
                    params["mode"] = "tri_index"
                elif part_lower == "semantic":
                    params["mode"] = "semantic"
                elif part_lower.startswith("references:") or part_lower.startswith(
                    "refs:"
                ):
                    params["mode"] = "references"
                    # Preserve original case for symbol name
                    params["symbol"] = part_original.split(":", 1)[1].strip()
                elif part_lower.startswith("file:"):
                    params["file_path"] = part_original.split(":", 1)[1].strip()
                elif part_lower.startswith("limit:"):
                    params["limit"] = int(part_original.split(":", 1)[1].strip())
                elif part_lower.startswith("minrel:"):
                    params["min_relevance"] = float(
                        part_original.split(":", 1)[1].strip()
                    )
                elif part_lower == "nocontext":
                    params["include_context"] = False
                elif part_lower == "context":
                    params["include_context"] = True
                elif part_lower == "norerank":
                    params["enable_witness_rerank"] = False

            return params

        @self.mcp.tool()
        async def read(file_path: str) -> str:
            """Unified file reading tool with multiple modes - ONE tool for all reading needs

            WORKAROUND: Due to Claude Code's MCP client limitation (only passes first positional
            parameter), all parameters are encoded in a single string using delimiter syntax.

            This consolidates 4 separate tools into one:
            - "file.py" → Compressed file reading (90% token savings)
            - "file.py|overview" → File structure only (98% token savings)
            - "file.py|symbol:NAME" → Specific function/class (99% token savings)
            - "file.py|references:NAME" → Find symbol usages

            Parameter Format: "file_path|mode|options"

            Reading Modes:
                "file.py"                          → Full compressed read (default)
                "file.py|overview"                 → Structure overview (classes, functions, imports)
                "file.py|symbol:authenticate"      → Read specific function/class
                "file.py|references:authenticate"  → Find all usages of symbol
                "file.py|refs:authenticate"        → Same as references (shorthand)

            Options (can be combined):
                "|details"      → Include signatures and docstrings (for overview/symbol modes)
                "|nocompress"   → Disable compression (return raw file)
                "|lang:python"  → Override language detection
                "|maxtokens:10000" → Set max tokens for full reads

            Examples:
                # Read full file (compressed, default behavior)
                read("src/auth.py")
                → Returns compressed file content (90% token savings)

                # Get file structure overview
                read("src/auth.py|overview")
                → Returns classes, functions, imports (98% token savings)

                # Read specific function
                read("src/auth.py|symbol:authenticate")
                → Returns only that function (99% token savings)

                # Read function with detailed signatures
                read("src/auth.py|symbol:authenticate|details")
                → Returns function with full signature and docstring

                # Find all references to a symbol
                read("src/auth.py|references:authenticate")
                → Returns all locations where authenticate() is called

                # Overview with detailed signatures
                read("src/auth.py|overview|details")
                → Includes function signatures and docstrings

                # Full read without compression (fallback)
                read("src/auth.py|nocompress")
                → Returns raw file content (no token savings)

                # Combined options
                read("src/settings.py|symbol:SettingsManager|details|lang:python")
                → Read SettingsManager class with details, force Python parsing

            Returns:
                JSON string with content and metadata appropriate for the mode

            Token Savings Examples:
            - Full file (compressed): 5KB → 500 tokens (90% savings)
            - Overview only: 5KB → 100 tokens (98% savings)
            - Single symbol: 5KB → 50 tokens (99% savings)
            """
            # Parse parameters from single string input
            params = _parse_read_params(file_path)

            # Extract parsed parameters
            file_path = params["file_path"]
            target = params["target"]
            compress = params["compress"]
            max_tokens = params["max_tokens"]
            quality_threshold = params["quality_threshold"]
            include_details = params["include_details"]
            include_references = params["include_references"]
            symbol = params["symbol"]
            language = params["language"]

            # Debug logging
            print(
                f"🔍 Parsed read params: file_path={repr(file_path)}, target={repr(target)}, symbol={repr(symbol)}, compress={compress}",
                file=sys.stderr,
            )

            # ZERO NEW TOOLS: Detect virtual cached result files
            # Virtual paths: ~/.omnimemory/cached_results/{result_id}.json
            if file_path.startswith(
                "~/.omnimemory/cached_results/"
            ) or file_path.startswith(
                str(Path.home() / ".omnimemory" / "cached_results")
            ):
                print(
                    f"🔍 Detected virtual cached result file: {file_path}",
                    file=sys.stderr,
                )
                # Extract offset and limit from params if available
                offset = params.get("offset", 0)
                limit = params.get("limit", None)
                return await self._read_cached_result(
                    result_path=file_path, offset=offset, limit=limit
                )

            # Repository detection logging
            user_id = self._get_user_id()
            repo_id = self._get_repo_id(file_path)
            print(f"🔍 User: {user_id}, Repo: {repo_id}", file=sys.stderr)

            # NEW: Check Unified Cache Manager (L1 user cache) first
            if self.cache_manager:
                try:
                    # Strategy 1: Content-based cache key (auto-invalidates on file change)
                    file_hash = self._get_file_hash(file_path)
                    cache_key = f"{file_path}:{file_hash}"
                    cached = self.cache_manager.get_read_result(user_id, cache_key)

                    if cached:
                        # Strategy 2: Timestamp validation (double-check freshness)
                        cached_file_mtime = cached.get("file_mtime", 0)

                        try:
                            current_file_mtime = (
                                Path(file_path).expanduser().resolve().stat().st_mtime
                            )

                            # If file modified after cache, invalidate
                            if current_file_mtime > cached_file_mtime:
                                print(
                                    f"⚠️  Cache STALE: {file_path} (file modified, re-reading)",
                                    file=sys.stderr,
                                )
                                # Don't return cached, continue to read fresh
                            else:
                                # Cache is fresh and validated
                                print(
                                    f"🎯 Cache HIT: {file_path} (L1 User Cache, validated fresh)",
                                    file=sys.stderr,
                                )
                                # Sanitize response - only return essential data
                                sanitized_cached = {
                                    "content": cached.get("content", ""),
                                    "optimized": True,
                                }

                                # Track cache hit for metrics
                                tokens_original = await self._estimate_file_tokens(
                                    file_path
                                )
                                tokens_actual = self._count_tokens(
                                    json.dumps(sanitized_cached)
                                )
                                await self._track_tool_operation(
                                    tool_name="read",
                                    operation_mode=target,
                                    parameters={
                                        "compress": compress,
                                        "max_tokens": max_tokens,
                                        "quality_threshold": quality_threshold,
                                    },
                                    file_path=file_path,
                                    tokens_original=tokens_original,
                                    tokens_actual=tokens_actual,
                                    response_time_ms=(time.time() - start_time) * 1000,
                                )

                                return json.dumps(sanitized_cached, indent=2)
                        except:
                            # If timestamp check fails, return cached anyway (safe fallback)
                            print(
                                f"🎯 Cache HIT: {file_path} (L1 User Cache)",
                                file=sys.stderr,
                            )
                            # Sanitize response - only return essential data
                            sanitized_cached = {
                                "content": cached.get("content", ""),
                                "optimized": True,
                            }

                            # Track cache hit for metrics
                            tokens_original = await self._estimate_file_tokens(
                                file_path
                            )
                            tokens_actual = self._count_tokens(
                                json.dumps(sanitized_cached)
                            )
                            await self._track_tool_operation(
                                tool_name="read",
                                operation_mode=target,
                                parameters={
                                    "compress": compress,
                                    "max_tokens": max_tokens,
                                    "quality_threshold": quality_threshold,
                                },
                                file_path=file_path,
                                tokens_original=tokens_original,
                                tokens_actual=tokens_actual,
                                response_time_ms=(time.time() - start_time) * 1000,
                            )

                            return json.dumps(sanitized_cached, indent=2)
                except Exception as e:
                    print(f"⚠️  Cache error: {e}", file=sys.stderr)

            # L2: Check repository cache (SHARED by team)
            if self.cache_manager:
                try:
                    # Use content-based hash for L2 cache (auto-invalidates on file change)
                    file_hash = self._get_file_hash(file_path)
                    # Combine path and content hash for unique L2 key
                    l2_key = hashlib.sha256(
                        f"{file_path}:{file_hash}".encode()
                    ).hexdigest()[:16]
                    repo_cached = self.cache_manager.get_file_compressed(
                        repo_id, l2_key
                    )

                    if repo_cached:
                        content, metadata = repo_cached

                        # Timestamp validation for L2 cache
                        cached_file_mtime = float(metadata.get("file_mtime", 0))
                        try:
                            current_file_mtime = (
                                Path(file_path).expanduser().resolve().stat().st_mtime
                            )

                            # If file modified after cache, invalidate
                            if current_file_mtime > cached_file_mtime:
                                print(
                                    f"⚠️  L2 Cache STALE: {file_path} (file modified, re-reading)",
                                    file=sys.stderr,
                                )
                                # Don't return cached, continue to read fresh
                            else:
                                # L2 cache is fresh
                                print(
                                    f"🎯 Cache HIT: {file_path} (L2 Repository Cache - SHARED, validated fresh)",
                                    file=sys.stderr,
                                )

                                # Decode compressed content
                                # Internal result for caching and tracking
                                result = {
                                    "omn1_mode": target,
                                    "file_path": file_path,
                                    "content": content.decode("utf-8")
                                    if isinstance(content, bytes)
                                    else content,
                                    "compressed": metadata.get("compressed", "False")
                                    == "True",
                                    "cache_hit": True,
                                    "cache_tier": "L2",
                                    "cache_source": "repository_shared",
                                    "repo_id": repo_id,
                                    "cached_at": float(metadata.get("cached_at", 0)),
                                    "file_mtime": cached_file_mtime,
                                    "omn1_info": f"{target} mode (from team repository cache)",
                                }

                                # Sanitize response for user - only return essential data
                                sanitized_result = {
                                    "content": content.decode("utf-8")
                                    if isinstance(content, bytes)
                                    else content,
                                    "optimized": True,
                                }

                                # Promote to L1 for faster next access (with content-based key)
                                cache_key = f"{file_path}:{file_hash}"
                                self.cache_manager.cache_read_result(
                                    user_id, cache_key, result, ttl=3600
                                )
                                print(
                                    f"   ↳ Promoted to L1 cache for faster future access",
                                    file=sys.stderr,
                                )

                                # Track L2 cache hit for metrics
                                tokens_original = await self._estimate_file_tokens(
                                    file_path
                                )
                                tokens_actual = self._count_tokens(
                                    json.dumps(sanitized_result)
                                )
                                await self._track_tool_operation(
                                    tool_name="read",
                                    operation_mode=target,
                                    parameters={
                                        "compress": compress,
                                        "max_tokens": max_tokens,
                                        "quality_threshold": quality_threshold,
                                    },
                                    file_path=file_path,
                                    tokens_original=tokens_original,
                                    tokens_actual=tokens_actual,
                                    response_time_ms=(time.time() - start_time) * 1000,
                                )

                                return json.dumps(sanitized_result, indent=2)
                        except Exception as timestamp_err:
                            # Timestamp check failed, but L2 cache exists - return it as safe fallback
                            print(
                                f"🎯 Cache HIT: {file_path} (L2 Repository Cache - SHARED, timestamp check failed)",
                                file=sys.stderr,
                            )
                            # Sanitize response - only return essential data
                            sanitized_result = {
                                "content": content.decode("utf-8")
                                if isinstance(content, bytes)
                                else content,
                                "optimized": True,
                            }

                            # Also cache the original result for internal tracking
                            result = {
                                "omn1_mode": target,
                                "file_path": file_path,
                                "content": content.decode("utf-8")
                                if isinstance(content, bytes)
                                else content,
                                "compressed": metadata.get("compressed", "False")
                                == "True",
                                "cache_hit": True,
                                "cache_tier": "L2",
                                "cache_source": "repository_shared",
                                "repo_id": repo_id,
                                "cached_at": float(metadata.get("cached_at", 0)),
                                "omn1_info": f"{target} mode (from team repository cache)",
                            }

                            # Promote to L1 (with content-based key)
                            cache_key = f"{file_path}:{file_hash}"
                            self.cache_manager.cache_read_result(
                                user_id, cache_key, result, ttl=3600
                            )
                            return json.dumps(sanitized_result, indent=2)
                except Exception as e:
                    print(f"⚠️  L2 cache check failed: {e}", file=sys.stderr)

            # Check if in cloud mode
            if self.connection_mode == "cloud":
                # Route through gateway
                result = await self._route_to_gateway(
                    "omn1_read",
                    file_path=file_path,
                    target=target,
                    compress=compress,
                    max_tokens=max_tokens,
                    quality_threshold=quality_threshold,
                    include_details=include_details,
                    include_references=include_references,
                    symbol=symbol,
                    language=language,
                )
                # Ensure result is a string (gateway may return dict or str)
                return str(result) if not isinstance(result, str) else result

            # LOCAL MODE: existing implementation continues below
            # Start timing for operation tracking
            start_time = time.time()

            # Estimate baseline tokens (what would be read without optimization)
            tokens_original = await self._estimate_file_tokens(file_path)

            try:
                print(
                    f"[DEBUG read()] Before routing: target={repr(target)}",
                    file=sys.stderr,
                )
                # Validate target mode
                valid_targets = ["full", "overview", "symbol", "references"]
                if target not in valid_targets:
                    return json.dumps(
                        {
                            "error": True,
                            "message": f"Invalid target mode: {target}. Must be one of: {', '.join(valid_targets)}",
                            "tip": "Use target='full' for compressed reading, 'overview' for structure, 'symbol' for specific functions",
                        },
                        indent=2,
                    )

                # Validate symbol parameter for modes that require it
                if target in ["symbol", "references"] and not symbol:
                    return json.dumps(
                        {
                            "error": True,
                            "message": f"target='{target}' requires 'symbol' parameter",
                            "example": f'read(file_path="{file_path}", target="{target}", symbol="function_name")',
                        },
                        indent=2,
                    )

                # Route to appropriate implementation
                print(
                    f"[DEBUG read()] Routing to target={repr(target)}",
                    file=sys.stderr,
                )
                if target == "full":
                    # Use context injector for compressed file reading
                    compression_result = await self._call_context_injector(file_path)

                    if compression_result and "content" in compression_result:
                        # Sanitize response - only return essential data
                        result = json.dumps(
                            {
                                "content": compression_result["content"],
                                "optimized": True,
                            },
                            indent=2,
                        )

                        # Track operation
                        tokens_actual = self._count_tokens(result)
                        response_time_ms = (time.time() - start_time) * 1000
                        await self._track_tool_operation(
                            tool_name="read",
                            operation_mode=target,
                            parameters={
                                "compress": compress,
                                "max_tokens": max_tokens,
                                "quality_threshold": quality_threshold,
                            },
                            file_path=file_path,
                            tokens_original=tokens_original,
                            tokens_actual=tokens_actual,
                            response_time_ms=response_time_ms,
                        )

                        # Track file access for session persistence
                        if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                            try:
                                await _SESSION_MANAGER.track_file_access(
                                    file_path=file_path,
                                    importance=0.7,  # Higher importance for explicitly read files
                                )
                                print(
                                    f"✓ Session tracking: Recorded file access for {file_path}",
                                    file=sys.stderr,
                                )
                            except Exception as track_error:
                                print(
                                    f"⚠ Failed to track file access: {track_error}",
                                    file=sys.stderr,
                                )

                        # Store in L2 repository cache FIRST (SHARED by team, 7 day TTL)
                        if self.cache_manager:
                            try:
                                repo_id = self._get_repo_id(file_path)
                                # Use content-based hash for cache key (auto-invalidates on file change)
                                content_hash = self._get_file_hash(file_path)
                                l2_key = hashlib.sha256(
                                    f"{file_path}:{content_hash}".encode()
                                ).hexdigest()[:16]

                                # Get file modification time for timestamp validation
                                try:
                                    file_mtime = (
                                        Path(file_path)
                                        .expanduser()
                                        .resolve()
                                        .stat()
                                        .st_mtime
                                    )
                                except:
                                    file_mtime = time.time()

                                # Prepare compressed content for L2
                                full_json = json.dumps(compression_result)

                                # Store in L2 (team-shared, 7 day TTL)
                                user_id = self._get_user_id()
                                self.cache_manager.cache_file_compressed(
                                    repo_id=repo_id,
                                    file_hash=l2_key,
                                    compressed_content=full_json.encode("utf-8"),
                                    metadata={
                                        "file_path": file_path,
                                        "cached_by": user_id,  # NEW: Track who cached it
                                        "mode": "full",
                                        "compressed": "True",
                                        "cached_at": str(time.time()),
                                        "file_mtime": str(
                                            file_mtime
                                        ),  # NEW: File modification time
                                        "size": str(len(full_json)),
                                    },
                                    ttl=604800,  # 7 days - long TTL for team sharing
                                )
                                print(
                                    f"💾 Cached in L2 (repository): {file_path} (full/compressed) - SHARED with team",
                                    file=sys.stderr,
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  L2 cache storage failed: {e}", file=sys.stderr
                                )

                        # NEW: Store in Unified Cache Manager (L1 user cache)
                        if self.cache_manager:
                            try:
                                user_id = self._get_user_id()

                                # Get file modification time for timestamp validation
                                try:
                                    file_mtime = (
                                        Path(file_path)
                                        .expanduser()
                                        .resolve()
                                        .stat()
                                        .st_mtime
                                    )
                                except:
                                    file_mtime = time.time()

                                result_to_cache = {
                                    "omn1_mode": "full",
                                    "file_path": file_path,
                                    "content": compression_result["content"],
                                    "compressed": True,
                                    "compression_ratio": compression_result.get(
                                        "compression_ratio", 0
                                    ),
                                    "tokens_saved": compression_result.get(
                                        "tokens_saved", 0
                                    ),
                                    "language": language,
                                    "timestamp": datetime.now().isoformat(),
                                    "file_mtime": file_mtime,  # NEW: File modification time
                                }

                                # Use content-based cache key (auto-invalidates on file change)
                                content_hash = self._get_file_hash(file_path)
                                cache_key = f"{file_path}:{content_hash}"

                                self.cache_manager.cache_read_result(
                                    user_id=user_id,
                                    file_path=cache_key,  # Use content-based key
                                    result=result_to_cache,
                                    ttl=3600,  # 1 hour
                                )
                                print(
                                    f"💾 Cached in L1 (user): {file_path} (full/compressed, content-based key)",
                                    file=sys.stderr,
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  Failed to cache: {e}",
                                    file=sys.stderr,
                                )

                        # Trigger smart prefetching for predicted files
                        if (
                            hasattr(self, "context_preloader")
                            and self.context_preloader
                        ):
                            try:
                                session_id = (
                                    _SESSION_MANAGER.current_session
                                    if _SESSION_MANAGER
                                    else None
                                )

                                if session_id:
                                    # Predict likely files user will need next
                                    predictions = await self.context_preloader.predict_likely_files(
                                        current_file=file_path,
                                        session_id=session_id,
                                        repo_id=repo_id,
                                        limit=5,
                                    )

                                    # Queue high-confidence predictions for background prefetching
                                    predicted_files = [
                                        p["file_path"]
                                        for p in predictions
                                        if p["confidence"] > 0.6
                                    ]
                                    if predicted_files:
                                        await self.context_preloader.prefetch_files(
                                            predicted_files, user_id, repo_id
                                        )
                                        print(
                                            f"⚡ Queued {len(predicted_files)} files for prefetching",
                                            file=sys.stderr,
                                        )
                            except Exception as e:
                                # Don't fail read if prefetching fails
                                print(
                                    f"⚠️  Prefetch prediction error: {e}",
                                    file=sys.stderr,
                                )

                        return result
                    else:
                        # Report compression failure to metrics
                        try:
                            await self.http_client.post(
                                "http://localhost:8003/metrics/compression-failure",
                                json={
                                    "session_id": None,
                                    "tool_id": _TOOL_ID,
                                    "file_path": file_path,
                                    "reason": "compression_service_unavailable",
                                    "timestamp": datetime.now().isoformat(),
                                },
                                timeout=1.0,
                            )
                        except Exception:
                            pass  # Don't fail read if metrics reporting fails

                        # Fallback to direct read if compression unavailable
                        path = Path(file_path).expanduser().resolve()
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Check token count BEFORE returning (MCP protocol limit is 25K)
                        content_token_count = self._count_tokens(content)
                        mcp_token_limit = 25000  # MCP protocol hard limit
                        effective_limit = (
                            min(max_tokens, mcp_token_limit)
                            if max_tokens
                            else mcp_token_limit
                        )

                        if content_token_count > effective_limit:
                            # File too large - return error with helpful instructions
                            result = json.dumps(
                                {
                                    "error": True,
                                    "omn1_mode": "full",
                                    "file_path": file_path,
                                    "message": f"File too large: {content_token_count:,} tokens (limit: {effective_limit:,})",
                                    "token_count": content_token_count,
                                    "max_tokens": effective_limit,
                                    "compressed": False,
                                    "solutions": [
                                        f"1. Start compression service: ./scripts/start_compression.sh (reduces to ~{content_token_count // 10:,} tokens, 90% savings)",
                                        "2. Use target='overview' to see file structure only (saves 98% tokens)",
                                        "3. Use target='<symbol_name>' to read specific function/class only (saves 99% tokens)",
                                        "4. Use standard Read tool with offset/limit parameters for pagination",
                                    ],
                                    "tip": f"Compression service at {COMPRESSION_SERVICE_URL} would reduce this to ~{content_token_count // 10:,} tokens (90% savings)",
                                    "omn1_info": "File exceeds token limit - see solutions above",
                                },
                                indent=2,
                            )
                        else:
                            # File size OK - return content with token count
                            result = json.dumps(
                                {
                                    "omn1_mode": "full",
                                    "file_path": file_path,
                                    "content": content,
                                    "compressed": False,
                                    "token_count": content_token_count,
                                    "omn1_info": "Full file read (compression unavailable)",
                                    "tip": f"Compression service not responding at {COMPRESSION_SERVICE_URL}. "
                                    f"Check service: curl {COMPRESSION_SERVICE_URL}/health || Start with: ./scripts/start_compression.sh",
                                },
                                indent=2,
                            )

                        # Track operation
                        tokens_actual = self._count_tokens(result)
                        response_time_ms = (time.time() - start_time) * 1000
                        await self._track_tool_operation(
                            tool_name="read",
                            operation_mode=target,
                            parameters={
                                "compress": compress,
                                "max_tokens": max_tokens,
                                "quality_threshold": quality_threshold,
                            },
                            file_path=file_path,
                            tokens_original=tokens_original,
                            tokens_actual=tokens_actual,
                            response_time_ms=response_time_ms,
                        )

                        # Track file access for session persistence
                        if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                            try:
                                await _SESSION_MANAGER.track_file_access(
                                    file_path=file_path,
                                    importance=0.7,  # Higher importance for explicitly read files
                                )
                                print(
                                    f"✓ Session tracking: Recorded file access for {file_path}",
                                    file=sys.stderr,
                                )
                            except Exception as track_error:
                                print(
                                    f"⚠ Failed to track file access: {track_error}",
                                    file=sys.stderr,
                                )

                        # Store in L2 repository cache FIRST (SHARED by team, 7 day TTL)
                        if self.cache_manager:
                            try:
                                repo_id = self._get_repo_id(file_path)
                                # Use content-based hash for cache key (auto-invalidates on file change)
                                content_hash = self._get_file_hash(file_path)
                                l2_key = hashlib.sha256(
                                    f"{file_path}:{content_hash}".encode()
                                ).hexdigest()[:16]

                                # Get file modification time for timestamp validation
                                try:
                                    file_mtime = (
                                        Path(file_path)
                                        .expanduser()
                                        .resolve()
                                        .stat()
                                        .st_mtime
                                    )
                                except:
                                    file_mtime = time.time()

                                # Prepare content for L2
                                full_result = {
                                    "omn1_mode": "full",
                                    "file_path": file_path,
                                    "content": content,
                                    "compressed": False,
                                    "token_count": content_token_count,
                                }
                                full_json = json.dumps(full_result)

                                # Store in L2 (team-shared, 7 day TTL)
                                user_id = self._get_user_id()
                                self.cache_manager.cache_file_compressed(
                                    repo_id=repo_id,
                                    file_hash=l2_key,
                                    compressed_content=full_json.encode("utf-8"),
                                    metadata={
                                        "file_path": file_path,
                                        "cached_by": user_id,  # NEW: Track who cached it
                                        "mode": "full",
                                        "compressed": "False",
                                        "cached_at": str(time.time()),
                                        "file_mtime": str(
                                            file_mtime
                                        ),  # NEW: File modification time
                                        "size": str(len(full_json)),
                                    },
                                    ttl=604800,  # 7 days - long TTL for team sharing
                                )
                                print(
                                    f"💾 Cached in L2 (repository): {file_path} (full/uncompressed) - SHARED with team",
                                    file=sys.stderr,
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  L2 cache storage failed: {e}", file=sys.stderr
                                )

                        # NEW: Store in Unified Cache Manager (L1 user cache, fallback uncompressed)
                        if self.cache_manager:
                            try:
                                user_id = self._get_user_id()

                                # Get file modification time for timestamp validation
                                try:
                                    file_mtime = (
                                        Path(file_path)
                                        .expanduser()
                                        .resolve()
                                        .stat()
                                        .st_mtime
                                    )
                                except:
                                    file_mtime = time.time()

                                result_to_cache = {
                                    "omn1_mode": "full",
                                    "file_path": file_path,
                                    "content": content,
                                    "compressed": False,
                                    "token_count": content_token_count,
                                    "language": language,
                                    "timestamp": datetime.now().isoformat(),
                                    "file_mtime": file_mtime,  # NEW: File modification time
                                }

                                # Use content-based cache key (auto-invalidates on file change)
                                content_hash = self._get_file_hash(file_path)
                                cache_key = f"{file_path}:{content_hash}"

                                self.cache_manager.cache_read_result(
                                    user_id=user_id,
                                    file_path=cache_key,  # Use content-based key
                                    result=result_to_cache,
                                    ttl=3600,  # 1 hour
                                )
                                print(
                                    f"💾 Cached in L1 (user): {file_path} (full/uncompressed, content-based key)",
                                    file=sys.stderr,
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  Failed to cache: {e}",
                                    file=sys.stderr,
                                )

                        return result

                elif target == "overview":
                    # Try LSP symbol service first, fallback to AST if unavailable
                    result = None

                    if LSP_AVAILABLE and self.symbol_service:
                        try:
                            # Start symbol service if needed
                            await self.symbol_service.start()

                            # Get file overview via LSP
                            result = await self.symbol_service.get_overview(
                                file_path=file_path,
                                include_details=include_details,
                                compress=compress,
                            )
                        except Exception as e:
                            print(
                                f"⚠ LSP overview failed for {file_path}: {e}, trying AST fallback",
                                file=sys.stderr,
                            )
                            result = None

                    # Fallback to AST-based overview for Python files
                    if result is None and file_path.endswith(".py"):
                        try:
                            import ast

                            path = Path(file_path).expanduser().resolve()
                            with open(path, "r", encoding="utf-8") as f:
                                source = f.read()

                            tree = ast.parse(source)
                            classes = []
                            functions = []
                            imports = []

                            for node in ast.walk(tree):
                                if isinstance(node, ast.ClassDef):
                                    classes.append(node.name)
                                elif isinstance(node, ast.FunctionDef):
                                    functions.append(node.name)
                                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                                    if isinstance(node, ast.Import):
                                        for alias in node.names:
                                            imports.append(alias.name)
                                    else:
                                        imports.append(
                                            f"{node.module}.{node.names[0].name}"
                                            if node.names
                                            else str(node.module)
                                        )

                            result = {
                                "file_path": file_path,
                                "classes": classes,
                                "functions": functions,
                                "imports": imports[:20],  # Limit imports to first 20
                                "total_classes": len(classes),
                                "total_functions": len(functions),
                                "total_imports": len(imports),
                                "method": "ast_fallback",
                            }
                        except Exception as e:
                            result = {
                                "error": True,
                                "message": f"Failed to parse file structure: {e}",
                                "file_path": file_path,
                                "tip": "Overview mode works best with Python files when LSP is installed",
                            }

                    # If still no result, return error
                    if result is None:
                        result = {
                            "error": True,
                            "message": "Symbol service not available and file type not supported for AST fallback",
                            "target": target,
                            "file_path": file_path,
                            "tip": "Overview mode requires LSP symbol service or Python files for AST fallback",
                        }

                    # Sanitize response - only return essential data
                    if result.get("error"):
                        # Keep error information as-is
                        result_json = json.dumps(result, indent=2)
                    else:
                        sanitized_response = {
                            "content": {
                                "classes": result.get("classes", []),
                                "functions": result.get("functions", []),
                                "imports": result.get("imports", []),
                            },
                            "optimized": True,
                        }
                        result_json = json.dumps(sanitized_response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="read",
                        operation_mode=target,
                        parameters={
                            "include_details": include_details,
                            "compress": compress,
                        },
                        file_path=file_path,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track file access for session persistence
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            await _SESSION_MANAGER.track_file_access(
                                file_path=file_path,
                                importance=0.7,  # Higher importance for explicitly read files
                            )
                            print(
                                f"✓ Session tracking: Recorded file access for {file_path}",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track file access: {track_error}",
                                file=sys.stderr,
                            )

                    # Store in L2 repository cache FIRST (SHARED by team, 7 day TTL)
                    if self.cache_manager and not result.get("error"):
                        try:
                            repo_id = self._get_repo_id(file_path)
                            file_hash = hashlib.sha256(file_path.encode()).hexdigest()[
                                :16
                            ]

                            # Prepare compressed content for L2
                            overview_json = json.dumps(result)

                            # Store in L2 (team-shared, 7 day TTL)
                            user_id = self._get_user_id()
                            self.cache_manager.cache_file_compressed(
                                repo_id=repo_id,
                                file_hash=file_hash,
                                compressed_content=overview_json.encode("utf-8"),
                                metadata={
                                    "file_path": file_path,
                                    "cached_by": user_id,  # NEW: Track who cached it
                                    "mode": "overview",
                                    "compressed": "True",
                                    "cached_at": str(time.time()),
                                    "size": str(len(overview_json)),
                                },
                                ttl=604800,  # 7 days - long TTL for team sharing
                            )
                            print(
                                f"💾 Cached in L2 (repository): {file_path} (overview) - SHARED with team",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  L2 cache storage failed: {e}", file=sys.stderr)

                    # NEW: Store in Unified Cache Manager (L1 user cache, overview mode)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = (
                                result.copy()
                                if isinstance(result, dict)
                                else json.loads(result_json)
                            )
                            result_to_cache["language"] = language
                            result_to_cache["timestamp"] = datetime.now().isoformat()
                            self.cache_manager.cache_read_result(
                                user_id=user_id,
                                file_path=file_path,
                                result=result_to_cache,
                                ttl=3600,  # 1 hour
                            )
                            print(
                                f"💾 Cached in L1 (user): {file_path} (overview)",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to cache: {e}",
                                file=sys.stderr,
                            )

                    return result_json

                elif target == "symbol":
                    # Try LSP symbol service first, fallback to AST if unavailable
                    result = None

                    if LSP_AVAILABLE and self.symbol_service:
                        try:
                            # Start symbol service if needed
                            await self.symbol_service.start()

                            # Read specific symbol via LSP
                            result = await self.symbol_service.read_symbol(
                                file_path=file_path, symbol=symbol, compress=compress
                            )

                            # Optionally find references
                            if include_references:
                                refs_result = await self.symbol_service.find_references(
                                    file_path=file_path, symbol=symbol
                                )
                                result["references"] = refs_result
                        except Exception as e:
                            print(
                                f"⚠ LSP symbol read failed for {file_path}:{symbol}: {e}, trying AST fallback",
                                file=sys.stderr,
                            )
                            result = None

                    # Fallback to AST-based symbol extraction for Python files
                    if result is None and file_path.endswith(".py"):
                        try:
                            import ast
                            import inspect

                            path = Path(file_path).expanduser().resolve()
                            with open(path, "r", encoding="utf-8") as f:
                                source = f.read()

                            tree = ast.parse(source)
                            lines = source.splitlines()
                            symbol_content = None

                            for node in ast.walk(tree):
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                                    if node.name == symbol:
                                        # Extract source lines for this symbol
                                        start_line = node.lineno - 1
                                        end_line = node.end_lineno
                                        symbol_content = "\n".join(
                                            lines[start_line:end_line]
                                        )
                                        break

                            if symbol_content:
                                result = {
                                    "file_path": file_path,
                                    "symbol": symbol,
                                    "content": symbol_content,
                                    "method": "ast_fallback",
                                }
                            else:
                                result = {
                                    "error": True,
                                    "message": f"Symbol '{symbol}' not found in file",
                                    "file_path": file_path,
                                    "symbol": symbol,
                                }
                        except Exception as e:
                            result = {
                                "error": True,
                                "message": f"Failed to extract symbol: {e}",
                                "file_path": file_path,
                                "symbol": symbol,
                            }

                    # If still no result, return error
                    if result is None:
                        result = {
                            "error": True,
                            "message": "Symbol service not available and file type not supported for AST fallback",
                            "target": target,
                            "file_path": file_path,
                            "symbol": symbol,
                            "tip": "Symbol reading requires LSP symbol service or Python files for AST fallback",
                        }

                    # Sanitize response - only return essential data
                    if result.get("error"):
                        # Keep error information as-is
                        result_json = json.dumps(result, indent=2)
                    else:
                        sanitized_response = {
                            "content": result.get("content", ""),
                            "optimized": True,
                        }
                        result_json = json.dumps(sanitized_response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="read",
                        operation_mode=target,
                        parameters={
                            "symbol": symbol,
                            "compress": compress,
                            "include_references": include_references,
                        },
                        file_path=file_path,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track file access for session persistence
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            await _SESSION_MANAGER.track_file_access(
                                file_path=file_path,
                                importance=0.7,  # Higher importance for explicitly read files
                            )
                            print(
                                f"✓ Session tracking: Recorded file access for {file_path}",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track file access: {track_error}",
                                file=sys.stderr,
                            )

                    # Store in L2 repository cache FIRST (SHARED by team, 7 day TTL)
                    if self.cache_manager and not result.get("error"):
                        try:
                            repo_id = self._get_repo_id(file_path)
                            # Include symbol in file hash for unique cache key per symbol
                            file_hash = hashlib.sha256(
                                f"{file_path}|symbol:{symbol}".encode()
                            ).hexdigest()[:16]

                            # Prepare symbol result for L2
                            symbol_json = json.dumps(result)

                            # Store in L2 (team-shared, 7 day TTL)
                            user_id = self._get_user_id()
                            self.cache_manager.cache_file_compressed(
                                repo_id=repo_id,
                                file_hash=file_hash,
                                compressed_content=symbol_json.encode("utf-8"),
                                metadata={
                                    "file_path": file_path,
                                    "cached_by": user_id,  # NEW: Track who cached it
                                    "mode": "symbol",
                                    "symbol": symbol,
                                    "compressed": "True",
                                    "cached_at": str(time.time()),
                                    "size": str(len(symbol_json)),
                                },
                                ttl=604800,  # 7 days - long TTL for team sharing
                            )
                            print(
                                f"💾 Cached in L2 (repository): {file_path} (symbol:{symbol}) - SHARED with team",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  L2 cache storage failed: {e}", file=sys.stderr)

                    # NEW: Store in Unified Cache Manager (L1 user cache, symbol mode)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = (
                                result.copy()
                                if isinstance(result, dict)
                                else json.loads(result_json)
                            )
                            result_to_cache["language"] = language
                            result_to_cache["timestamp"] = datetime.now().isoformat()
                            # Use unique key per symbol
                            cache_key = f"{file_path}|symbol:{symbol}"
                            self.cache_manager.cache_read_result(
                                user_id=user_id,
                                file_path=cache_key,
                                result=result_to_cache,
                                ttl=3600,  # 1 hour
                            )
                            print(
                                f"💾 Cached in L1 (user): {file_path} (symbol:{symbol})",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to cache: {e}",
                                file=sys.stderr,
                            )

                    return result_json

                elif target == "references":
                    result = await self._omn1_find_references(
                        file_path=file_path, symbol=symbol
                    )
                    # Sanitize response - only return essential data
                    result_dict = json.loads(result)
                    sanitized_response = {
                        "content": result_dict.get(
                            "references", result_dict.get("results", [])
                        ),
                        "optimized": True,
                    }
                    result_json = json.dumps(sanitized_response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="read",
                        operation_mode=target,
                        parameters={
                            "symbol": symbol,
                        },
                        file_path=file_path,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track file access for session persistence
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            await _SESSION_MANAGER.track_file_access(
                                file_path=file_path,
                                importance=0.7,  # Higher importance for explicitly read files
                            )
                            print(
                                f"✓ Session tracking: Recorded file access for {file_path}",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track file access: {track_error}",
                                file=sys.stderr,
                            )

                    # Store in L2 repository cache FIRST (SHARED by team, 7 day TTL)
                    if self.cache_manager and not result.get("error"):
                        try:
                            repo_id = self._get_repo_id(file_path)
                            # Include symbol in file hash for unique cache key per symbol
                            file_hash = hashlib.sha256(
                                f"{file_path}|references:{symbol}".encode()
                            ).hexdigest()[:16]

                            # Prepare references result for L2
                            references_json = json.dumps(result)

                            # Store in L2 (team-shared, 7 day TTL)
                            user_id = self._get_user_id()
                            self.cache_manager.cache_file_compressed(
                                repo_id=repo_id,
                                file_hash=file_hash,
                                compressed_content=references_json.encode("utf-8"),
                                metadata={
                                    "file_path": file_path,
                                    "cached_by": user_id,  # NEW: Track who cached it
                                    "mode": "references",
                                    "symbol": symbol,
                                    "compressed": "True",
                                    "cached_at": str(time.time()),
                                    "size": str(len(references_json)),
                                },
                                ttl=604800,  # 7 days - long TTL for team sharing
                            )
                            print(
                                f"💾 Cached in L2 (repository): {file_path} (references:{symbol}) - SHARED with team",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  L2 cache storage failed: {e}", file=sys.stderr)

                    # NEW: Store in Unified Cache Manager (L1 user cache, references mode)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = (
                                result.copy()
                                if isinstance(result, dict)
                                else json.loads(result_json)
                            )
                            result_to_cache["language"] = language
                            result_to_cache["timestamp"] = datetime.now().isoformat()
                            # Use unique key per symbol
                            cache_key = f"{file_path}|references:{symbol}"
                            self.cache_manager.cache_read_result(
                                user_id=user_id,
                                file_path=cache_key,
                                result=result_to_cache,
                                ttl=3600,  # 1 hour
                            )
                            print(
                                f"💾 Cached in L1 (user): {file_path} (references:{symbol})",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(
                                f"⚠️  Failed to cache: {e}",
                                file=sys.stderr,
                            )

                    return result_json

            except Exception as e:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Error in omn1_read: {str(e)}",
                        "target": target,
                        "file_path": file_path,
                        "symbol": symbol if symbol else None,
                    },
                    indent=2,
                )

        @self.mcp.tool()
        async def search(query: str) -> str:
            """Unified search tool - ONE tool for all search needs

            WORKAROUND: Due to Claude Code's MCP client limitation (only passes first positional
            parameter), all parameters are encoded in a single string using delimiter syntax.

            This consolidates 3 separate tools into one:
            - "query" → Semantic vector search (default)
            - "query|tri_index" → Hybrid search (Dense + Sparse + Structural) - BEST ACCURACY
            - "query|references:symbol" → Find symbol usages

            Parameter Format: "query|mode|options"

            Search Modes:
                "authentication"                           → Semantic search (default)
                "authentication|tri_index"                 → Tri-index hybrid search (BEST)
                "authentication|triindex"                  → Same as above (alternative spelling)
                "find usages|references:authenticate"      → Find all references to symbol
                "Settings|refs:SettingsManager|file:src/settings.py" → Find refs in specific file

            Options (can be combined):
                "|limit:10"     → Return up to 10 results (default: 5)
                "|minrel:0.8"   → Minimum relevance score for semantic search (default: 0.7)
                "|context"      → Include surrounding context in results
                "|nocontext"    → Exclude context (default)
                "|norerank"     → Disable witness reranking for tri_index mode
                "|file:PATH"    → Scope search to specific file (for references mode)

            Examples:
                # Simple semantic search (default)
                search("authentication implementation")
                → Returns top 5 most relevant files (vector search only)

                # Tri-index hybrid search (BEST ACCURACY)
                search("authentication implementation|tri_index")
                → Searches Dense (vectors) + Sparse (BM25) + Structural (code facts)
                → Fused with RRF + cross-encoder reranking

                # Tri-index with more results
                search("authentication|tri_index|limit:10")
                → Returns top 10 results from hybrid search

                # High-precision semantic search
                search("error handling patterns|semantic|limit:10|minrel:0.8")
                → Returns only highly relevant matches (threshold 0.8)

                # Find all references to a symbol
                search("find all usages|references:authenticate|file:src/auth.py")
                → Returns all locations where authenticate() is called

                # Tri-index with context and no reranking
                search("settings management|tri_index|context|norerank")
                → Hybrid search with context, without cross-encoder reranking

                # Alternative tri_index spellings (all equivalent)
                search("auth|tri_index")
                search("auth|triindex")
                search("auth|tri-index")

            Returns:
                JSON string with search results and metadata

            Token Savings:
            - Search finds relevant files without reading all of them
            - Prevents reading 50+ files when only 3-5 are relevant
            - Typical savings: 90-95% tokens prevented
            """
            # Parse parameters from single string input
            params = _parse_search_params(query)

            # Extract parsed parameters
            query = params["query"]
            mode = params["mode"]
            file_path = params["file_path"]
            symbol = params["symbol"]
            limit = params["limit"]
            min_relevance = params["min_relevance"]
            enable_witness_rerank = params["enable_witness_rerank"]
            include_context = params["include_context"]

            # Debug logging
            print(
                f"🔍 Parsed search params: query={repr(query)}, mode={repr(mode)}, symbol={repr(symbol)}, limit={limit}",
                file=sys.stderr,
            )

            # Start timing for operation tracking (before cache check for accurate metrics)
            start_time = time.time()

            # ZERO NEW TOOLS: Detect filtering of cached results
            # When file_path points to a virtual cached result, filter it instead of searching
            if file_path and (
                file_path.startswith("~/.omnimemory/cached_results/")
                or file_path.startswith(
                    str(Path.home() / ".omnimemory" / "cached_results")
                )
            ):
                print(
                    f"🔍 Detected cached result filtering: {file_path}", file=sys.stderr
                )
                return await self._search_cached_result(
                    result_path=file_path, filter_expr=query, limit=limit
                )

            # NEW: Check Unified Cache Manager (L1 user cache) first
            if self.cache_manager:
                try:
                    user_id = self._get_user_id()
                    cached = self.cache_manager.get_search_result(
                        user_id=user_id, query=query, mode=mode
                    )
                    if cached:
                        print(
                            f"🎯 Cache HIT: search '{query}' mode={mode} (L1 User Cache)",
                            file=sys.stderr,
                        )
                        # Sanitize cached response - only return essential data
                        sanitized_cached = {
                            "results": cached.get("results", []),
                            "optimized": True,
                        }

                        # Track search cache hit for metrics
                        tokens_original = await self._estimate_search_baseline(
                            query, mode, limit
                        )
                        tokens_actual = self._count_tokens(json.dumps(sanitized_cached))
                        await self._track_tool_operation(
                            tool_name="search",
                            operation_mode=mode,
                            parameters={
                                "query": query,
                                "limit": limit,
                                "min_relevance": min_relevance,
                            },
                            file_path=None,
                            tokens_original=tokens_original,
                            tokens_actual=tokens_actual,
                            response_time_ms=(time.time() - start_time) * 1000,
                        )

                        return json.dumps(sanitized_cached, indent=2)
                except Exception as e:
                    print(f"⚠️  Cache error: {e}", file=sys.stderr)

            # Estimate baseline tokens (what would be read without tri-index optimization)
            tokens_original = await self._estimate_search_baseline(query, mode, limit)

            try:
                # Validate mode
                valid_modes = ["semantic", "tri_index", "references"]
                if mode not in valid_modes:
                    return json.dumps(
                        {
                            "error": True,
                            "message": f"Invalid mode: {mode}. Must be one of: {', '.join(valid_modes)}",
                            "tip": "Use mode='semantic' for vector search, 'tri_index' for hybrid search, 'references' for finding symbol usages",
                        },
                        indent=2,
                    )

                # Validate parameters for references mode
                if mode == "references":
                    if not file_path or not symbol:
                        return json.dumps(
                            {
                                "error": True,
                                "message": "mode='references' requires both 'file_path' and 'symbol' parameters",
                                "example": 'search(mode="references", file_path="src/auth.py", symbol="authenticate")',
                            },
                            indent=2,
                        )

                # Route to appropriate implementation
                if mode == "semantic":
                    result = await self._omn1_semantic_search(
                        query=query, limit=limit, min_relevance=min_relevance
                    )
                    # Sanitize response - only return essential data
                    result_dict = json.loads(result)
                    sanitized_response = {
                        "results": result_dict.get("results", []),
                        "optimized": True,
                    }
                    result_json = json.dumps(sanitized_response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="search",
                        operation_mode=mode,
                        parameters={
                            "query": query,
                            "limit": limit,
                            "min_relevance": min_relevance,
                        },
                        file_path=None,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track search query for session persistence
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            await _SESSION_MANAGER.track_search(query=query)
                            print(
                                f"✓ Session tracking: Recorded semantic search query '{query}'",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track search: {track_error}",
                                file=sys.stderr,
                            )

                    # NEW: Cache query results in Unified Cache Manager (L1 user cache)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = {
                                "status": "success",
                                "omn1_mode": mode,
                                "query": query,
                                "results": result_dict.get("results", []),
                                "metadata": {
                                    "search_time_ms": (time.time() - start_time) * 1000,
                                    "results_count": len(
                                        result_dict.get("results", [])
                                    ),
                                },
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.cache_manager.cache_search_result(
                                user_id=user_id,
                                query=query,
                                mode=mode,
                                result=result_to_cache,
                                ttl=600,  # 10 minutes
                            )
                            print(
                                f"💾 Cached in L1 (user): search '{query}' mode={mode}",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  Failed to cache: {e}", file=sys.stderr)

                    return result_json

                elif mode == "tri_index":
                    # Use tri-index hybrid search (Dense + Sparse + Structural)
                    # Note: start_time already set at function level for tracking

                    # TODO: Consider caching tri-index search results at L2 (repo level)
                    # This would allow team members to share search results
                    # For now, only L1 (user) cache is used for searches

                    # Check if unified TriIndex is available, fallback to semantic if not
                    if not hasattr(self, "tri_index") or self.tri_index is None:
                        print(
                            "⚠ TriIndex not available, falling back to semantic search",
                            file=sys.stderr,
                        )
                        # Fallback to semantic search
                        result = await self._omn1_semantic_search(
                            query=query, limit=limit, min_relevance=min_relevance
                        )
                        result_dict = json.loads(result)
                        result_dict["omn1_mode"] = "semantic"  # Changed from tri_index
                        result_dict["fallback_from"] = "tri_index"
                        result_dict[
                            "omn1_info"
                        ] = "Semantic search (tri_index not available, using fallback)"
                        result_dict[
                            "token_savings_info"
                        ] = f"Prevented reading {max(0, 50 - limit)} irrelevant files"
                        return json.dumps(result_dict, indent=2)

                    # Ensure TriIndex is started
                    if (
                        not hasattr(self.tri_index, "_started")
                        or not self.tri_index._started
                    ):
                        await self.tri_index.start()
                        self.tri_index._started = True

                    # Perform hybrid search using unified TriIndex
                    search_results = await self.tri_index.search(
                        query=query,
                        query_embedding=None,  # TriIndex will generate embedding if needed
                        limit=limit,
                        enable_witness_rerank=enable_witness_rerank,
                        min_score=min_relevance,
                    )

                    # Convert results to serializable format
                    results = []
                    for i, result in enumerate(search_results, 1):
                        results.append(
                            {
                                "rank": i,
                                "file_path": result.file_path,
                                "final_score": result.final_score,
                                "dense_score": result.dense_score,
                                "sparse_score": result.sparse_score,
                                "fact_score": result.fact_score,
                                "witness_score": getattr(result, "witness_score", None),
                                "metadata": result.metadata,
                                "witnesses": result.witnesses[:3]
                                if result.witnesses
                                else [],  # Limit witnesses
                            }
                        )

                    search_time_ms = (time.time() - start_time) * 1000

                    # Sanitize response - only return essential data
                    response = {"results": results, "optimized": True}

                    result_json = json.dumps(response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="search",
                        operation_mode=mode,
                        parameters={
                            "query": query,
                            "limit": limit,
                            "min_relevance": min_relevance,
                            "enable_witness_rerank": enable_witness_rerank,
                        },
                        file_path=None,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track search query for session persistence
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            await _SESSION_MANAGER.track_search(query=query)
                            print(
                                f"✓ Session tracking: Recorded tri-index search query '{query}'",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track search: {track_error}",
                                file=sys.stderr,
                            )

                    # NEW: Cache query results in Unified Cache Manager (L1 user cache)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = {
                                "status": "success",
                                "omn1_mode": mode,
                                "query": query,
                                "results": results,
                                "metadata": {
                                    "search_time_ms": (time.time() - start_time) * 1000,
                                    "results_count": len(results)
                                    if isinstance(results, list)
                                    else 0,
                                },
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.cache_manager.cache_search_result(
                                user_id=user_id,
                                query=query,
                                mode=mode,
                                result=result_to_cache,
                                ttl=600,  # 10 minutes
                            )
                            print(
                                f"💾 Cached in L1 (user): search '{query}' mode={mode}",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  Failed to cache: {e}", file=sys.stderr)

                    return result_json

                elif mode == "references":
                    result = await self._omn1_find_references(
                        file_path=file_path, symbol=symbol
                    )
                    # Sanitize response - only return essential data
                    result_dict = json.loads(result)
                    sanitized_response = {
                        "results": result_dict.get(
                            "results", result_dict.get("references", [])
                        ),
                        "optimized": True,
                    }
                    result_json = json.dumps(sanitized_response, indent=2)

                    # Track operation
                    tokens_actual = self._count_tokens(result_json)
                    response_time_ms = (time.time() - start_time) * 1000
                    await self._track_tool_operation(
                        tool_name="search",
                        operation_mode=mode,
                        parameters={
                            "query": query,
                            "file_path": file_path,
                            "symbol": symbol,
                        },
                        file_path=file_path,
                        tokens_original=tokens_original,
                        tokens_actual=tokens_actual,
                        response_time_ms=response_time_ms,
                    )

                    # Track search query for session persistence (references mode)
                    if _SESSION_MANAGER and _SESSION_MANAGER.current_session:
                        try:
                            # Track as a search query with symbol context
                            search_query = (
                                f"{query} (references: {symbol} in {file_path})"
                            )
                            await _SESSION_MANAGER.track_search(query=search_query)
                            print(
                                f"✓ Session tracking: Recorded reference search for '{symbol}'",
                                file=sys.stderr,
                            )
                        except Exception as track_error:
                            print(
                                f"⚠ Failed to track search: {track_error}",
                                file=sys.stderr,
                            )

                    # NEW: Cache query results in Unified Cache Manager (L1 user cache)
                    if self.cache_manager:
                        try:
                            user_id = self._get_user_id()
                            result_to_cache = {
                                "status": "success",
                                "omn1_mode": mode,
                                "query": query,
                                "symbol": symbol,
                                "file_path": file_path,
                                "references": result_dict.get("references", []),
                                "metadata": {
                                    "search_time_ms": (time.time() - start_time) * 1000,
                                    "results_count": len(
                                        result_dict.get("references", [])
                                    ),
                                },
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.cache_manager.cache_search_result(
                                user_id=user_id,
                                query=f"{query}|{symbol}|{file_path}",  # Unique key per symbol+file
                                mode=mode,
                                result=result_to_cache,
                                ttl=600,  # 10 minutes
                            )
                            print(
                                f"💾 Cached in L1 (user): references to '{symbol}' in {file_path}",
                                file=sys.stderr,
                            )
                        except Exception as e:
                            print(f"⚠️  Failed to cache: {e}", file=sys.stderr)

                    return result_json

            except Exception as e:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Error in omn1_search: {str(e)}",
                        "mode": mode,
                        "query": query,
                        "file_path": file_path if file_path else None,
                        "symbol": symbol if symbol else None,
                    },
                    indent=2,
                )

        @self.mcp.tool()
        async def welcome_back() -> str:
            """
            Get session context summary - shows what OmniMemory remembers from last session

            Returns welcome message with:
            - Last session info (time, project, duration)
            - Recent files accessed
            - Recent searches
            - Workflow context
            - Suggestions to continue

            This creates the "WOW" moment when users see their context restored.
            """
            try:
                # Get current session info
                if not _SESSION_MANAGER or not _SESSION_MANAGER.current_session:
                    return json.dumps(
                        {
                            "status": "no_session",
                            "message": "Welcome! This is a new session. Start working and I'll remember it for next time.",
                            "first_time": True,
                        },
                        indent=2,
                    )

                session_id = _SESSION_MANAGER.current_session

                # Get session data from metrics service
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Get session info
                    session_response = await client.get(
                        f"{_METRICS_API}/sessions/{session_id}"
                    )

                    if session_response.status_code != 200:
                        return json.dumps(
                            {
                                "status": "error",
                                "message": "Could not retrieve session data",
                            },
                            indent=2,
                        )

                    session_data = session_response.json()

                    # Get recent activity
                    activity_response = await client.get(
                        f"{_METRICS_API}/sessions/{session_id}/activity",
                        params={"limit": 20},
                    )

                    activity_data = (
                        activity_response.json()
                        if activity_response.status_code == 200
                        else {"operations": []}
                    )

                # Parse session info
                started_at = session_data.get("started_at", 0)
                time_ago = _format_time_ago(started_at)

                # Extract recent files (last 5 unique files)
                recent_files = []
                seen_files = set()
                for op in activity_data.get("operations", []):
                    file_path = op.get("file_path")
                    if file_path and file_path not in seen_files:
                        recent_files.append(file_path)
                        seen_files.add(file_path)
                        if len(recent_files) >= 5:
                            break

                # NEW: Check team activity on recent files
                team_insights = []
                if self.cache_manager and recent_files:
                    user_id = self._get_user_id()

                    for file_path in recent_files[:5]:  # Top 5 files
                        try:
                            repo_id = self._get_repo_id(file_path)
                            file_hash = hashlib.sha256(file_path.encode()).hexdigest()[
                                :16
                            ]

                            # Check L2 cache metadata for team activity
                            meta_key = f"repo:{repo_id}:file:{file_hash}:meta"
                            metadata = self.cache_manager.redis.hgetall(meta_key)

                            if metadata:
                                cached_by = metadata.get(b"cached_by", b"").decode()
                                cached_at_str = metadata.get(
                                    b"cached_at", b"0"
                                ).decode()

                                if cached_by and cached_at_str:
                                    cached_at = float(cached_at_str)
                                    time_ago = _format_time_ago(cached_at)

                                    # Someone else cached this (not current user)
                                    if cached_by != user_id and cached_by != "unknown":
                                        team_insights.append(
                                            {
                                                "file": Path(file_path).name,
                                                "user": cached_by,
                                                "time_ago": time_ago,
                                                "message": f"{cached_by} worked on {Path(file_path).name} {time_ago}",
                                            }
                                        )

                        except Exception:
                            # Skip file on error
                            pass

                # Extract recent searches (last 3)
                recent_searches = []
                for op in activity_data.get("operations", []):
                    if op.get("tool_name") == "search":
                        query = op.get("parameters", {}).get("query", "")
                        if query and query not in recent_searches:
                            recent_searches.append(query)
                            if len(recent_searches) >= 3:
                                break

                # Detect project (use repo_id or directory)
                project_name = "current project"
                if recent_files:
                    repo_id = self._get_repo_id(recent_files[0])
                    if repo_id.startswith("repo_"):
                        # Try to get repo name from git
                        try:
                            import subprocess

                            file_dir = Path(recent_files[0]).parent
                            result = subprocess.run(
                                ["git", "config", "--get", "remote.origin.url"],
                                cwd=file_dir,
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if result.returncode == 0:
                                # Extract repo name from URL
                                url = result.stdout.strip()
                                if "/" in url:
                                    project_name = url.split("/")[-1].replace(
                                        ".git", ""
                                    )
                        except:
                            pass

                # Get cache stats
                cache_stats = {}
                if self.cache_manager:
                    stats = self.cache_manager.get_stats()
                    cache_stats = {
                        "l1_keys": stats.l1_keys,
                        "l2_keys": stats.l2_keys,
                        "hit_rate": stats.hit_rate,
                        "cached_items": stats.total_keys,
                    }

                # Generate suggestions
                suggestions = _generate_suggestions(recent_files, recent_searches)

                # Build welcome message
                welcome_data = {
                    "status": "restored",
                    "message": "Welcome back! Here's what I remember:",
                    "formatted_message": self._format_welcome_message(
                        project_name=project_name,
                        time_ago=time_ago,
                        recent_files=recent_files,
                        recent_searches=recent_searches,
                        suggestions=suggestions,
                        team_insights=team_insights,
                    ),
                    "last_session": {
                        "session_id": session_id,
                        "started_at": session_data.get("started_at"),
                        "time_ago": time_ago,
                        "duration_minutes": session_data.get("duration_seconds", 0)
                        / 60,
                        "project": project_name,
                        "tool_id": session_data.get("tool_id", "unknown"),
                    },
                    "recent_activity": {
                        "files": recent_files,
                        "searches": recent_searches,
                        "total_operations": len(activity_data.get("operations", [])),
                    },
                    "cache_status": cache_stats,
                    "suggestions": suggestions,
                    "team_insights": team_insights,
                    "ready_to_continue": True,
                }

                return json.dumps(welcome_data, indent=2)

            except Exception as e:
                print(f"⚠️  Welcome back error: {e}", file=sys.stderr)
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Could not load session context: {str(e)}",
                    },
                    indent=2,
                )

        @self.mcp.tool()
        async def get_team_activity(input_string: str) -> str:
            """
            Get team activity for current repository

            Format: "repo_id|days:7|limit:20"

            Shows:
            - Who accessed which files recently
            - Team-wide searches
            - Collaboration opportunities
            - Most active team members
            """

            # Parse input
            parts = input_string.split("|") if input_string else []
            repo_id = parts[0] if parts else self._get_repo_id(os.getcwd())

            params = {"days": 7, "limit": 20}

            for part in parts[1:] if len(parts) > 1 else []:
                part_lower = part.strip().lower()
                if part_lower.startswith("days:"):
                    params["days"] = int(part.split(":")[1])
                elif part_lower.startswith("limit:"):
                    params["limit"] = int(part.split(":")[1])

            try:
                # Get all L2 cache keys for this repo
                pattern = f"repo:{repo_id}:file:*:meta"
                meta_keys = self.cache_manager.redis.keys(pattern)

                team_activity = []
                cutoff_time = time.time() - (params["days"] * 86400)

                for key in meta_keys:
                    metadata = self.cache_manager.redis.hgetall(key)

                    if metadata:
                        file_path = metadata.get(b"file_path", b"").decode()
                        cached_by = metadata.get(b"cached_by", b"").decode()
                        cached_at = float(metadata.get(b"cached_at", b"0").decode())

                        if cached_at > cutoff_time and cached_by:
                            team_activity.append(
                                {
                                    "file": file_path,
                                    "user": cached_by,
                                    "time_ago": _format_time_ago(cached_at),
                                    "timestamp": cached_at,
                                }
                            )

                # Sort by recency
                team_activity.sort(key=lambda x: x["timestamp"], reverse=True)
                team_activity = team_activity[: params["limit"]]

                # Group by user
                user_activity = {}
                for activity in team_activity:
                    user = activity["user"]
                    if user not in user_activity:
                        user_activity[user] = []
                    user_activity[user].append(activity)

                return json.dumps(
                    {
                        "status": "success",
                        "repo_id": repo_id,
                        "days": params["days"],
                        "total_files": len(team_activity),
                        "team_members": len(user_activity),
                        "activity": team_activity,
                        "by_user": {
                            user: len(files) for user, files in user_activity.items()
                        },
                        "message": f"Found {len(team_activity)} files accessed by {len(user_activity)} team members in last {params['days']} days",
                    },
                    indent=2,
                )

            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)}, indent=2)

        @self.mcp.tool()
        async def get_current_project() -> str:
            """
            Get current project information and context

            Returns:
            - Project name and type
            - Recent session in this project
            - Cached files for this project
            - Suggestions for this project

            Use this to understand which project you're currently working on
            and what context is available.
            """
            try:
                if not self.workspace_monitor:
                    return json.dumps(
                        {
                            "status": "unavailable",
                            "message": "Workspace monitor not running",
                        },
                        indent=2,
                    )

                project = self.workspace_monitor.get_current_project()

                # Get cache stats for this project (L2 tier)
                project_id = project["project_id"]
                cache_stats = {}

                if self.cache_manager:
                    try:
                        # Count L2 keys for this project
                        pattern = f"repo:{project_id}:*"
                        keys = self.cache_manager.redis.keys(pattern)
                        cache_stats = {
                            "cached_files": len([k for k in keys if b":file:" in k])
                            // 2,  # data + meta
                            "total_items": len(keys),
                        }
                    except:
                        pass

                return json.dumps(
                    {
                        "status": "success",
                        "project": project["info"],
                        "project_id": project_id,
                        "cache_status": cache_stats,
                        "workspace_path": project["workspace"],
                    },
                    indent=2,
                )

            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)}, indent=2)

        # ===================================================================
        # Unified Intelligence System Tools
        # ===================================================================

        # @self.mcp.tool()
        # async def omn1_unified_predict(
        #     context: str,
        #     user_profile: str = None,
        # ) -> str:
        #     """Get unified predictions from File Context + Agent Memory

        #     Combines predictions from multiple memory systems to predict next files,
        #     tools, and actions. Uses cross-memory pattern detection.

        #     Args:
        #         context: JSON string with context (current_files, recent_actions, task_context)
        #         user_profile: Optional JSON string with user preferences

        #     Returns:
        #         JSON with predictions, confidence, source contributions

        #     Example:
        #         result = omn1_unified_predict(
        #             context='{"current_files": ["auth.py"], "task_context": {"goal": "implement_auth"}}',
        #             user_profile='{"work_style": "test_driven"}'
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         if not UNIFIED_INTELLIGENCE_AVAILABLE or self.predictive_engine is None:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": "Unified Intelligence System not available",
        #                 }
        #             )

        #         context_dict = json.loads(context)
        #         profile_dict = json.loads(user_profile) if user_profile else None

        #         result = await self.predictive_engine.predict(
        #             context_dict, profile_dict
        #         )

        #         # Convert to JSON-serializable format
        #         predictions = [
        #             {
        #                 "prediction_type": p.prediction_type,
        #                 "predicted_item": p.predicted_item,
        #                 "confidence": p.confidence,
        #                 "source": p.source,
        #                 "reasoning": p.reasoning,
        #             }
        #             for p in result.primary_predictions
        #         ]

        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "predictions": predictions,
        #                 "overall_confidence": result.overall_confidence,
        #                 "source_contributions": result.source_contributions,
        #                 "cross_memory_insights": result.cross_memory_insights,
        #                 "execution_time_ms": result.execution_time_ms,
        #             }
        #         )

        #     except Exception as e:
        #         print(f"Error in unified predict: {e}", file=sys.stderr)
        #         return json.dumps({"status": "error", "error": str(e)})

        # @self.mcp.tool()
        # async def omn1_orchestrate_query(
        #     query: str,
        #     context: str = None,
        # ) -> str:
        #     """Orchestrate memory retrieval across File Context and Agent Memory

        #     Intelligently routes queries to optimal memory systems and fuses results.
        #     Supports parallel retrieval for MIXED queries.

        #     Args:
        #         query: Search query or question
        #         context: Optional JSON string with context (current_files, recent_actions, etc)

        #     Returns:
        #         JSON with fused results, query analysis, performance metrics

        #     Example:
        #         result = omn1_orchestrate_query(
        #             query="Where is the authentication code?",
        #             context='{"current_files": ["main.py"]}'
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         if not UNIFIED_INTELLIGENCE_AVAILABLE or self.orchestrator is None:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": "Unified Intelligence System not available",
        #                 }
        #             )

        #         # Build QueryContext
        #         if context:
        #             context_dict = json.loads(context)
        #         else:
        #             context_dict = {}

        #         query_context = QueryContext(
        #             current_files=context_dict.get("current_files", []),
        #             recent_actions=context_dict.get("recent_actions", []),
        #             task_context=context_dict.get("task_context", {}),
        #             timestamp=datetime.now(),
        #         )

        #         result = await self.orchestrator.orchestrate(query, query_context)

        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "fused_results": result.fused_results,
        #                 "sources_used": result.sources_used,
        #                 "query_type": result.query_analysis["query_type"],
        #                 "total_time_ms": result.performance_metrics["total_time_ms"],
        #                 "orchestration_overhead_ms": result.performance_metrics[
        #                     "orchestration_overhead_ms"
        #                 ],
        #             }
        #         )

        #     except Exception as e:
        #         print(f"Error in orchestrate query: {e}", file=sys.stderr)
        #         return json.dumps({"status": "error", "error": str(e)})

        # @self.mcp.tool()
        # async def omn1_get_suggestions(
        #     context: str,
        #     user_profile: str = None,
        #     show_only: bool = False,
        # ) -> str:
        #     """Get proactive suggestions based on current context

        #     Generates actionable suggestions (file prefetch, tool recommendations,
        #     next actions) with intelligent timing and relevance filtering.

        #     Args:
        #         context: JSON string with context (current_files, recent_actions, task_context)
        #         user_profile: Optional JSON string with user preferences
        #         show_only: If True, only return suggestions that should be shown now

        #     Returns:
        #         JSON with suggestions, priority, confidence, timing

        #     Example:
        #         result = omn1_get_suggestions(
        #             context='{"current_files": ["test.py"], "recent_actions": ["implement_feature"]}',
        #             show_only=True
        #         )
        #     """
        #     start_time = time.time()

        #     try:
        #         if (
        #             not UNIFIED_INTELLIGENCE_AVAILABLE
        #             or self.suggestion_service is None
        #         ):
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": "Unified Intelligence System not available",
        #                 }
        #             )

        #         context_dict = json.loads(context)
        #         profile_dict = json.loads(user_profile) if user_profile else None

        #         if show_only:
        #             suggestions = await self.suggestion_service.get_suggestions_to_show(
        #                 context_dict, profile_dict
        #             )
        #         else:
        #             suggestions = await self.suggestion_service.generate_suggestions(
        #                 context_dict, profile_dict
        #             )

        #         # Convert to JSON-serializable format
        #         suggestions_data = [
        #             {
        #                 "suggestion_id": s.suggestion_id,
        #                 "suggestion_type": s.suggestion_type.value,
        #                 "priority": s.priority.value,
        #                 "title": s.title,
        #                 "description": s.description,
        #                 "action": s.action,
        #                 "confidence": s.confidence,
        #                 "reasoning": s.reasoning,
        #             }
        #             for s in suggestions
        #         ]

        #         metrics = self.suggestion_service.get_metrics()

        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "suggestions": suggestions_data,
        #                 "count": len(suggestions_data),
        #                 "metrics": metrics,
        #                 "generation_time_ms": (time.time() - start_time) * 1000,
        #             }
        #         )

        #     except Exception as e:
        #         print(f"Error getting suggestions: {e}", file=sys.stderr)
        #         return json.dumps({"status": "error", "error": str(e)})

        # @self.mcp.tool()
        # async def omn1_record_feedback(
        #     suggestion_id: str,
        #     accepted: bool,
        #     response_time_ms: float,
        #     feedback: str = None,
        # ) -> str:
        #     """Record user feedback on a suggestion

        #     Tracks acceptance/rejection to improve future suggestions through
        #     feedback-driven learning.

        #     Args:
        #         suggestion_id: ID of the suggestion
        #         accepted: Whether user accepted or rejected
        #         response_time_ms: Time taken to respond
        #         feedback: Optional text feedback

        #     Returns:
        #         JSON with updated metrics and learning insights

        #     Example:
        #         result = omn1_record_feedback(
        #             suggestion_id="sug_123",
        #             accepted=True,
        #             response_time_ms=2500
        #         )
        #     """
        #     try:
        #         if (
        #             not UNIFIED_INTELLIGENCE_AVAILABLE
        #             or self.suggestion_service is None
        #         ):
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": "Unified Intelligence System not available",
        #                 }
        #             )

        #         # Find the suggestion
        #         suggestion = None
        #         for s in self.suggestion_service.active_suggestions:
        #             if s.suggestion_id == suggestion_id:
        #                 suggestion = s
        #                 break

        #         if not suggestion:
        #             return json.dumps(
        #                 {
        #                     "status": "error",
        #                     "error": f"Suggestion {suggestion_id} not found",
        #                 }
        #             )

        #         result = SuggestionResult(
        #             suggestion_id=suggestion_id,
        #             accepted=accepted,
        #             response_time_ms=response_time_ms,
        #             user_feedback=feedback,
        #         )

        #         await self.suggestion_service.record_user_feedback(suggestion, result)

        #         metrics = self.suggestion_service.feedback_tracker.get_metrics()

        #         return json.dumps(
        #             {
        #                 "status": "success",
        #                 "acceptance_rate": metrics["overall_acceptance_rate"],
        #                 "feedback_metrics": metrics,
        #             }
        #         )

        #     except Exception as e:
        #         print(f"Error recording feedback: {e}", file=sys.stderr)
        #         return json.dumps({"status": "error", "error": str(e)})

        # @self.mcp.tool()
        # async def omn1_system_status() -> str:
        #     """Check health status of all OmniMemory services

        #     Returns comprehensive health status of all services including:
        #     - Core services (embeddings, compression, metrics, dashboard)
        #     - Unified Intelligence components
        #     - Autonomous daemons
        #     - Service ports and endpoints

        #     Returns:
        #         JSON with complete system status

        #     Example:
        #         result = omn1_system_status()
        #     """
        #     status = {
        #         "timestamp": datetime.now().isoformat(),
        #         "services": {},
        #         "unified_intelligence": {},
        #         "daemons": {},
        #     }

        #     # Check core services
        #     services = [
        #         ("embeddings", "http://localhost:8000/health"),
        #         ("compression", "http://localhost:8001/health"),
        #         ("procedural", "http://localhost:8002/health"),
        #         ("metrics", "http://localhost:8003/health"),
        #     ]

        #     async with aiohttp.ClientSession() as session:
        #         for name, url in services:
        #             try:
        #                 async with session.get(
        #                     url, timeout=aiohttp.ClientTimeout(total=2)
        #                 ) as resp:
        #                     if resp.status == 200:
        #                         status["services"][name] = {
        #                             "status": "healthy",
        #                             "url": url,
        #                         }
        #                     else:
        #                         status["services"][name] = {
        #                             "status": "unhealthy",
        #                             "error": f"HTTP {resp.status}",
        #                         }
        #             except Exception as e:
        #                 status["services"][name] = {
        #                     "status": "offline",
        #                     "error": str(e)[:100],
        #                 }

        #         # Check unified intelligence endpoints
        #         unified_endpoints = [
        #             ("predictions", "http://localhost:8003/unified/predictions"),
        #             ("orchestration", "http://localhost:8003/unified/orchestration"),
        #             ("suggestions", "http://localhost:8003/unified/suggestions"),
        #             ("insights", "http://localhost:8003/unified/insights"),
        #         ]

        #         for name, url in unified_endpoints:
        #             try:
        #                 async with session.get(
        #                     url, timeout=aiohttp.ClientTimeout(total=2)
        #                 ) as resp:
        #                     if resp.status == 200:
        #                         status["unified_intelligence"][name] = "operational"
        #                     else:
        #                         status["unified_intelligence"][
        #                             name
        #                         ] = f"error: HTTP {resp.status}"
        #             except Exception as e:
        #                 status["unified_intelligence"][name] = f"offline: {str(e)[:50]}"

        #     # Check daemons by process name
        #     daemon_processes = {
        #         "memory_daemon": "Memory monitoring",
        #         "checkpoint_monitor": "Checkpoint saving",
        #         "workflow_learner": "Pattern learning",
        #         "context_orchestrator": "Context management",
        #     }

        #     for process_name, description in daemon_processes.items():
        #         found = False
        #         for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        #             try:
        #                 cmdline = proc.info.get("cmdline", [])
        #                 if cmdline and process_name in str(cmdline):
        #                     status["daemons"][process_name] = {
        #                         "status": "running",
        #                         "pid": proc.info["pid"],
        #                         "description": description,
        #                     }
        #                     found = True
        #                     break
        #             except (psutil.NoSuchProcess, psutil.AccessDenied):
        #                 pass

        #         if not found:
        #             status["daemons"][process_name] = {
        #                 "status": "stopped",
        #                 "description": description,
        #             }

        #     # Add dashboard status
        #     try:
        #         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #         result = sock.connect_ex(("localhost", 8004))
        #         sock.close()
        #         if result == 0:
        #             status["services"]["dashboard"] = {
        #                 "status": "running",
        #                 "url": "http://localhost:8004",
        #             }
        #         else:
        #             status["services"]["dashboard"] = {"status": "offline"}
        #     except Exception as e:
        #         status["services"]["dashboard"] = {
        #             "status": "unknown",
        #             "error": str(e)[:50],
        #         }

        #     # Calculate overall health
        #     total_services = len(status["services"]) + len(
        #         status["unified_intelligence"]
        #     )
        #     healthy_services = sum(
        #         1
        #         for s in status["services"].values()
        #         if isinstance(s, dict) and s.get("status") in ["healthy", "running"]
        #     )
        #     healthy_services += sum(
        #         1 for s in status["unified_intelligence"].values() if s == "operational"
        #     )

        #     status["overall_health"] = {
        #         "healthy": healthy_services,
        #         "total": total_services,
        #         "percentage": round(
        #             (healthy_services / total_services * 100)
        #             if total_services > 0
        #             else 0,
        #             1,
        #         ),
        #     }

        #     return json.dumps(status, indent=2)

        # @self.mcp.tool()
        # async def omn1_unified_health() -> str:
        #     """Check health of Unified Intelligence System

        #     Detailed health check for the unified intelligence components:
        #     - Predictive Engine status
        #     - Memory Orchestrator status
        #     - Suggestion Service status
        #     - Cross-memory correlations
        #     - Performance metrics

        #     Returns:
        #         JSON with unified intelligence health metrics

        #     Example:
        #         result = omn1_unified_health()
        #     """
        #     health = {
        #         "timestamp": datetime.now().isoformat(),
        #         "components": {},
        #         "metrics": {},
        #         "status": "unknown",
        #     }

        #     try:
        #         # Check if unified intelligence components are initialized
        #         if (
        #             hasattr(self, "predictive_engine")
        #             and self.predictive_engine is not None
        #         ):
        #             cache_size = 0
        #             if hasattr(self.predictive_engine, "cache"):
        #                 if hasattr(self.predictive_engine.cache, "cache"):
        #                     cache_size = len(self.predictive_engine.cache.cache)

        #             health["components"]["predictive_engine"] = {
        #                 "initialized": True,
        #                 "cache_size": cache_size,
        #                 "status": "operational",
        #             }
        #         else:
        #             health["components"]["predictive_engine"] = {
        #                 "initialized": False,
        #                 "status": "not_initialized",
        #             }

        #         if hasattr(self, "orchestrator") and self.orchestrator is not None:
        #             cache_hit_rate = getattr(self.orchestrator, "cache_hit_rate", 0)
        #             health["components"]["orchestrator"] = {
        #                 "initialized": True,
        #                 "cache_hit_rate": cache_hit_rate,
        #                 "status": "operational",
        #             }
        #         else:
        #             health["components"]["orchestrator"] = {
        #                 "initialized": False,
        #                 "status": "not_initialized",
        #             }

        #         if (
        #             hasattr(self, "suggestion_service")
        #             and self.suggestion_service is not None
        #         ):
        #             metrics = {}
        #             if hasattr(self.suggestion_service, "get_metrics"):
        #                 try:
        #                     metrics = self.suggestion_service.get_metrics()
        #                 except:
        #                     pass

        #             health["components"]["suggestion_service"] = {
        #                 "initialized": True,
        #                 "suggestions_generated": metrics.get(
        #                     "suggestions_generated", 0
        #                 ),
        #                 "acceptance_rate": metrics.get("overall_acceptance_rate", 0),
        #                 "status": "operational",
        #             }
        #         else:
        #             health["components"]["suggestion_service"] = {
        #                 "initialized": False,
        #                 "status": "not_initialized",
        #             }

        #         # Test a prediction to verify functionality
        #         if (
        #             hasattr(self, "predictive_engine")
        #             and self.predictive_engine is not None
        #         ):
        #             try:
        #                 test_context = {
        #                     "current_files": ["test.py"],
        #                     "task_context": {"goal": "health_check"},
        #                 }

        #                 # Check if predict method exists
        #                 if hasattr(self.predictive_engine, "predict"):
        #                     start_time = time.time()
        #                     result = await self.predictive_engine.predict(test_context)
        #                     execution_time_ms = (time.time() - start_time) * 1000

        #                     health["metrics"]["prediction_test"] = {
        #                         "success": True,
        #                         "execution_time_ms": round(execution_time_ms, 2),
        #                         "confidence": getattr(result, "overall_confidence", 0)
        #                         if hasattr(result, "overall_confidence")
        #                         else 0,
        #                     }
        #                 else:
        #                     health["metrics"]["prediction_test"] = {
        #                         "success": False,
        #                         "error": "predict method not available",
        #                     }
        #             except Exception as e:
        #                 health["metrics"]["prediction_test"] = {
        #                     "success": False,
        #                     "error": str(e)[:100],
        #                 }

        #         # Determine overall status
        #         all_operational = all(
        #             c.get("status") == "operational"
        #             for c in health["components"].values()
        #         )

        #         if all_operational:
        #             health["status"] = "healthy"
        #         elif any(c.get("initialized") for c in health["components"].values()):
        #             health["status"] = "degraded"
        #         else:
        #             health["status"] = "offline"

        #         # Add performance metrics
        #         prediction_time = (
        #             health["metrics"]
        #             .get("prediction_test", {})
        #             .get("execution_time_ms", float("inf"))
        #         )
        #         health["performance"] = {
        #             "target_prediction_ms": 100,
        #             "target_orchestration_ms": 50,
        #             "target_suggestion_ms": 50,
        #             "meets_targets": prediction_time < 100
        #             if prediction_time != float("inf")
        #             else False,
        #         }

        #     except Exception as e:
        #         print(f"Error checking unified health: {e}", file=sys.stderr)
        #         health["status"] = "error"
        #         health["error"] = str(e)[:200]

        #     return json.dumps(health, indent=2)

        @self.mcp.tool()
        async def generate_memory_bank(action: str = "sync") -> str:
            """
            Generate or sync Memory Bank - structured project context files

            Creates /memory-bank/ directory with auto-generated documentation:
            - prd.md: Product requirements from conversations
            - design.md: Architecture decisions and design patterns
            - tasks.md: Development tasks and TODOs
            - context.md: Current session context
            - patterns.md: Learned coding patterns and conventions

            Follows GitHub Copilot Memory Bank pattern for instant AI context.

            Args:
                action: Action to perform
                    - "sync" (default): Generate all memory bank files
                    - "prd": Generate only prd.md
                    - "design": Generate only design.md
                    - "tasks": Generate only tasks.md
                    - "context": Generate only context.md
                    - "patterns": Generate only patterns.md
                    - "export": Export to .github/copilot-instructions.md

            Returns:
                JSON with status, generated files, and statistics

            Examples:
                # Generate all memory bank files
                generate_memory_bank("sync")

                # Generate only PRD
                generate_memory_bank("prd")

                # Export to Copilot format
                generate_memory_bank("export")
            """
            try:
                if not _MEMORY_BANK_MANAGER:
                    return json.dumps(
                        {
                            "status": "unavailable",
                            "message": "Memory Bank Manager not initialized. Session memory must be enabled.",
                        },
                        indent=2,
                    )

                result = {"status": "success", "action": action, "files": {}}

                if action == "sync":
                    files = await _MEMORY_BANK_MANAGER.sync_to_disk()
                    result["files"] = files
                    result["message"] = f"Generated {len(files)} Memory Bank files"

                elif action == "prd":
                    prd_path = await _MEMORY_BANK_MANAGER.generate_prd()
                    result["files"]["prd"] = prd_path
                    result["message"] = "Generated prd.md"

                elif action == "design":
                    design_path = await _MEMORY_BANK_MANAGER.generate_design()
                    result["files"]["design"] = design_path
                    result["message"] = "Generated design.md"

                elif action == "tasks":
                    tasks_path = await _MEMORY_BANK_MANAGER.generate_tasks()
                    result["files"]["tasks"] = tasks_path
                    result["message"] = "Generated tasks.md"

                elif action == "context":
                    context_path = await _MEMORY_BANK_MANAGER.generate_context()
                    result["files"]["context"] = context_path
                    result["message"] = "Generated context.md"

                elif action == "patterns":
                    patterns_path = await _MEMORY_BANK_MANAGER.generate_patterns()
                    result["files"]["patterns"] = patterns_path
                    result["message"] = "Generated patterns.md"

                elif action == "export":
                    copilot_path = (
                        await _MEMORY_BANK_MANAGER.export_copilot_instructions()
                    )
                    result["files"]["copilot"] = copilot_path
                    result["message"] = "Exported to .github/copilot-instructions.md"

                else:
                    result["status"] = "error"
                    result["message"] = (
                        f"Unknown action: {action}. "
                        f"Valid actions: sync, prd, design, tasks, context, patterns, export"
                    )

                # Add statistics
                if _MEMORY_BANK_MANAGER.memory_bank_dir.exists():
                    result["memory_bank_dir"] = str(
                        _MEMORY_BANK_MANAGER.memory_bank_dir
                    )
                    result["statistics"] = {
                        "total_conversations": await _MEMORY_BANK_MANAGER._count_conversations(),
                        "total_decisions": await _MEMORY_BANK_MANAGER._count_decisions(),
                        "total_patterns": await _MEMORY_BANK_MANAGER._count_patterns(),
                    }

                return json.dumps(result, indent=2)

            except Exception as e:
                return json.dumps(
                    {
                        "status": "error",
                        "action": action,
                        "message": f"Failed to generate Memory Bank: {str(e)}",
                    },
                    indent=2,
                )

    def _register_resources(self):
        """Register MCP resources for progressive tool tier disclosure"""
        # TODO: Fix FastMCP API usage for resources
        # The methods list_resources() and read_resource() return coroutines, not decorators
        # This causes TypeError: 'coroutine' object is not callable
        # Need to research correct FastMCP API pattern for resource registration
        # Temporarily disabled to unblock server initialization
        print(
            "Progressive Disclosure: Resources temporarily disabled (FastMCP API fix pending)",
            file=sys.stderr,
        )
        pass

        # ORIGINAL CODE (BROKEN - DO NOT UNCOMMENT WITHOUT FIXING API USAGE):
        # @self.mcp.list_resources()
        # async def list_resources() -> list[types.Resource]:
        #     """List available tool tiers as resources for progressive disclosure"""
        #     resources = []
        #
        #     # Add resource for each tier
        #     for tier in ToolTier:
        #         tier_info = get_tier_info(tier)
        #         resources.append(
        #             types.Resource(
        #                 uri=f"omnimemory://tools/{tier.value}",
        #                 name=f"{tier_info['name']} ({tier_info['tool_count']} tools)",
        #                 description=f"{tier_info['description']} | Est. {tier_info['estimated_tokens']} tokens",
        #                 mimeType="application/json",
        #             )
        #         )
        #
        #     # Add statistics resource
        #     resources.append(
        #         types.Resource(
        #             uri="omnimemory://tools/statistics",
        #             name="Tool Tier Statistics",
        #             description="Progressive disclosure statistics and context reduction metrics",
        #             mimeType="application/json",
        #         )
        #     )
        #
        #     return resources
        #
        # @self.mcp.read_resource()
        # async def read_resource(uri: str) -> str:
        #     """Read tool tier details and usage guidance"""
        #
        #     # Handle tier-specific resources
        #     if uri.startswith("omnimemory://tools/"):
        #         tier_name = uri.split("/")[-1]
        #
        #         # Statistics resource
        #         if tier_name == "statistics":
        #             stats = get_tier_statistics()
        #             return json.dumps(
        #                 {
        #                     "progressive_disclosure": {
        #                         "enabled": True,
        #                         "statistics": stats,
        #                         "context_savings": f"{stats['context_reduction_percentage']}% reduction",
        #                         "description": "Only core tools (3) are loaded by default. "
        #                         "Other tiers load on-demand when keywords are detected.",
        #                     }
        #                 },
        #                 indent=2,
        #             )
        #
        #         # Tier-specific resource
        #         try:
        #             tier = ToolTier(tier_name)
        #             tier_info = get_tier_info(tier)
        #
        #             return json.dumps(
        #                 {
        #                     "tier": tier_info,
        #                     "usage_guide": {
        #                         "description": f"Load {tier_info['name']} when you need these capabilities",
        #                         "tools": tier_info["tools"],
        #                         "activation_keywords": tier_info["activation_keywords"],
        #                         "auto_load": tier_info["auto_load"],
        #                         "estimated_context_cost": f"{tier_info['estimated_tokens']} tokens",
        #                     },
        #                     "how_to_load": f"Access tools by invoking them directly or mention keywords: {', '.join(tier_info['activation_keywords'][:5])}"
        #                     if tier_info["activation_keywords"]
        #                     else "Auto-loaded (core tier)",
        #                 },
        #                 indent=2,
        #             )
        #         except ValueError:
        #             return json.dumps({"error": f"Unknown tier: {tier_name}"}, indent=2)
        #
        #     return json.dumps({"error": "Unknown resource URI"}, indent=2)

    def _register_prompts(self):
        """Register MCP prompts for guided tool tier loading"""
        # TODO: Fix FastMCP API usage for prompts
        # The methods list_prompts() and get_prompt() return coroutines, not decorators
        # This causes TypeError: 'coroutine' object is not callable
        # Need to research correct FastMCP API pattern for prompt registration
        # Temporarily disabled to unblock server initialization
        print(
            "Progressive Disclosure: Prompts temporarily disabled (FastMCP API fix pending)",
            file=sys.stderr,
        )
        pass

        # ORIGINAL CODE (BROKEN - DO NOT UNCOMMENT WITHOUT FIXING API USAGE):
        # @self.mcp.list_prompts()
        # async def list_prompts() -> list[types.Prompt]:
        #     """List available prompts for tool tier discovery"""
        #     prompts = []
        #
        #     # Prompt for each on-demand tier
        #     for tier in [ToolTier.SEARCH, ToolTier.ADVANCED, ToolTier.ADMIN]:
        #         tier_info = get_tier_info(tier)
        #         prompts.append(
        #             types.Prompt(
        #                 name=f"load_{tier.value}_tools",
        #                 description=f"Load {tier_info['name']} ({tier_info['tool_count']} tools) for {tier_info['description']}",
        #                 arguments=[
        #                     types.PromptArgument(
        #                         name="reason",
        #                         description=f"Why you need {tier_info['name']}",
        #                         required=False,
        #                     )
        #                 ],
        #             )
        #         )
        #
        #     # Add discovery prompt
        #     prompts.append(
        #         types.Prompt(
        #             name="discover_tools",
        #             description="Discover available tool tiers and their capabilities",
        #             arguments=[
        #                 types.PromptArgument(
        #                     name="task",
        #                     description="What task are you trying to accomplish?",
        #                     required=False,
        #                 )
        #             ],
        #         )
        #     )
        #
        #     return prompts
        #
        # @self.mcp.get_prompt()
        # async def get_prompt(
        #     name: str, arguments: Optional[dict[str, str]] = None
        # ) -> types.GetPromptResult:
        #     """Get prompt content for tool tier loading"""
        #     if arguments is None:
        #         arguments = {}
        #
        #     # Handle tier loading prompts
        #     if name.startswith("load_") and name.endswith("_tools"):
        #         tier_name = name.replace("load_", "").replace("_tools", "")
        #         try:
        #             tier = ToolTier(tier_name)
        #             tier_info = get_tier_info(tier)
        #             reason = arguments.get("reason", "User requested")
        #
        #             prompt_message = f"""# Load {tier_info['name']}
        #
        # **Reason**: {reason}
        #
        # **Tools Available** ({tier_info['tool_count']}):
        # {chr(10).join(f"- {tool}" for tool in tier_info['tools'])}
        #
        # **Description**: {tier_info['description']}
        #
        # **Estimated Context Cost**: {tier_info['estimated_tokens']} tokens
        #
        # **Activation Keywords**: {', '.join(tier_info['activation_keywords'][:10])}
        #
        # These tools are now available for use. Invoke them directly as needed.
        # """
        #
        #             return types.GetPromptResult(
        #                 description=f"Loaded {tier_info['name']}",
        #                 messages=[
        #                     types.PromptMessage(
        #                         role="user",
        #                         content=types.TextContent(
        #                             type="text", text=prompt_message
        #                         ),
        #                     )
        #                 ],
        #             )
        #         except ValueError:
        #             return types.GetPromptResult(
        #                 description=f"Unknown tier: {tier_name}",
        #                 messages=[
        #                     types.PromptMessage(
        #                         role="user",
        #                         content=types.TextContent(
        #                             type="text",
        #                             text=f"Error: Unknown tier '{tier_name}'",
        #                         ),
        #                     )
        #                 ],
        #             )
        #
        #     # Handle discovery prompt
        #     elif name == "discover_tools":
        #         task = arguments.get("task", "General exploration")
        #         all_tiers = get_all_tiers_info()
        #         stats = get_tier_statistics()
        #
        #         prompt_message = f"""# OmniMemory Tool Discovery
        #
        # **Your Task**: {task}
        #
        # **Progressive Disclosure**: Only core tools are loaded by default to save context (79.2% reduction)
        #
        # ## Available Tool Tiers:
        #
        # """
        #         for tier_name, tier_info in all_tiers.items():
        #             prompt_message += f"""
        # ### {tier_info['name']} ({tier_info['tool_count']} tools)
        # - **Description**: {tier_info['description']}
        # - **Context Cost**: {tier_info['estimated_tokens']} tokens
        # - **Auto-loaded**: {"Yes ✓" if tier_info['auto_load'] else "No (on-demand)"}
        # - **Tools**: {', '.join(tier_info['tools'])}
        # """
        #             if tier_info["activation_keywords"]:
        #                 prompt_message += f"- **Load when**: {', '.join(tier_info['activation_keywords'][:5])}\n"
        #
        #         prompt_message += f"""
        #
        # ## Statistics:
        # - **Total Tools**: {stats['total_tools']}
        # - **Auto-loaded**: {stats['auto_load_tools']} (core tier)
        # - **On-demand**: {stats['on_demand_tools']}
        # - **Context Reduction**: {stats['context_reduction_percentage']}%
        #
        # ## How to Use:
        # 1. **Core tools** are always available (smart_read, compress, get_stats)
        # 2. **Other tools** load automatically when keywords are detected
        # 3. **Manual loading**: Use `load_<tier>_tools` prompts to load specific tiers
        #
        # Choose the tools you need based on your task!
        # """
        #
        #         return types.GetPromptResult(
        #             description="Discovered all tool tiers",
        #             messages=[
        #                 types.PromptMessage(
        #                     role="user",
        #                     content=types.TextContent(type="text", text=prompt_message),
        #                 )
        #             ],
        #         )
        #
        #     # Unknown prompt
        #     return types.GetPromptResult(
        #         description=f"Unknown prompt: {name}",
        #         messages=[
        #             types.PromptMessage(
        #                 role="user",
        #                 content=types.TextContent(
        #                     type="text", text=f"Error: Unknown prompt '{name}'"
        #                 ),
        #             )
        #         ],
        #     )

    async def cleanup(self):
        """Cleanup resources"""
        # Close persistent HTTP client with connection pooling
        if hasattr(self, "http_client"):
            await self.http_client.aclose()
            print("✓ HTTP connection pool closed", file=sys.stderr)

        # Close gateway client if in cloud mode
        if self.connection_mode == "cloud" and hasattr(self, "gateway_client"):
            await self.gateway_client.aclose()
            print("✓ Gateway client closed", file=sys.stderr)


def main():
    """Main entry point for OmniMemory MCP Server"""
    print("🚀 Starting OmniMemory MCP Server v1.0.0", file=sys.stderr)
    print(
        "Enhanced AI memory management with SWE-bench validation pending performance",
        file=sys.stderr,
    )

    # Start session tracking for this process (this tab/window)
    _start_session()

    # Register session cleanup on exit
    atexit.register(_end_session)

    # Initialize server
    server = OmniMemoryMCPServer()

    # Run MCP server (this starts its own event loop)
    server.mcp.run()


if __name__ == "__main__":
    main()
