"""
Tool Tier Configuration for Progressive Disclosure

Implements tiered tool exposure to reduce context consumption from 36,220 tokens to ~7,000 tokens.
Tools are organized into 4 tiers based on usage frequency and functional grouping.

Phase 5B: Progressive Disclosure for MCP Tools
Target: 60-80% context reduction (36,220 → ~7,000 tokens)
"""

from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum


class ToolTier(Enum):
    """Tool tier classification for progressive disclosure"""

    CORE = "core"  # Always visible (~1,500 tokens)
    SEARCH = "search"  # Load on-demand (~2,500 tokens)
    ADVANCED = "advanced"  # Load on-demand (~2,000 tokens)
    ADMIN = "admin"  # Load on-demand (~1,500 tokens)


@dataclass
class TierMetadata:
    """Metadata for a tool tier"""

    name: str
    description: str
    estimated_tokens: int
    activation_keywords: List[str]
    tools: List[str]
    auto_load: bool = False


# Tool tier configuration
TOOL_TIERS: Dict[ToolTier, TierMetadata] = {
    ToolTier.CORE: TierMetadata(
        name="Core Tools",
        description="Essential file operations: smart_read, compress, get_stats",
        estimated_tokens=1500,
        activation_keywords=[],  # Always loaded
        tools=[
            "omnimemory_smart_read",
            "omnimemory_compress",
            "omnimemory_get_stats",
        ],
        auto_load=True,
    ),
    ToolTier.SEARCH: TierMetadata(
        name="Search & Analysis",
        description="Advanced search: semantic, hybrid, graph, vector search",
        estimated_tokens=2500,
        activation_keywords=[
            "search",
            "find",
            "query",
            "retrieve",
            "lookup",
            "semantic",
            "vector",
            "graph",
            "hybrid",
        ],
        tools=[
            "omnimemory_search",
            "omnimemory_semantic_search",
            "omnimemory_hybrid_search",
            "omnimemory_graph_search",
            "omnimemory_retrieve",
        ],
        auto_load=False,
    ),
    ToolTier.ADVANCED: TierMetadata(
        name="Advanced Operations",
        description="Workflow management, session resumption, context optimization, storage",
        estimated_tokens=2000,
        activation_keywords=[
            "workflow",
            "resume",
            "optimize",
            "store",
            "save",
            "session",
            "context",
            "checkpoint",
            "learn",
        ],
        tools=[
            "omnimemory_workflow_context",
            "omnimemory_resume_workflow",
            "omnimemory_optimize_context",
            "omnimemory_store",
            "omnimemory_learn_workflow",
        ],
        auto_load=False,
    ),
    ToolTier.ADMIN: TierMetadata(
        name="Admin & Development",
        description="Code execution, predictions, cache operations, development tools",
        estimated_tokens=1500,
        activation_keywords=[
            "execute",
            "run",
            "code",
            "predict",
            "cache",
            "benchmark",
            "test",
            "evaluate",
            "validate",
            "debug",
        ],
        tools=[
            "omnimemory_execute_python",
            "omnimemory_predict_next",
            "omnimemory_cache_lookup",
            "omnimemory_cache_store",
        ],
        auto_load=False,
    ),
}


# Reverse mapping: tool name -> tier
TOOL_TO_TIER: Dict[str, ToolTier] = {}
for tier, metadata in TOOL_TIERS.items():
    for tool in metadata.tools:
        TOOL_TO_TIER[tool] = tier


def get_tier_for_tool(tool_name: str) -> ToolTier:
    """Get the tier for a specific tool"""
    return TOOL_TO_TIER.get(tool_name, ToolTier.CORE)


def get_tools_for_tier(tier: ToolTier) -> List[str]:
    """Get all tools in a specific tier"""
    return TOOL_TIERS[tier].tools


def get_auto_load_tools() -> List[str]:
    """Get all tools that should auto-load (core tier)"""
    auto_load_tools = []
    for tier, metadata in TOOL_TIERS.items():
        if metadata.auto_load:
            auto_load_tools.extend(metadata.tools)
    return auto_load_tools


def detect_tier_from_keywords(text: str) -> Set[ToolTier]:
    """
    Detect which tiers should be loaded based on keywords in text

    Args:
        text: User input or query text

    Returns:
        Set of tiers that should be loaded
    """
    text_lower = text.lower()
    matched_tiers = set()

    # Always include core tier
    matched_tiers.add(ToolTier.CORE)

    # Check other tiers
    for tier, metadata in TOOL_TIERS.items():
        if tier == ToolTier.CORE:
            continue

        for keyword in metadata.activation_keywords:
            if keyword in text_lower:
                matched_tiers.add(tier)
                break

    return matched_tiers


def get_estimated_token_cost(tiers: Set[ToolTier]) -> int:
    """
    Calculate estimated token cost for a set of tiers

    Args:
        tiers: Set of tool tiers

    Returns:
        Estimated total token cost
    """
    total_tokens = 0
    for tier in tiers:
        total_tokens += TOOL_TIERS[tier].estimated_tokens
    return total_tokens


def get_tier_info(tier: ToolTier) -> Dict[str, any]:
    """
    Get information about a specific tier

    Args:
        tier: The tool tier

    Returns:
        Dictionary with tier information
    """
    metadata = TOOL_TIERS[tier]
    return {
        "tier": tier.value,
        "name": metadata.name,
        "description": metadata.description,
        "estimated_tokens": metadata.estimated_tokens,
        "tools": metadata.tools,
        "tool_count": len(metadata.tools),
        "activation_keywords": metadata.activation_keywords,
        "auto_load": metadata.auto_load,
    }


def get_all_tiers_info() -> Dict[str, Dict[str, any]]:
    """
    Get information about all tiers

    Returns:
        Dictionary mapping tier names to tier information
    """
    return {tier.value: get_tier_info(tier) for tier in ToolTier}


# Statistics
def get_tier_statistics() -> Dict[str, any]:
    """
    Get statistics about tool tier configuration

    Returns:
        Dictionary with tier statistics
    """
    total_tools = sum(len(metadata.tools) for metadata in TOOL_TIERS.values())
    auto_load_count = len(get_auto_load_tools())
    total_tokens = sum(metadata.estimated_tokens for metadata in TOOL_TIERS.values())
    core_tokens = TOOL_TIERS[ToolTier.CORE].estimated_tokens

    return {
        "total_tools": total_tools,
        "total_tiers": len(TOOL_TIERS),
        "auto_load_tools": auto_load_count,
        "on_demand_tools": total_tools - auto_load_count,
        "total_tokens_all_tiers": total_tokens,
        "core_tier_tokens": core_tokens,
        "context_reduction_percentage": round(
            (1 - core_tokens / total_tokens) * 100, 1
        ),
        "average_tokens_per_tier": round(total_tokens / len(TOOL_TIERS), 0),
        "tiers": {
            tier.value: {
                "tool_count": len(metadata.tools),
                "tokens": metadata.estimated_tokens,
            }
            for tier, metadata in TOOL_TIERS.items()
        },
    }
