"""
Tool Configuration and Detection for Multi-Tool OmniMemory Support

Supports: Claude Code, Cursor, ChatGPT, Codex, Continue, Aider
"""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SupportedTool(Enum):
    """Supported AI development tools"""
    CLAUDE_CODE = "claude-code"
    CURSOR = "cursor"
    CHATGPT = "chatgpt"
    CODEX = "codex"
    CONTINUE = "continue"
    AIDER = "aider"
    UNKNOWN = "unknown"


@dataclass
class ToolConfig:
    """Configuration for a specific AI tool"""
    tool_id: str
    tool_name: str
    tool_version: str
    mcp_transport: str = "stdio"  # Default transport
    supports_hooks: bool = True
    supports_code_execution: bool = True
    
    
# Tool-specific configurations
TOOL_CONFIGS = {
    SupportedTool.CLAUDE_CODE: ToolConfig(
        tool_id="claude-code",
        tool_name="Claude Code",
        tool_version=os.getenv("CLAUDE_VERSION", "1.0.0"),
        mcp_transport="stdio",
        supports_hooks=True,
        supports_code_execution=True,
    ),
    SupportedTool.CURSOR: ToolConfig(
        tool_id="cursor",
        tool_name="Cursor",
        tool_version=os.getenv("CURSOR_VERSION", "0.42.0"),
        mcp_transport="stdio",
        supports_hooks=False,  # Cursor uses MCP tools only
        supports_code_execution=True,
    ),
    SupportedTool.CHATGPT: ToolConfig(
        tool_id="chatgpt",
        tool_name="ChatGPT",
        tool_version=os.getenv("CHATGPT_VERSION", "1.0.0"),
        mcp_transport="stdio",
        supports_hooks=False,
        supports_code_execution=True,
    ),
    SupportedTool.CODEX: ToolConfig(
        tool_id="codex",
        tool_name="Codex",
        tool_version=os.getenv("CODEX_VERSION", "1.0.0"),
        mcp_transport="stdio",
        supports_hooks=False,
        supports_code_execution=True,
    ),
    SupportedTool.CONTINUE: ToolConfig(
        tool_id="continue",
        tool_name="Continue",
        tool_version=os.getenv("CONTINUE_VERSION", "1.0.0"),
        mcp_transport="stdio",
        supports_hooks=False,
        supports_code_execution=True,
    ),
    SupportedTool.AIDER: ToolConfig(
        tool_id="aider",
        tool_name="Aider",
        tool_version=os.getenv("AIDER_VERSION", "1.0.0"),
        mcp_transport="stdio",
        supports_hooks=False,
        supports_code_execution=True,
    ),
}


def detect_tool_from_env() -> SupportedTool:
    """
    Detect which AI tool is running based on environment variables
    
    Priority order:
    1. OMNIMEMORY_TOOL_ID (explicit override)
    2. Tool-specific environment variables
    3. Process name detection
    4. Default to UNKNOWN
    """
    # Explicit override
    tool_id = os.getenv("OMNIMEMORY_TOOL_ID")
    if tool_id:
        for tool in SupportedTool:
            if tool.value == tool_id.lower():
                return tool
    
    # Claude Code detection
    if os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        return SupportedTool.CLAUDE_CODE
    
    # Cursor detection
    if os.getenv("CURSOR_USER") or os.getenv("CURSOR_HOME"):
        return SupportedTool.CURSOR
    
    # ChatGPT detection
    if os.getenv("OPENAI_API_KEY"):
        # Could be ChatGPT, Codex, or Continue
        # Need more specific detection
        if os.getenv("CHATGPT_DESKTOP"):
            return SupportedTool.CHATGPT
        elif os.getenv("CODEX_MODE"):
            return SupportedTool.CODEX
        elif os.getenv("CONTINUE_GLOBAL_DIR"):
            return SupportedTool.CONTINUE
        # Default to ChatGPT if OpenAI key is present
        return SupportedTool.CHATGPT
    
    # Aider detection
    if os.getenv("AIDER_HOME"):
        return SupportedTool.AIDER
    
    return SupportedTool.UNKNOWN


def get_tool_config(tool: Optional[SupportedTool] = None) -> ToolConfig:
    """
    Get configuration for a specific tool
    
    Args:
        tool: The tool to get config for. If None, auto-detect.
        
    Returns:
        ToolConfig for the specified or detected tool
    """
    if tool is None:
        tool = detect_tool_from_env()
    
    if tool == SupportedTool.UNKNOWN:
        # Return default config
        return ToolConfig(
            tool_id=os.getenv("OMNIMEMORY_TOOL_ID", "unknown"),
            tool_name="Unknown Tool",
            tool_version=os.getenv("OMNIMEMORY_TOOL_VERSION", "1.0.0"),
            mcp_transport="stdio",
            supports_hooks=False,
            supports_code_execution=True,
        )
    
    return TOOL_CONFIGS[tool]


def create_mcp_config_entry(tool: SupportedTool, omnimemory_path: str) -> dict:
    """
    Create MCP server configuration entry for a specific tool
    
    Args:
        tool: The tool to create config for
        omnimemory_path: Path to omnimemory mcp_server directory
        
    Returns:
        MCP server configuration dictionary
    """
    config = get_tool_config(tool)
    
    return {
        "omnimemory": {
            "command": "python",
            "args": ["-m", "omnimemory_mcp"],
            "cwd": omnimemory_path,
            "env": {
                "OMNIMEMORY_TOOL_ID": config.tool_id,
                "OMNIMEMORY_TOOL_VERSION": config.tool_version,
            }
        }
    }
