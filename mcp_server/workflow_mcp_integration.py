"""
MCP Integration for Workflow Pattern Miner

Registers workflow pattern mining tools with the MCP server.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from workflow_pattern_miner import WorkflowPatternMiner, ActionStep

logger = logging.getLogger(__name__)


class WorkflowMCPIntegration:
    """
    Integrates Workflow Pattern Miner with MCP server.
    Provides tools for pattern discovery, workflow suggestions, and automation.
    """

    def __init__(
        self,
        db_path: str = "~/.omnimemory/workflow_patterns.db",
        min_support: int = 3,
        min_length: int = 2,
    ):
        """
        Initialize workflow MCP integration

        Args:
            db_path: Database path for pattern storage
            min_support: Minimum pattern frequency
            min_length: Minimum pattern length
        """
        self.miner = WorkflowPatternMiner(
            db_path=db_path, min_support=min_support, min_length=min_length
        )
        logger.info("Workflow MCP Integration initialized")

    def register_tools(self, mcp_server):
        """
        Register workflow tools with MCP server

        Args:
            mcp_server: FastMCP server instance
        """

        @mcp_server.tool()
        async def omnimemory_discover_patterns(
            min_support: int = 3,
            min_length: int = 2,
            lookback_hours: int = 168,
        ) -> Dict[str, Any]:
            """
            Discover recurring workflow patterns from session history.

            Analyzes recent action history to find frequently occurring sequences
            using sequential pattern mining (PrefixSpan algorithm).

            Args:
                min_support: Minimum times a pattern must occur (default: 3)
                min_length: Minimum number of actions in a pattern (default: 2)
                lookback_hours: How far back to analyze in hours (default: 168 = 1 week)

            Returns:
                Dictionary with discovered patterns and statistics

            Example patterns that might be discovered:
            - "Debug Cycle": grep error → read file → edit file → run test
            - "Feature Branch": git checkout → multiple edits → git add → git commit
            - "API Integration": read docs → create types → implement handler → write tests
            """
            try:
                patterns = await self.miner.mine_patterns(
                    min_support=min_support, min_length=min_length
                )

                return {
                    "status": "success",
                    "patterns_discovered": len(patterns),
                    "patterns": [
                        {
                            "pattern_id": p.pattern_id,
                            "sequence": [
                                f"{step.action_type}({step.target})"
                                for step in p.sequence
                            ],
                            "frequency": p.frequency,
                            "success_rate": p.success_rate,
                            "avg_duration": p.avg_duration,
                            "confidence": p.confidence,
                        }
                        for p in patterns[:20]  # Limit to top 20
                    ],
                    "stats": self.miner.get_pattern_stats(),
                }

            except Exception as e:
                logger.error(f"Error discovering patterns: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_suggest_workflow(
            context: str = "", top_k: int = 3
        ) -> Dict[str, Any]:
            """
            Suggest next workflow steps based on current context and history.

            Analyzes recent actions and matches them against known patterns to provide
            intelligent suggestions for what to do next.

            Args:
                context: Optional context description of what you're working on
                top_k: Number of suggestions to return (default: 3)

            Returns:
                Dictionary with workflow suggestions

            Example:
                If you just ran "grep error" and "read file.py", this might suggest:
                - "Edit file.py to fix the error (85% confidence)"
                - "Run tests to verify fix (72% confidence)"
            """
            try:
                suggestions = await self.miner.suggest_next_steps(context, top_k=top_k)

                return {
                    "status": "success",
                    "suggestions": [
                        {
                            "pattern_id": s.pattern_id,
                            "next_steps": [
                                f"{step.action_type}({step.target})"
                                for step in s.next_steps
                            ],
                            "confidence": s.confidence,
                            "reason": s.reason,
                            "estimated_duration": s.estimated_duration,
                            "success_probability": s.success_probability,
                        }
                        for s in suggestions
                    ],
                }

            except Exception as e:
                logger.error(f"Error suggesting workflow: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_create_automation(
            pattern_id: str, name: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Convert a workflow pattern into an executable automation.

            Takes a discovered pattern and creates an automation that can be executed
            to repeat the workflow automatically.

            Args:
                pattern_id: Pattern identifier from omnimemory_discover_patterns
                name: Optional name for the automation

            Returns:
                Dictionary with automation configuration

            Safety:
                All automations require user confirmation before execution.
            """
            try:
                automation = await self.miner.create_automation(pattern_id, name=name)

                return {"status": "success", "automation": automation}

            except ValueError as e:
                return {"status": "error", "error": f"Pattern not found: {pattern_id}"}
            except Exception as e:
                logger.error(f"Error creating automation: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_execute_automation(
            automation_id: str, dry_run: bool = True
        ) -> Dict[str, Any]:
            """
            Execute a workflow automation.

            Runs a previously created automation to repeat a workflow pattern.

            Args:
                automation_id: Automation identifier from omnimemory_create_automation
                dry_run: If True, only simulate execution without making changes (default: True)

            Returns:
                Dictionary with execution results

            Safety:
                - Dry run mode is enabled by default
                - Set dry_run=False only when you're sure about the automation
                - Review the automation steps before executing
            """
            try:
                result = await self.miner.execute_automation(
                    automation_id, dry_run=dry_run
                )

                return {"status": "success", "result": result}

            except ValueError as e:
                return {
                    "status": "error",
                    "error": f"Automation not found: {automation_id}",
                }
            except Exception as e:
                logger.error(f"Error executing automation: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_list_patterns(
            min_confidence: float = 0.0, limit: int = 20
        ) -> Dict[str, Any]:
            """
            List discovered workflow patterns.

            Retrieves all patterns matching the specified criteria, sorted by confidence.

            Args:
                min_confidence: Minimum confidence threshold (0.0-1.0, default: 0.0)
                limit: Maximum number of patterns to return (default: 20)

            Returns:
                Dictionary with list of patterns
            """
            try:
                patterns = self.miner.list_patterns(
                    min_confidence=min_confidence, limit=limit
                )

                return {
                    "status": "success",
                    "count": len(patterns),
                    "patterns": [
                        {
                            "pattern_id": p.pattern_id,
                            "sequence": [
                                {
                                    "action": step.action_type,
                                    "target": step.target,
                                }
                                for step in p.sequence
                            ],
                            "frequency": p.frequency,
                            "success_rate": p.success_rate,
                            "confidence": p.confidence,
                            "avg_duration": p.avg_duration,
                            "triggers": p.triggers,
                            "outcomes": p.outcomes,
                        }
                        for p in patterns
                    ],
                }

            except Exception as e:
                logger.error(f"Error listing patterns: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_get_pattern_details(
            pattern_id: str,
        ) -> Dict[str, Any]:
            """
            Get detailed information about a specific workflow pattern.

            Args:
                pattern_id: Pattern identifier

            Returns:
                Dictionary with detailed pattern information including variations
            """
            try:
                pattern = self.miner.get_pattern(pattern_id)

                if not pattern:
                    return {
                        "status": "error",
                        "error": f"Pattern not found: {pattern_id}",
                    }

                return {
                    "status": "success",
                    "pattern": {
                        "pattern_id": pattern.pattern_id,
                        "sequence": [step.to_dict() for step in pattern.sequence],
                        "frequency": pattern.frequency,
                        "success_rate": pattern.success_rate,
                        "avg_duration": pattern.avg_duration,
                        "confidence": pattern.confidence,
                        "variations": [
                            [step.to_dict() for step in var]
                            for var in pattern.variations
                        ],
                        "triggers": pattern.triggers,
                        "outcomes": pattern.outcomes,
                        "last_seen": (
                            pattern.last_seen.isoformat() if pattern.last_seen else None
                        ),
                    },
                }

            except Exception as e:
                logger.error(f"Error getting pattern details: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        @mcp_server.tool()
        async def omnimemory_workflow_stats() -> Dict[str, Any]:
            """
            Get statistics about workflow pattern mining.

            Returns comprehensive statistics about discovered patterns,
            mining performance, and usage metrics.

            Returns:
                Dictionary with workflow mining statistics
            """
            try:
                stats = self.miner.get_pattern_stats()
                return {"status": "success", "stats": stats}

            except Exception as e:
                logger.error(f"Error getting workflow stats: {e}", exc_info=True)
                return {"status": "error", "error": str(e)}

        logger.info("Registered 7 workflow pattern mining tools")

    async def track_action(
        self,
        action_type: str,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        success: bool = True,
    ):
        """
        Track an action for pattern mining.

        This should be called whenever a significant action occurs (file read, edit,
        command execution, search, etc.) to build the action history for pattern mining.

        Args:
            action_type: Type of action (file_read, file_edit, command, search, grep, write)
            target: Target of the action (file path, command, query)
            parameters: Additional parameters
            session_id: Session identifier
            success: Whether the action succeeded
        """
        await self.miner.record_action(
            action_type=action_type,
            target=target,
            parameters=parameters,
            session_id=session_id,
            success=success,
        )

    async def auto_suggest_on_action(
        self, action_type: str, target: str
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically check for workflow suggestions after an action.

        This can be called after significant actions to provide real-time suggestions.

        Args:
            action_type: Type of action just performed
            target: Target of the action

        Returns:
            Suggestions dictionary if any, None otherwise
        """
        # Get recent action history (last 5)
        recent_actions = (
            self.miner.action_history[-5:] if self.miner.action_history else []
        )

        if len(recent_actions) < 2:
            return None

        suggestions = await self.miner.detect_current_workflow(recent_actions, top_k=3)

        if suggestions and suggestions[0].confidence > 0.5:
            return {
                "type": "workflow_suggestion",
                "trigger_action": f"{action_type}({target})",
                "suggestions": [
                    {
                        "next_steps": [
                            f"{step.action_type}({step.target})"
                            for step in s.next_steps
                        ],
                        "confidence": s.confidence,
                        "reason": s.reason,
                    }
                    for s in suggestions
                ],
            }

        return None


# Convenience function for integration
def integrate_workflow_miner(mcp_server, **kwargs):
    """
    Convenience function to integrate workflow pattern miner with MCP server.

    Args:
        mcp_server: FastMCP server instance
        **kwargs: Additional arguments for WorkflowMCPIntegration

    Returns:
        WorkflowMCPIntegration instance
    """
    integration = WorkflowMCPIntegration(**kwargs)
    integration.register_tools(mcp_server)
    return integration
