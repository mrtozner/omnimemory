"""
Integration code for Sleep Consolidation Engine with MCP Server

This file shows how to integrate the SleepConsolidationEngine with the
OmniMemory MCP Server.

Add the following code to omnimemory_mcp.py:
"""

# ============================================================================
# 1. Add import at the top of omnimemory_mcp.py
# ============================================================================
"""
from sleep_consolidation import SleepConsolidationEngine
"""

# ============================================================================
# 2. Add to OmniMemoryMCPServer.__init__ pre-initialization section (around line 1337)
# ============================================================================
"""
        self.sleep_consolidation = None
"""

# ============================================================================
# 3. Add to _initialize_components method (around line 1430)
# ============================================================================
"""
            # Initialize Sleep Consolidation Engine
            try:
                db_path = os.path.expanduser("~/.omnimemory/sessions.db")
                self.sleep_consolidation = SleepConsolidationEngine(
                    db_path=db_path,
                    redis_url=integration_config.redis_url if integration_config else "redis://localhost:6379",
                    qdrant_url="http://localhost:6333",
                    embeddings_url="http://localhost:8000",
                    idle_threshold_minutes=30,
                    nightly_schedule_hour=2,
                    enable_background_worker=True,
                )
                print("✓ Sleep Consolidation Engine initialized", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Sleep Consolidation initialization failed: {e}", file=sys.stderr)
                self.sleep_consolidation = None
"""

# ============================================================================
# 4. Add to _register_tools method (add these MCP tools)
# ============================================================================

# Tool 1: Trigger consolidation manually
"""
        @self.mcp.tool()
        async def trigger_memory_consolidation(
            aggressive: bool = False
        ) -> str:
            '''Manually trigger memory consolidation cycle

            Runs the sleep-inspired memory consolidation process that:
            - Replays recent memories to identify patterns
            - Strengthens frequently accessed memories
            - Prunes low-value memories
            - Discovers cross-session insights

            Args:
                aggressive: Use aggressive pruning (default: False)

            Returns:
                JSON with consolidation metrics

            Example:
                trigger_memory_consolidation(aggressive=False)
            '''
            try:
                if not self.sleep_consolidation:
                    return json.dumps({
                        "error": "Sleep consolidation engine not initialized",
                        "status": "unavailable"
                    })

                # Mark activity to prevent auto-consolidation during manual run
                self.sleep_consolidation.mark_activity()

                # Run consolidation
                metrics = await self.sleep_consolidation.trigger_consolidation(
                    aggressive=aggressive
                )

                return json.dumps({
                    "status": "success",
                    "metrics": metrics
                }, indent=2)

            except Exception as e:
                logger.error(f"Consolidation trigger failed: {e}", exc_info=True)
                return json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
"""

# Tool 2: Get consolidation status
"""
        @self.mcp.tool()
        async def get_consolidation_status() -> str:
            '''Get current memory consolidation status

            Returns information about:
            - Whether consolidation is currently running
            - Current phase (idle, replay, strengthen, prune, synthesize)
            - Last activity time
            - Idle status

            Returns:
                JSON with current status

            Example:
                get_consolidation_status()
            '''
            try:
                if not self.sleep_consolidation:
                    return json.dumps({
                        "error": "Sleep consolidation engine not initialized",
                        "status": "unavailable"
                    })

                status = self.sleep_consolidation.get_consolidation_status()

                return json.dumps({
                    "status": "success",
                    "consolidation": status
                }, indent=2)

            except Exception as e:
                logger.error(f"Failed to get consolidation status: {e}", exc_info=True)
                return json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
"""

# Tool 3: Get consolidation statistics
"""
        @self.mcp.tool()
        async def get_consolidation_stats() -> str:
            '''Get memory consolidation statistics

            Returns comprehensive statistics including:
            - Total consolidation cycles run
            - Recent cycle details (last 10)
            - Total cross-session insights discovered
            - Current consolidation status

            Returns:
                JSON with consolidation statistics

            Example:
                get_consolidation_stats()
            '''
            try:
                if not self.sleep_consolidation:
                    return json.dumps({
                        "error": "Sleep consolidation engine not initialized",
                        "status": "unavailable"
                    })

                stats = self.sleep_consolidation.get_consolidation_stats()

                return json.dumps({
                    "status": "success",
                    "statistics": stats
                }, indent=2)

            except Exception as e:
                logger.error(f"Failed to get consolidation stats: {e}", exc_info=True)
                return json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
"""

# Tool 4: Get consolidated insights
"""
        @self.mcp.tool()
        async def get_consolidated_insights(
            limit: int = 10,
            insight_type: str | None = None
        ) -> str:
            '''Get cross-session insights discovered during consolidation

            Returns insights like:
            - File access patterns (files frequently accessed together)
            - Common decision patterns
            - Antipatterns (inefficient workflows)

            Args:
                limit: Maximum number of insights to return (default: 10)
                insight_type: Filter by type: pattern, decision, antipattern, workflow (optional)

            Returns:
                JSON with list of insights

            Example:
                get_consolidated_insights(limit=5, insight_type="pattern")
            '''
            try:
                if not self.sleep_consolidation:
                    return json.dumps({
                        "error": "Sleep consolidation engine not initialized",
                        "status": "unavailable"
                    })

                import sqlite3

                # Query insights from database
                conn = sqlite3.connect(self.sleep_consolidation.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if insight_type:
                    cursor.execute(
                        '''
                        SELECT * FROM consolidated_insights
                        WHERE insight_type = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        ''',
                        (insight_type, limit)
                    )
                else:
                    cursor.execute(
                        '''
                        SELECT * FROM consolidated_insights
                        ORDER BY timestamp DESC
                        LIMIT ?
                        ''',
                        (limit,)
                    )

                insights = []
                for row in cursor.fetchall():
                    insights.append({
                        "insight_id": row["insight_id"],
                        "type": row["insight_type"],
                        "title": row["title"],
                        "description": row["description"],
                        "supporting_sessions": json.loads(row["supporting_sessions"]) if row["supporting_sessions"] else [],
                        "confidence": row["confidence"],
                        "timestamp": row["timestamp"]
                    })

                conn.close()

                return json.dumps({
                    "status": "success",
                    "count": len(insights),
                    "insights": insights
                }, indent=2)

            except Exception as e:
                logger.error(f"Failed to get consolidated insights: {e}", exc_info=True)
                return json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
"""

# ============================================================================
# 5. Add to async startup hook (create if doesn't exist)
# ============================================================================
"""
    async def startup(self):
        '''Async startup hook - starts background workers'''
        try:
            # Start sleep consolidation worker
            if self.sleep_consolidation:
                await self.sleep_consolidation.start()
                print("✓ Sleep consolidation background worker started", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Failed to start background workers: {e}", file=sys.stderr)
"""

# ============================================================================
# 6. Add to cleanup/shutdown hook
# ============================================================================
"""
    async def shutdown(self):
        '''Shutdown hook - stops background workers'''
        try:
            # Stop sleep consolidation worker
            if self.sleep_consolidation:
                await self.sleep_consolidation.stop()
                print("✓ Sleep consolidation background worker stopped", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Failed to stop background workers: {e}", file=sys.stderr)
"""

# ============================================================================
# 7. Hook into session activity tracking
# ============================================================================
"""
# Add to any method that handles user activity (e.g., read, search, save_memory)

        # Mark activity for consolidation engine
        if self.sleep_consolidation:
            self.sleep_consolidation.mark_activity()
"""

# ============================================================================
# Example Usage (from Claude Code or any MCP client)
# ============================================================================
"""
# Get consolidation status
result = await get_consolidation_status()

# Manually trigger consolidation
result = await trigger_memory_consolidation(aggressive=False)

# Get statistics
result = await get_consolidation_stats()

# Get insights
result = await get_consolidated_insights(limit=10, insight_type="pattern")
"""

if __name__ == "__main__":
    print("=" * 70)
    print("Sleep Consolidation Engine - MCP Integration Guide")
    print("=" * 70)
    print("\nThis file contains the integration code snippets.")
    print("Follow the numbered sections above to integrate with omnimemory_mcp.py")
    print("\nNew MCP Tools:")
    print("  1. trigger_memory_consolidation(aggressive)")
    print("  2. get_consolidation_status()")
    print("  3. get_consolidation_stats()")
    print("  4. get_consolidated_insights(limit, insight_type)")
    print("\n" + "=" * 70)
