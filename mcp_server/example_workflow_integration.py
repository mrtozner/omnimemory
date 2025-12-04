#!/usr/bin/env python3
"""
Example: Integrating Workflow Pattern Miner with MCP Server

This demonstrates how to integrate the Workflow Pattern Miner into the
main OmniMemory MCP server.
"""

import asyncio
import logging
from workflow_mcp_integration import integrate_workflow_miner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_usage():
    """
    Example of using Workflow Pattern Miner
    """
    print("=" * 60)
    print("Workflow Pattern Miner - Example Usage")
    print("=" * 60)

    # In the actual MCP server, you would do this:
    # integration = integrate_workflow_miner(mcp_server)

    # For this example, we'll use the miner directly
    from workflow_pattern_miner import WorkflowPatternMiner

    miner = WorkflowPatternMiner(
        db_path="/tmp/example_workflow.db", min_support=2, min_length=2
    )

    print("\n1. Recording Actions (simulating a debug workflow)")
    print("-" * 60)

    # Simulate a debug workflow happening 3 times
    for i in range(3):
        session_id = f"session_{i}"
        print(f"\n  Session {i+1}:")

        # Typical debug workflow
        await miner.record_action("grep", "error", session_id=session_id)
        print("    - grep 'error'")
        await asyncio.sleep(0.01)  # Small delay to separate actions

        await miner.record_action(
            "file_read", f"src/handler_{i}.py", session_id=session_id
        )
        print(f"    - read src/handler_{i}.py")
        await asyncio.sleep(0.01)

        await miner.record_action(
            "file_edit", f"src/handler_{i}.py", session_id=session_id
        )
        print(f"    - edit src/handler_{i}.py")
        await asyncio.sleep(0.01)

        await miner.record_action("command", "pytest tests/", session_id=session_id)
        print("    - run pytest")
        await asyncio.sleep(0.01)

    print("\n2. Mining Patterns")
    print("-" * 60)

    patterns = await miner.mine_patterns(min_support=2, min_length=2)

    print(f"\n  Discovered {len(patterns)} patterns:")
    for i, pattern in enumerate(patterns[:5], 1):
        print(f"\n  Pattern {i}:")
        print(f"    ID: {pattern.pattern_id}")
        print(f"    Sequence: {' → '.join([s.normalize() for s in pattern.sequence])}")
        print(f"    Frequency: {pattern.frequency}")
        print(f"    Confidence: {pattern.confidence:.2f}")

    print("\n3. Getting Workflow Suggestions")
    print("-" * 60)

    # Simulate current actions
    from workflow_pattern_miner import ActionStep

    recent_actions = [
        ActionStep(action_type="grep", target="error", parameters={}),
        ActionStep(action_type="file_read", target="src/new_file.py", parameters={}),
    ]

    suggestions = await miner.detect_current_workflow(recent_actions, top_k=3)

    print(f"\n  Based on recent actions:")
    print(f"    - grep 'error'")
    print(f"    - read src/new_file.py")
    print(f"\n  Suggestions ({len(suggestions)} found):")

    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n  Suggestion {i}:")
        print(f"    Confidence: {suggestion.confidence:.2f}")
        print(f"    Reason: {suggestion.reason}")
        print("    Next steps:")
        for step in suggestion.next_steps:
            print(f"      - {step.action_type}({step.target})")

    print("\n4. Creating Automation")
    print("-" * 60)

    if patterns:
        pattern = patterns[0]
        automation = await miner.create_automation(
            pattern.pattern_id, name="Debug Workflow"
        )

        print(f"\n  Created automation:")
        print(f"    ID: {automation['automation_id']}")
        print(f"    Name: {automation['name']}")
        print(f"    Steps: {len(automation['steps'])}")
        print(f"    Success Rate: {automation['success_rate']:.1%}")
        print(f"    Requires Confirmation: {automation['requires_confirmation']}")

        print("\n5. Executing Automation (Dry Run)")
        print("-" * 60)

        result = await miner.execute_automation(
            automation["automation_id"], dry_run=True
        )

        print(f"\n  Dry run result:")
        print(f"    Status: {result['status']}")
        print(f"    Message: {result['message']}")
        print(f"    Steps: {len(result['steps'])}")

    print("\n6. Statistics")
    print("-" * 60)

    stats = miner.get_pattern_stats()

    print(f"\n  Mining Statistics:")
    print(f"    Total Patterns: {stats['total_patterns']}")
    print(f"    Patterns Discovered: {stats['mining_stats']['patterns_discovered']}")
    print(f"    Suggestions Made: {stats['mining_stats']['suggestions_made']}")
    print(f"    Automations Executed: {stats['mining_stats']['automations_executed']}")

    if stats["patterns_by_frequency"]:
        print(f"\n  Top Patterns by Frequency:")
        for p in stats["patterns_by_frequency"][:3]:
            print(f"    - {p['pattern_id']}: {p['frequency']} occurrences")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


async def integration_example():
    """
    Example of how to integrate with the actual MCP server
    """
    print("\n" + "=" * 60)
    print("MCP Server Integration Example")
    print("=" * 60)

    print(
        """
To integrate Workflow Pattern Miner into omnimemory_mcp.py:

1. Import the integration module:
   ```python
   from workflow_mcp_integration import integrate_workflow_miner
   ```

2. In the OmniMemoryServer.__init__ method, add:
   ```python
   # Initialize Workflow Pattern Miner
   self.workflow_integration = integrate_workflow_miner(
       self.mcp,
       db_path="~/.omnimemory/workflow_patterns.db",
       min_support=3,
       min_length=2
   )
   ```

3. Track actions in existing MCP tools:
   ```python
   # In omnimemory_smart_read or other tools:
   await self.workflow_integration.track_action(
       action_type="file_read",
       target=file_path,
       session_id=self.session_manager.current_session.session_id
   )
   ```

4. Optional: Auto-suggest after actions:
   ```python
   # After significant actions:
   suggestion = await self.workflow_integration.auto_suggest_on_action(
       action_type="file_read",
       target=file_path
   )
   if suggestion:
       print(f"💡 Workflow suggestion: {suggestion}")
   ```

5. The following MCP tools will be automatically registered:
   - omnimemory_discover_patterns
   - omnimemory_suggest_workflow
   - omnimemory_create_automation
   - omnimemory_execute_automation
   - omnimemory_list_patterns
   - omnimemory_get_pattern_details
   - omnimemory_workflow_stats

6. Users can then call these tools:
   ```python
   # Discover patterns
   result = await omnimemory_discover_patterns(min_support=3)

   # Get suggestions
   result = await omnimemory_suggest_workflow(context="debugging")

   # Create automation
   result = await omnimemory_create_automation(
       pattern_id="abc123",
       name="My Workflow"
   )
   ```
"""
    )

    print("=" * 60)


if __name__ == "__main__":
    print("\nWorkflow Pattern Miner - Example Demonstrations\n")

    # Run the main example
    asyncio.run(example_usage())

    # Show integration instructions
    asyncio.run(integration_example())

    print("\n✅ All examples completed!\n")
    print("To integrate with the MCP server, follow the integration example above.\n")
