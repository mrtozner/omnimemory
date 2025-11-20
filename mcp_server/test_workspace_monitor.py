#!/usr/bin/env python3
"""
Quick test for WorkspaceMonitor functionality
"""

import asyncio
from workspace_monitor import WorkspaceMonitor


async def main():
    """Test workspace monitor"""
    print("Testing WorkspaceMonitor...")

    # Track switch events
    switch_events = []

    async def on_switch(switch_data):
        """Record switch events"""
        switch_events.append(switch_data)
        print(f"\n✓ Switch detected!")
        print(f"  Old: {switch_data['old_project']}")
        print(f"  New: {switch_data['new_project']}")
        print(f"  Project: {switch_data['project_info']['name']}")
        print(f"  Type: {switch_data['project_info']['type']}")

    # Create monitor
    monitor = WorkspaceMonitor(
        check_interval=1, on_switch_callback=on_switch  # Fast for testing
    )

    # Test project detection
    current = monitor.get_current_project()
    print(f"\n✓ Current project detected:")
    print(f"  ID: {current['project_id']}")
    print(f"  Name: {current['info']['name']}")
    print(f"  Type: {current['info']['type']}")
    print(f"  Path: {current['workspace']}")

    # Start monitoring
    monitor.start()
    print(f"\n✓ Monitor started (checking every {monitor.check_interval}s)")

    # Run for a few seconds
    print("\n⏱  Running for 3 seconds (try changing directory in another terminal)...")
    await asyncio.sleep(3)

    # Stop monitor
    monitor.stop()
    print(f"\n✓ Monitor stopped")

    if switch_events:
        print(f"\n✓ Detected {len(switch_events)} project switches")
    else:
        print(
            "\n✓ No project switches detected (expected if you didn't change directory)"
        )

    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
