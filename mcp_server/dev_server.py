#!/usr/bin/env python3
"""
Development server with hot reload for OmniMemory MCP Server

Usage:
    python3 dev_server.py

Features:
- Auto-restarts on file changes in mcp_server/ directory
- Only for development (DO NOT use in production)
- Monitors .py files for changes
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# MCP server script
MCP_SERVER_SCRIPT = SCRIPT_DIR / "omnimemory_mcp.py"


class MCPServerReloader(FileSystemEventHandler):
    """Handles file changes and restarts MCP server"""

    def __init__(self):
        self.process = None
        self.last_restart = 0
        self.restart_delay = 1.0  # Debounce restarts (1 second)
        self.start_server()

    def start_server(self):
        """Start the MCP server process"""
        if self.process:
            print("🔄 Stopping previous server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        print(f"🚀 Starting MCP server: {MCP_SERVER_SCRIPT}")
        print("-" * 80)

        # Start MCP server in subprocess
        self.process = subprocess.Popen(
            [sys.executable, str(MCP_SERVER_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            env=os.environ.copy(),
        )
        self.last_restart = time.time()

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        # Only restart for .py files
        if not event.src_path.endswith(".py"):
            return

        # Debounce: ignore rapid successive changes
        now = time.time()
        if now - self.last_restart < self.restart_delay:
            return

        print(f"\n📝 Detected change: {event.src_path}")
        self.start_server()


def main():
    """Run development server with hot reload"""
    print("=" * 80)
    print("🔥 OmniMemory MCP Server - Development Mode with Hot Reload")
    print("=" * 80)
    print(f"Watching: {SCRIPT_DIR}")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    # Create reloader
    reloader = MCPServerReloader()

    # Set up file watcher
    observer = Observer()
    observer.schedule(reloader, str(SCRIPT_DIR), recursive=False)
    observer.start()

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if process crashed
            if reloader.process and reloader.process.poll() is not None:
                print(f"\n⚠️  MCP server exited with code {reloader.process.poll()}")
                print("🔄 Restarting in 2 seconds...")
                time.sleep(2)
                reloader.start_server()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down development server...")
        observer.stop()
        if reloader.process:
            reloader.process.terminate()
            reloader.process.wait()

    observer.join()
    print("✓ Stopped")


if __name__ == "__main__":
    # Check if watchdog is installed
    try:
        import watchdog
    except ImportError:
        print("❌ Error: watchdog library not installed")
        print("Install it with: pip install watchdog")
        sys.exit(1)

    main()
