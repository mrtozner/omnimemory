#!/usr/bin/env python3
"""
Test script for OmniMemory MCP server
Tests that the MCP server can start and respond to requests
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the OmniMemory MCP server"""

    server_params = StdioServerParameters(
        command="/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/.venv/bin/python",
        args=[
            "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py"
        ],
    )

    print("🚀 Testing OmniMemory MCP Server")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            print("\n✓ Connected to MCP server")

            # Initialize the session
            await session.initialize()
            print("✓ Session initialized")

            # List available tools
            print("\n📋 Listing available tools...")
            tools = await session.list_tools()

            print(f"\n✓ Found {len(tools.tools)} tools:")
            for tool in tools.tools:
                print(f"  • {tool.name}: {tool.description}")

            # Test calling a tool
            print("\n🧪 Testing omnimemory_get_stats tool...")
            try:
                result = await session.call_tool("omnimemory_get_stats", {})
                print(f"✓ Tool executed successfully")

                # Parse the result
                if result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        stats = json.loads(content.text)
                        print(f"\n📊 Service Statistics:")
                        for service, data in stats.items():
                            print(f"  {service}: {data.get('status', 'N/A')}")

            except Exception as e:
                print(f"⚠️  Tool call failed: {e}")

            print("\n" + "=" * 60)
            print("✅ MCP Server Test Complete!")
            print("\nNext steps:")
            print("1. Add config to ~/.config/claude/config.json")
            print("2. Restart Claude Code")
            print("3. Run: /mcp list")
            print("4. You should see 'omnimemory' with 6 tools!")


if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_server())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        print("\nMake sure:")
        print("  • All OmniMemory services are running (ports 8000-8002)")
        print("  • MCP server dependencies are installed (uv sync)")
