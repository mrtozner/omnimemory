#!/usr/bin/env python3
"""
Consolidate MCP tools from 4 to 2:
- read + omn1_read → unified read
- grep + omn1_search → unified search
"""

import re
from pathlib import Path


def consolidate_mcp_tools():
    """Main consolidation logic"""

    mcp_file = Path(
        "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py"
    )

    # Read the file
    with open(mcp_file, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    print(f"📄 Read {len(lines)} lines from {mcp_file.name}")

    # Step 1: Uncomment _omn1_find_references (lines 5579-5717)
    print("\n Step 1: Uncommenting _omn1_find_references...")
    in_find_refs = False
    uncommented_count = 0

    for i in range(len(lines)):
        # Start uncommenting at line 5579
        if i >= 5578 and "# async def _omn1_find_references(" in lines[i]:
            in_find_refs = True
            print(f"   Found start at line {i+1}")

        # Stop uncommenting at line 5717 (end of function)
        if in_find_refs and i >= 5716:
            in_find_refs = False
            print(f"   Found end at line {i+1}")

        # Uncomment lines in the range
        if in_find_refs and lines[i].startswith("        #"):
            lines[i] = lines[i].replace("        #", "        ", 1)
            uncommented_count += 1

    print(f"   ✓ Uncommented {uncommented_count} lines")

    # Step 2: Remove old 'read' tool (lines 2323-2472)
    print("\n✂️  Step 2: Removing old 'read' tool...")
    # Mark for deletion by commenting out
    for i in range(2322, 2472):  # 0-indexed, so 2323-2472 becomes 2322-2471
        if i < len(lines):
            lines[i] = "        # REMOVED (consolidated): " + lines[i]
    print(f"   ✓ Marked old 'read' tool for removal (lines 2323-2472)")

    # Step 3: Remove old 'grep' tool (lines 2475-2621)
    print("\n✂️  Step 3: Removing old 'grep' tool...")
    for i in range(2474, 2621):  # 0-indexed
        if i < len(lines):
            lines[i] = "        # REMOVED (consolidated): " + lines[i]
    print(f"   ✓ Marked old 'grep' tool for removal (lines 2475-2621)")

    # Step 4: Rename omn1_read → read
    print("\n🔄 Step 4: Renaming omn1_read → read...")
    content_str = "".join(lines)

    # Replace function definition
    content_str = content_str.replace(
        "        async def omn1_read(",
        "        async def read(",
        1,  # Only first occurrence
    )

    # Update docstring references
    content_str = content_str.replace(
        "omn1_read(file_path=",
        "read(file_path=",
    )
    content_str = content_str.replace(
        "omn1_read(path,",
        "read(path,",
    )

    print(f"   ✓ Renamed omn1_read → read")

    # Step 5: Rename omn1_search → search
    print("\n🔄 Step 5: Renaming omn1_search → search...")

    # Replace function definition
    content_str = content_str.replace(
        "        async def omn1_search(",
        "        async def search(",
        1,  # Only first occurrence
    )

    # Update docstring references
    content_str = content_str.replace(
        "omn1_search(query=",
        "search(query=",
    )
    content_str = content_str.replace(
        "omn1_search(mode=",
        "search(mode=",
    )

    print(f"   ✓ Renamed omn1_search → search")

    # Step 6: Add _omn1_semantic_search implementation
    print("\n➕ Step 6: Adding _omn1_semantic_search implementation...")

    # Find where to insert (before omn1_search, now renamed to search)
    search_tool_pos = content_str.find(
        "        @self.mcp.tool()\n        async def search("
    )

    if search_tool_pos > 0:
        semantic_search_impl = '''
        async def _omn1_semantic_search(
            self, query: str, limit: int = 5, min_relevance: float = 0.7
        ) -> str:
            """
            Internal semantic search implementation.

            Uses vector store (Qdrant) to find semantically relevant files.
            """
            try:
                # Check if vector store is available
                if not hasattr(self, 'vector_store') or self.vector_store is None:
                    return json.dumps({
                        "error": True,
                        "message": "Vector store not initialized for semantic search",
                        "query": query,
                        "tip": "Semantic search requires Qdrant vector store"
                    }, indent=2)

                # Perform semantic search
                results = await self.vector_store.search(
                    query=query,
                    k=limit
                )

                # Format results
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append({
                        "rank": i,
                        "file_path": result.get("file_path", "unknown"),
                        "relevance_score": result.get("score", 0.0),
                        "snippet": result.get("snippet", "")[:200],  # Truncate snippets
                        "metadata": result.get("metadata", {})
                    })

                return json.dumps({
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                    "total_found": len(formatted_results),
                    "search_type": "semantic_vector_search"
                }, indent=2)

            except Exception as e:
                return json.dumps({
                    "error": True,
                    "message": f"Semantic search failed: {str(e)}",
                    "query": query
                }, indent=2)

'''
        # Insert before search tool
        content_str = (
            content_str[:search_tool_pos]
            + semantic_search_impl
            + content_str[search_tool_pos:]
        )
        print(f"   ✓ Added _omn1_semantic_search implementation")
    else:
        print(f"   ⚠️  Warning: Could not find search tool position")

    # Step 7: Write the modified file
    output_file = Path(
        "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp_consolidated.py"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content_str)

    print(f"\n✅ Consolidation complete!")
    print(f"📝 Output written to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Uncommented _omn1_find_references helper")
    print(f"  - Removed old 'read' tool (lines 2323-2472)")
    print(f"  - Removed old 'grep' tool (lines 2475-2621)")
    print(f"  - Renamed omn1_read → read")
    print(f"  - Renamed omn1_search → search")
    print(f"  - Added _omn1_semantic_search implementation")
    print(f"\n⚠️  Review {output_file.name} before replacing the original!")


if __name__ == "__main__":
    consolidate_mcp_tools()
