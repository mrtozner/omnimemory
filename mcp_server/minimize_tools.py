#!/usr/bin/env python3
"""Script to minimize MCP tools to only essential ones"""

# Tools to remove (will be commented out)
TOOLS_TO_REMOVE = [
    ("omn1_store", 2619, 2733),
    ("omn1_compress", 2734, 2832),
    ("omn1_analyze", 2833, 2961),
    ("omn1_optimize", 2962, 3077),
    ("omn1_metrics", 3078, 3175),
    ("omn1_get_stats", 3176, 3324),
    ("omn1_learn_workflow", 3325, 3405),
    ("omn1_execute_python", 3406, 4136),
    ("omn1_graph_search", 4137, 4247),
    ("omn1_hybrid_search", 4248, 4460),
    ("omn1_workflow_context", 4589, 4763),
    ("omn1_resume_workflow", 4764, 4942),
    ("omn1_optimize_context", 4943, 5663),
    ("omn1_unified_predict", 5989, 6056),
    ("omn1_orchestrate_query", 6057, 6122),
    ("omn1_get_suggestions", 6123, 6204),
    ("omn1_record_feedback", 6205, 6282),
    ("omn1_system_status", 6283, 6431),
    ("omn1_unified_health", 6432, 6586),
]


def main():
    input_file = "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py"
    output_file = "/Users/mertozoner/Documents/claude-idea-discussion/omni-memory/mcp_server/omnimemory_mcp.py.new"

    with open(input_file, "r") as f:
        lines = f.readlines()

    # Create a set of line numbers to comment out
    lines_to_comment = set()
    for tool_name, start, end in TOOLS_TO_REMOVE:
        for line_num in range(start, end + 1):
            lines_to_comment.add(line_num)

    # Add header comment before first removed tool
    modified_lines = []
    added_header = False

    for i, line in enumerate(lines, start=1):
        # Add explanatory header before first removed tool
        if i == 2619 and not added_header:
            modified_lines.append("\n")
            modified_lines.append(
                "        # ============================================================================\n"
            )
            modified_lines.append("        # MINIMAL MCP TOOLS CONFIGURATION\n")
            modified_lines.append(
                "        # ============================================================================\n"
            )
            modified_lines.append(
                "        # Token Savings: Reduced from 24 tools to 5 essential tools\n"
            )
            modified_lines.append(
                "        # - Before: 24 tools × 400 tokens = 9,600 tokens per conversation\n"
            )
            modified_lines.append(
                "        # - After: 5 tools × 400 tokens = 2,000 tokens per conversation\n"
            )
            modified_lines.append(
                "        # - Saved: 7,600 tokens per conversation (79% reduction)\n"
            )
            modified_lines.append("        #\n")
            modified_lines.append("        # ESSENTIAL TOOLS (ACTIVE):\n")
            modified_lines.append(
                "        # 1. read - File reading with automatic compression (90% token savings)\n"
            )
            modified_lines.append(
                "        # 2. grep - Search with semantic enhancement\n"
            )
            modified_lines.append(
                "        # 3. omn1_tri_index_search - Hybrid search (Dense + Sparse + Structural)\n"
            )
            modified_lines.append(
                "        # 4. omn1_read - Unified reading (full/overview/symbol/references)\n"
            )
            modified_lines.append(
                "        # 5. omn1_search - Unified search (semantic/references)\n"
            )
            modified_lines.append("        #\n")
            modified_lines.append("        # REMOVED TOOLS (COMMENTED OUT):\n")
            modified_lines.append(
                "        # All metrics/stats/health tools have been removed. Metrics are now:\n"
            )
            modified_lines.append(
                "        # - Tracked silently in background (no token cost)\n"
            )
            modified_lines.append(
                "        # - Viewable via dashboard: http://localhost:8004\n"
            )
            modified_lines.append(
                "        # - Accessible via metrics API: http://localhost:8003\n"
            )
            modified_lines.append("        #\n")
            modified_lines.append(
                "        # The following tools are commented out to reduce token usage:\n"
            )
            for tool_name, _, _ in TOOLS_TO_REMOVE:
                modified_lines.append(f"        # - {tool_name}\n")
            modified_lines.append(
                "        # ============================================================================\n"
            )
            modified_lines.append("\n")
            added_header = True

        # Comment out lines that should be removed
        if i in lines_to_comment:
            # Add "# " prefix but preserve indentation
            if line.strip():  # Only comment non-empty lines
                modified_lines.append(
                    "        # " + line[8:]
                    if line.startswith("        ")
                    else "# " + line
                )
            else:
                modified_lines.append(line)  # Keep empty lines as-is
        else:
            modified_lines.append(line)

    # Write output
    with open(output_file, "w") as f:
        f.writelines(modified_lines)

    print(f"✓ Created minimized version: {output_file}")
    print(
        f"✓ Commented out {len(lines_to_comment)} lines across {len(TOOLS_TO_REMOVE)} tools"
    )
    print(
        f"✓ Kept 5 essential tools: read, grep, omn1_tri_index_search, omn1_read, omn1_search"
    )
    print(f"\nNext steps:")
    print(f"1. Review the changes: diff {input_file} {output_file}")
    print(f"2. If satisfied: mv {output_file} {input_file}")


if __name__ == "__main__":
    main()
