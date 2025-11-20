#!/usr/bin/env python3
"""
Syntax verification for MCP server session memory integration.

This script verifies that the code has correct syntax and structure
without requiring runtime dependencies.
"""

import ast
import sys
from pathlib import Path


def verify_syntax(file_path):
    """Verify Python syntax by parsing the AST."""
    print("=" * 60)
    print(f"Verifying syntax: {file_path}")
    print("=" * 60)

    with open(file_path, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
        print("✓ Syntax is valid")
        return tree
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return None


def find_imports(tree):
    """Find all import statements in the AST."""
    print("\n" + "=" * 60)
    print("Checking Imports")
    print("=" * 60)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")

    # Check for session memory imports
    session_imports = [
        imp
        for imp in imports
        if "session_manager" in imp.lower()
        or "project_manager" in imp.lower()
        or "session_persistence" in imp.lower()
    ]

    if session_imports:
        print("✓ Session memory imports found:")
        for imp in session_imports:
            print(f"  - {imp}")
    else:
        print("✗ Session memory imports NOT found")

    return len(session_imports) > 0


def find_global_variables(tree):
    """Find module-level variable assignments."""
    print("\n" + "=" * 60)
    print("Checking Global Variables")
    print("=" * 60)

    variables = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append(target.id)

    # Check for session memory variables
    required_vars = [
        "_SESSION_MANAGER",
        "_PROJECT_MANAGER",
        "_PERSISTENCE_HOOK",
        "_SESSION_DB_PATH",
    ]

    found_vars = []
    for var in required_vars:
        if var in variables:
            found_vars.append(var)
            print(f"✓ {var} found")
        else:
            print(f"✗ {var} NOT found")

    return len(found_vars) == len(required_vars)


def find_functions(tree):
    """Find function definitions."""
    print("\n" + "=" * 60)
    print("Checking Function Definitions")
    print("=" * 60)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)

    # Check for required functions
    required_funcs = [
        "_start_session",
        "_end_session",
        "_extract_params",
        "_track_tool_call",
    ]

    found_funcs = []
    for func in required_funcs:
        if func in functions:
            found_funcs.append(func)
            print(f"✓ {func}() found")
        else:
            print(f"✗ {func}() NOT found")

    return len(found_funcs) == len(required_funcs)


def check_function_content(file_path):
    """Check that functions contain expected code patterns."""
    print("\n" + "=" * 60)
    print("Checking Function Content")
    print("=" * 60)

    with open(file_path, "r") as f:
        source = f.read()

    checks = [
        (
            "_start_session",
            "SESSION_MEMORY_ENABLED",
            "_start_session checks SESSION_MEMORY_ENABLED",
        ),
        (
            "_start_session",
            "SessionManager(",
            "_start_session initializes SessionManager",
        ),
        (
            "_start_session",
            "ProjectManager(",
            "_start_session initializes ProjectManager",
        ),
        (
            "_start_session",
            "SessionPersistenceHook(",
            "_start_session initializes SessionPersistenceHook",
        ),
        (
            "_start_session",
            "start_idle_monitoring",
            "_start_session starts idle monitoring",
        ),
        (
            "_end_session",
            "finalize_session",
            "_end_session calls finalize_session",
        ),
        (
            "_end_session",
            "stop_idle_monitoring",
            "_end_session stops idle monitoring",
        ),
        (
            "tracked_async",
            "_PERSISTENCE_HOOK",
            "tracked_async checks persistence hook",
        ),
        (
            "tracked_async",
            "before_tool_execution",
            "tracked_async calls before hook",
        ),
        (
            "tracked_async",
            "after_tool_execution",
            "tracked_async calls after hook",
        ),
        (
            "tracked_sync",
            "_PERSISTENCE_HOOK",
            "tracked_sync checks persistence hook",
        ),
        (
            "tracked_sync",
            "before_tool_execution",
            "tracked_sync calls before hook",
        ),
        (
            "tracked_sync",
            "after_tool_execution",
            "tracked_sync calls after hook",
        ),
    ]

    all_passed = True
    for func_name, pattern, description in checks:
        # Find function definition
        func_start = source.find(f"def {func_name}(")
        if func_start == -1:
            func_start = source.find(f"async def {func_name}(")

        if func_start == -1:
            print(f"✗ {description} - function not found")
            all_passed = False
            continue

        # Find next function definition
        next_func = source.find("\n    def ", func_start + 1)
        if next_func == -1:
            next_func = source.find("\nasync def ", func_start + 1)
        if next_func == -1:
            next_func = len(source)

        func_content = source[func_start:next_func]

        if pattern in func_content:
            print(f"✓ {description}")
        else:
            print(f"✗ {description} - pattern not found")
            all_passed = False

    return all_passed


def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  MCP Server Syntax Verification                        ║")
    print("╚" + "=" * 58 + "╝")
    print()

    file_path = Path(__file__).parent / "omnimemory_mcp.py"

    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return 1

    # Parse AST
    tree = verify_syntax(file_path)
    if not tree:
        return 1

    # Run checks
    results = []
    results.append(("Imports", find_imports(tree)))
    results.append(("Global Variables", find_global_variables(tree)))
    results.append(("Function Definitions", find_functions(tree)))
    results.append(("Function Content", check_function_content(file_path)))

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All syntax checks PASSED! Integration structure is correct.")
        return 0
    else:
        print("\n❌ Some checks FAILED. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
