#!/usr/bin/env python3
"""Test that api_server can start without torch dependency"""

import sys
import os


# Block torch completely
class TorchBlocker:
    def find_module(self, fullname, path=None):
        if "torch" in fullname.lower():
            return self

    def load_module(self, fullname):
        raise ImportError(f"torch import blocked by test: {fullname}")


sys.meta_path.insert(0, TorchBlocker())

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("Testing api_server imports with torch blocked...")

try:
    # Test that we can import the core components
    print("\n1. Testing basic imports...")
    from dataclasses import dataclass, field
    from typing import Optional, Dict, Any, List
    from pathlib import Path
    import logging

    print("   ✅ Basic imports OK")

    # Test FastAPI imports (these should not require torch)
    print("\n2. Testing FastAPI imports...")
    from fastapi import FastAPI
    from pydantic import BaseModel

    print("   ✅ FastAPI imports OK")

    # Test that our standalone config can be created
    print("\n3. Testing standalone config...")

    # Import config code from api_server
    exec(
        compile(open("src/api_server.py").read(), "api_server.py", "exec"),
        {
            "__name__": "__main__",
            "__file__": "src/api_server.py",
            "Dict": Dict,
            "Any": Any,
            "List": List,
            "Optional": Optional,
            "dataclass": dataclass,
            "field": field,
            "Path": Path,
            "logger": logging.getLogger(__name__),
        },
    )

    print("   ✅ Config classes defined OK")

    print("\n✅ SUCCESS: API server core components load without torch!")
    print("\nNote: Full server startup requires providers (MLX/OpenAI/Gemini),")
    print("but the configuration layer is now torch-independent.")

except ImportError as e:
    if "torch" in str(e).lower():
        print(f"\n❌ FAILED: torch was imported: {e}")
        sys.exit(1)
    else:
        print(f"\n❌ Import error (not torch-related): {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
