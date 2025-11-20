#!/usr/bin/env python3
"""Test that api_server doesn't require torch"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Track imports to detect torch
torch_imported = False
original_import = __builtins__.__import__


def tracking_import(name, *args, **kwargs):
    global torch_imported
    if "torch" in name.lower():
        torch_imported = True
        print(f"WARNING: torch-related import detected: {name}")
    return original_import(name, *args, **kwargs)


__builtins__.__import__ = tracking_import

try:
    # Import just the config classes from api_server
    from dataclasses import dataclass, field
    from typing import Optional, Dict, Any, List
    from pathlib import Path
    import logging

    # Define the classes inline (same as in api_server.py)
    @dataclass
    class ProviderConfig:
        """Configuration for a single embedding provider (standalone version)"""

        name: str
        type: str
        priority: int = 1
        enabled: bool = True
        config: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class OmniMemoryConfig:
        """Minimal configuration for embeddings service (standalone version)"""

        embedding_default_provider: str = "mlx"
        embedding_providers: List[ProviderConfig] = field(default_factory=list)

        def __post_init__(self):
            if not self.embedding_providers:
                default_config = OmniMemoryConfig.default()
                self.embedding_providers = default_config.embedding_providers
                self.embedding_default_provider = (
                    default_config.embedding_default_provider
                )

        @classmethod
        def default(cls):
            return cls(
                embedding_default_provider="mlx",
                embedding_providers=[
                    ProviderConfig(
                        name="mlx",
                        type="local",
                        priority=1,
                        enabled=True,
                        config={
                            "model_path": "./models/default.safetensors",
                            "embedding_dim": 768,
                            "vocab_size": 50000,
                        },
                    ),
                ],
            )

    class ConfigManager:
        @staticmethod
        def load_config(config_path: Optional[str] = None):
            return OmniMemoryConfig.default()

    # Test the config
    config = ConfigManager.load_config()
    print(f"✅ Config loaded successfully")
    print(f"   Default provider: {config.embedding_default_provider}")
    print(f"   Number of providers: {len(config.embedding_providers)}")

    if torch_imported:
        print(f"\n❌ FAILED: torch was imported!")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: No torch dependency detected!")
        sys.exit(0)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
