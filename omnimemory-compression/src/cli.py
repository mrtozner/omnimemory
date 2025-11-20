"""
CLI commands for OmniMemory Compression Service

Provides manual management commands:
- update-models: Fetch latest model metadata from providers
- show-models: Display all models in registry
- cache-stats: Show cache statistics

Usage:
    python -m src.cli update-models
    python -m src.cli show-models
    python -m src.cli cache-stats
"""

import asyncio
import logging
import sys
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click not installed. Install with: pip install click>=8.1.0")
    sys.exit(1)

from .model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """OmniMemory Compression CLI

    Manual management commands for the compression service.
    """
    pass


@cli.command()
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "--huggingface-token",
    envvar="HUGGINGFACE_TOKEN",
    help="HuggingFace token (or set HUGGINGFACE_TOKEN env var)",
)
def update_models(openai_key: Optional[str], huggingface_token: Optional[str]):
    """Update model registry from providers

    Fetches latest models from OpenAI API and HuggingFace Hub.
    Updates the local cache at ~/.omnimemory/model_registry.json

    Examples:
        # With API keys from environment
        python -m src.cli update-models

        # With explicit API key
        python -m src.cli update-models --openai-key sk-...
    """
    try:
        click.echo("🔄 Updating model registry...")

        # Run async update
        registry = ModelRegistry()
        stats = asyncio.run(
            registry.update_registry(
                openai_api_key=openai_key,
                huggingface_token=huggingface_token,
            )
        )

        # Display results
        click.echo("\n✅ Registry update complete!")
        click.echo(f"\nUpdated models:")
        click.echo(f"  - OpenAI: {stats['openai']} models")
        click.echo(f"  - HuggingFace: {stats['huggingface']} models")

        # Show cache info
        cache_stats = registry.get_cache_stats()
        click.echo(f"\nTotal models in registry: {cache_stats['total_models']}")
        click.echo(f"Cache location: {cache_stats['cache_path']}")

    except Exception as e:
        logger.error(f"Update failed: {e}")
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--family",
    help="Filter by model family (e.g., 'openai', 'anthropic')",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed model information",
)
def show_models(family: Optional[str], verbose: bool):
    """Show all models in registry

    Lists all models grouped by family. Optionally filter by family
    or show detailed information.

    Examples:
        # Show all models
        python -m src.cli show-models

        # Show only OpenAI models
        python -m src.cli show-models --family openai

        # Show with details
        python -m src.cli show-models --verbose
    """
    try:
        registry = ModelRegistry()
        models_by_family = registry.list_models()

        if not models_by_family:
            click.echo("📋 No models in registry. Run 'update-models' to populate.")
            return

        # Filter by family if specified
        if family:
            if family not in models_by_family:
                click.echo(f"❌ No models found for family '{family}'")
                click.echo(
                    f"\nAvailable families: {', '.join(models_by_family.keys())}"
                )
                return
            models_by_family = {family: models_by_family[family]}

        # Display models
        click.echo(
            f"📋 Models in registry ({sum(len(m) for m in models_by_family.values())} total)\n"
        )

        for family_name, model_ids in sorted(models_by_family.items()):
            click.echo(f"[{family_name}] ({len(model_ids)} models)")

            if verbose:
                # Show detailed info for each model
                for model_id in model_ids[:10]:  # Limit to 10 for verbosity
                    info = asyncio.run(registry.get_model_info(model_id))
                    click.echo(f"  - {model_id}")
                    click.echo(f"    Source: {info.get('source')}")
                    if "downloads" in info:
                        click.echo(f"    Downloads: {info['downloads']}")
                    if "likes" in info:
                        click.echo(f"    Likes: {info['likes']}")

                if len(model_ids) > 10:
                    click.echo(f"  ... and {len(model_ids) - 10} more")
            else:
                # Show compact list
                for model_id in model_ids[:5]:
                    click.echo(f"  - {model_id}")

                if len(model_ids) > 5:
                    click.echo(f"  ... and {len(model_ids) - 5} more")

            click.echo("")

    except Exception as e:
        logger.error(f"Failed to show models: {e}")
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def cache_stats():
    """Show cache statistics

    Displays information about the model registry cache including
    size, last update, and location.

    Example:
        python -m src.cli cache-stats
    """
    try:
        registry = ModelRegistry()
        stats = registry.get_cache_stats()

        click.echo("📊 Model Registry Cache Statistics\n")
        click.echo(f"Total models: {stats['total_models']}")
        click.echo(f"Model families: {stats['families']}")
        click.echo(f"Last updated: {stats['last_updated'] or 'Never'}")
        click.echo(f"Cache location: {stats['cache_path']}")
        click.echo(f"Cache exists: {'✅' if stats['cache_exists'] else '❌'}")

        if not stats["cache_exists"]:
            click.echo("\n💡 Run 'update-models' to create and populate the cache")

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_id")
def check_model(model_id: str):
    """Check information for a specific model

    Args:
        model_id: Model identifier to check (e.g., "gpt-4", "claude-3-5-sonnet")

    Example:
        python -m src.cli check-model gpt-4
        python -m src.cli check-model claude-3-5-sonnet
    """
    try:
        registry = ModelRegistry()
        info = asyncio.run(registry.get_model_info(model_id))

        click.echo(f"\n📝 Model Information: {model_id}\n")
        click.echo(f"Family: {info.get('family', 'unknown')}")
        click.echo(f"Source: {info.get('source', 'unknown')}")

        # Show additional metadata if available
        for key, value in info.items():
            if key not in ["model_id", "family", "source"]:
                click.echo(f"{key.replace('_', ' ').title()}: {value}")

        if info.get("source") == "pattern":
            click.echo("\n💡 This model was detected using pattern matching.")
            click.echo("   Run 'update-models' to fetch exact metadata from providers.")

    except Exception as e:
        logger.error(f"Failed to check model: {e}")
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
