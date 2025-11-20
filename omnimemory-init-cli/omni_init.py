#!/usr/bin/env python3
"""
OMN1 Init CLI - Configure AI tools for MCP-based memory

Usage:
    omni init --tool claude
    omni init --tool codex
    omni init --tool cody
    omni init --tool continue
    omni init --tool cursor
    omni init --tool gemini
    omni init --tool vscode
    omni init --tool windsurf
    omni status
"""

import sys
import time
import shutil
import click
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.configurators.claude import ClaudeConfigurator
from src.configurators.codex import CodexConfigurator
from src.configurators.continuedev import ContinueConfigurator
from src.configurators.cursor import CursorConfigurator
from src.configurators.vscode import VSCodeConfigurator
from src.configurators.windsurf import WindsurfConfigurator
from src.configurators.gemini import GeminiConfigurator
from src.configurators.cody import CodyConfigurator

console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Configure AI tools to use OMN1 MCP server."""
    pass


@cli.command()
@click.option(
    "--tool",
    type=click.Choice(
        [
            "claude",
            "codex",
            "cody",
            "continue",
            "cursor",
            "gemini",
            "vscode",
            "windsurf",
        ]
    ),
    required=True,
    help="Tool to configure",
)
def init(tool: str):
    """Initialize MCP for specified tool."""

    console.print(
        Panel.fit(
            "[bold blue]OMN1 MCP Configuration[/bold blue]\n"
            "[dim]Configure AI tools to use OMN1 via MCP[/dim]",
            border_style="blue",
        )
    )

    # Auto-detect OMNI_ROOT from script location
    omni_root = Path(__file__).parent.parent
    mcp_server = omni_root / "mcp_server"
    venv_python = mcp_server / ".venv" / "bin" / "python"
    mcp_script = mcp_server / "omnimemory_mcp.py"

    # Verify paths exist
    if not venv_python.exists():
        console.print(f"\n[red]✗ Error: MCP server venv not found[/red]")
        console.print(f"[dim]Expected: {venv_python}[/dim]\n")
        console.print("Please create the venv first:")
        console.print(f"  cd {mcp_server}")
        console.print("  python3 -m venv .venv")
        console.print("  .venv/bin/pip install -r requirements.txt\n")
        sys.exit(1)

    if not mcp_script.exists():
        console.print(f"\n[red]✗ Error: MCP server script not found[/red]")
        console.print(f"[dim]Expected: {mcp_script}[/dim]\n")
        sys.exit(1)

    # Check backend services
    console.print(f"\n[bold cyan]Checking backend services...[/bold cyan]")
    try:
        import requests

        response = requests.get("http://localhost:8003/health", timeout=2)
        if response.status_code == 200:
            console.print("[green]✅ Backend services are running[/green]\n")
        else:
            console.print("[yellow]⚠️  Backend services returned error[/yellow]\n")
    except ImportError:
        console.print(
            "[yellow]⚠️  requests library not installed (can't check services)[/yellow]"
        )
        console.print("[dim]Install with: pip install requests[/dim]\n")
    except Exception:
        console.print("[yellow]⚠️  Backend services not running[/yellow]")
        console.print(
            f"[dim]Start them with: cd {omni_root} && ./scripts/start_all.sh[/dim]\n"
        )

    # Configure tool
    console.print(f"[bold cyan]Configuring {tool.capitalize()}...[/bold cyan]\n")

    configurators = {
        "claude": ClaudeConfigurator(str(venv_python), str(mcp_script)),
        "codex": CodexConfigurator(str(venv_python), str(mcp_script)),
        "cody": CodyConfigurator(str(venv_python), str(mcp_script)),
        "continue": ContinueConfigurator(str(venv_python), str(mcp_script)),
        "cursor": CursorConfigurator(str(venv_python), str(mcp_script)),
        "gemini": GeminiConfigurator(str(venv_python), str(mcp_script)),
        "vscode": VSCodeConfigurator(str(venv_python), str(mcp_script)),
        "windsurf": WindsurfConfigurator(str(venv_python), str(mcp_script)),
    }

    configurator = configurators[tool]

    try:
        config_path = configurator.configure()
        console.print(f"[green]✅ {tool.capitalize()} configured successfully![/green]")
        console.print(f"[dim]   Config file: {config_path}[/dim]\n")

        # Print next steps
        console.print(
            Panel.fit(
                f"[bold green]Configuration Complete![/bold green]\n\n"
                f"[bold]Next steps:[/bold]\n"
                f"1. Completely quit {tool.capitalize()} (not just close window)\n"
                f"2. Relaunch {tool.capitalize()}\n"
                f"3. Wait 10 seconds for MCP to initialize\n"
                f"4. Check that OMN1 tools are available (20 tools total)\n\n"
                f"[bold]Example tools:[/bold]\n"
                f"• omn1_smart_read - Read files with compression\n"
                f"• omn1_semantic_search - Semantic search\n"
                f"• omn1_compress - Compress text\n"
                f"• omn1_checkpoint_conversation - Save conversation state",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]✗ Error configuring {tool}: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--tool",
    type=click.Choice(
        [
            "claude",
            "codex",
            "cody",
            "continue",
            "cursor",
            "gemini",
            "vscode",
            "windsurf",
        ]
    ),
    required=True,
    help="Tool to remove OmniMemory from",
)
def remove(tool: str):
    """Remove OmniMemory configuration from specified tool"""

    console.print(
        Panel.fit(
            "[bold red]Remove OmniMemory Configuration[/bold red]\n"
            "[dim]This will restore your tool to its previous state[/dim]",
            border_style="red",
        )
    )

    configurators = {
        "claude": ClaudeConfigurator,
        "codex": CodexConfigurator,
        "cody": CodyConfigurator,
        "continue": ContinueConfigurator,
        "cursor": CursorConfigurator,
        "gemini": GeminiConfigurator,
        "vscode": VSCodeConfigurator,
        "windsurf": WindsurfConfigurator,
    }

    # Dummy paths (not used for removal, just to instantiate)
    configurator = configurators[tool]("", "")

    # Get config path
    config_path = configurator.get_config_path()

    if not config_path.exists():
        console.print(f"\n[yellow]⚠️  {tool.capitalize()} config not found[/yellow]")
        console.print(f"[dim]Path: {config_path}[/dim]")
        console.print("\nNothing to remove.")
        sys.exit(0)

    # Find backups
    backup_dir = config_path.parent
    backups = sorted(
        backup_dir.glob(f"{config_path.name}.backup-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not backups:
        console.print(f"\n[red]✗ No backups found[/red]")
        console.print("[yellow]⚠️  Cannot safely remove without backup[/yellow]")
        console.print("\nOptions:")
        console.print("1. Manually edit config to remove OmniMemory")
        console.print("2. Reinstall tool to reset configuration")
        sys.exit(1)

    # Show latest backup
    latest_backup = backups[0]
    backup_date = datetime.fromtimestamp(latest_backup.stat().st_mtime)

    console.print(f"\n[cyan]Latest backup found:[/cyan]")
    console.print(f"  File: {latest_backup.name}")
    console.print(f"  Date: {backup_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Confirm removal
    if not click.confirm(
        f"\nRestore backup and remove OmniMemory from {tool.capitalize()}?"
    ):
        console.print("[yellow]Cancelled[/yellow]")
        sys.exit(0)

    # Restore backup
    try:
        # Create safety backup before restoring
        safety_backup = (
            backup_dir / f"{config_path.name}.before-removal-{int(time.time())}"
        )
        shutil.copy(config_path, safety_backup)

        # Restore from backup
        shutil.copy(latest_backup, config_path)

        console.print(f"\n[green]✅ Configuration restored from backup[/green]")
        console.print(f"[green]✅ OmniMemory removed from {tool.capitalize()}[/green]")
        console.print(f"[dim]   Safety backup: {safety_backup.name}[/dim]")

        console.print(
            Panel.fit(
                f"[bold green]Removal Complete![/bold green]\n\n"
                f"[bold]Next steps:[/bold]\n"
                f"1. Completely quit {tool.capitalize()}\n"
                f"2. Relaunch {tool.capitalize()}\n"
                f"3. OmniMemory tools will no longer be available\n\n"
                f"[bold]To re-enable:[/bold]\n"
                f"Run: omni init --tool {tool}",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[red]✗ Error removing configuration: {e}[/red]")
        console.print(
            f"[yellow]Your original config was backed up to: {safety_backup.name}[/yellow]"
        )
        sys.exit(1)


@cli.command()
def status():
    """Check OMN1 service status."""

    console.print(
        Panel.fit(
            "[bold blue]OMN1 Status Check[/bold blue]",
            border_style="blue",
        )
    )

    try:
        import requests
    except ImportError:
        console.print("\n[red]✗ requests library not installed[/red]")
        console.print("[dim]Install with: pip install requests[/dim]\n")
        sys.exit(1)

    services = {
        "Embeddings": ("http://localhost:8000/stats", "8000"),
        "Compression": ("http://localhost:8001/health", "8001"),
        "Metrics": ("http://localhost:8003/health", "8003"),
        "Dashboard": ("http://localhost:8004", "8004"),
        "Qdrant": ("http://localhost:6333", "6333"),
        "Gateway": ("http://localhost:8009/health", "8009"),
    }

    # Create status table
    table = Table(show_header=True, header_style="bold cyan", title="Backend Services")
    table.add_column("Service", style="bold")
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("URL")

    all_healthy = True

    for name, (url, port) in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                status = "[green]✅ Running[/green]"
            else:
                status = "[yellow]⚠️  Error[/yellow]"
                all_healthy = False
        except Exception:
            status = "[red]❌ Offline[/red]"
            all_healthy = False

        table.add_row(name, port, status, url)

    console.print("\n")
    console.print(table)
    console.print("\n")

    if all_healthy:
        console.print("[green]✅ All services are healthy![/green]\n")
    else:
        console.print("[yellow]⚠️  Some services are not running[/yellow]")
        console.print("[dim]Start them with: ./scripts/start_all.sh[/dim]\n")


@cli.group()
def repo():
    """Manage repository-level team sharing"""
    pass


@repo.command()
@click.option("--team", required=True, help="Team ID")
@click.option(
    "--path", default=".", help="Repository path (default: current directory)"
)
@click.option("--auto-index", is_flag=True, help="Auto-index top 100 files")
def init(team: str, path: str, auto_index: bool):
    """Initialize repository for team sharing"""

    console.print(
        Panel.fit(
            "[bold blue]Repository Initialization[/bold blue]\n"
            "[dim]Enable team-level cache sharing[/dim]",
            border_style="blue",
        )
    )

    # Detect git repository
    repo_path = Path(path).resolve()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode != 0:
            console.print(f"[red]✗ Not a git repository: {repo_path}[/red]")
            sys.exit(1)

        git_root = result.stdout.strip()
        repo_hash = hashlib.sha256(git_root.encode()).hexdigest()[:16]
        repo_id = f"repo_{repo_hash}"

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)

    # Connect to Redis and add team/repo
    try:
        # Import UnifiedCacheManager
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
        from unified_cache_manager import UnifiedCacheManager

        cache = UnifiedCacheManager()

        # Add repo to team
        cache.add_repo_to_team(team, repo_id)

        console.print(f"\n[green]✅ Repository initialized for team sharing![/green]")
        console.print(f"[dim]   Team: {team}[/dim]")
        console.print(f"[dim]   Repo ID: {repo_id}[/dim]")
        console.print(f"[dim]   Path: {git_root}[/dim]\n")

        # Auto-index top files if requested
        if auto_index:
            console.print("[cyan]🔍 Auto-indexing repository...[/cyan]")

            from collections import Counter

            try:
                # Get files sorted by git activity (last 30 days)
                result = subprocess.run(
                    [
                        "git",
                        "log",
                        "--pretty=format:",
                        "--name-only",
                        "--since=30.days",
                    ],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
                    file_counts = Counter(files)
                    top_files = [f for f, _ in file_counts.most_common(100)]

                    console.print(
                        f"   Found {len(top_files)} frequently modified files"
                    )

                    # Pre-load files
                    cached_count = 0
                    total_size = 0

                    with console.status("[cyan]Caching files...[/cyan]"):
                        for file_path in top_files:
                            full_path = repo_path / file_path

                            if full_path.exists() and full_path.is_file():
                                try:
                                    # Read file
                                    with open(full_path, "rb") as f:
                                        content = f.read()

                                    if (
                                        len(content) > 10 * 1024 * 1024
                                    ):  # Skip >10MB files
                                        continue

                                    # Store in L2 (repository tier)
                                    file_hash = hashlib.sha256(
                                        str(full_path).encode()
                                    ).hexdigest()[:16]

                                    cache.cache_file_compressed(
                                        repo_id=repo_id,
                                        file_hash=file_hash,
                                        compressed_content=content,
                                        metadata={
                                            "file_path": str(full_path),
                                            "size": str(len(content)),
                                            "cached_at": str(time.time()),
                                            "pre_indexed": "true",
                                        },
                                        ttl=604800,  # 7 days
                                    )

                                    cached_count += 1
                                    total_size += len(content)

                                except Exception:
                                    # Skip file on error
                                    pass

                    console.print(
                        f"\n[green]✅ Pre-cached {cached_count} files in L2 tier[/green]"
                    )
                    console.print(
                        f"[dim]   Total size: {total_size / 1024 / 1024:.2f} MB[/dim]"
                    )
                    console.print(
                        f"[dim]   Team members will have instant access![/dim]\n"
                    )

                else:
                    console.print("[yellow]⚠️  Could not read git history[/yellow]\n")

            except Exception as e:
                console.print(f"[yellow]⚠️  Auto-indexing failed: {e}[/yellow]")
                console.print("[dim]Repository initialized without pre-caching[/dim]\n")

        console.print(
            Panel.fit(
                f"[bold green]Next Steps:[/bold green]\n\n"
                f"1. Team members can now share cached files\n"
                f"2. First member to read a file → cached for team\n"
                f"3. Other members → instant access from L2 cache\n\n"
                f"[bold]Expected savings:[/bold]\n"
                f"• 80-90% token reduction for team members\n"
                f"• 20-100x faster on repeated operations\n"
                f"• 7-day cache TTL (long-lived team benefit)",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]✗ Failed to initialize: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@repo.command()
@click.option("--team", required=True, help="Team ID")
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
def status(team: str, detailed: bool):
    """Check repository sharing status for team"""

    console.print(
        Panel.fit("[bold blue]Repository Status[/bold blue]", border_style="blue")
    )

    try:
        # Import UnifiedCacheManager
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
        from unified_cache_manager import UnifiedCacheManager

        cache = UnifiedCacheManager()

        # Get team repos
        repos = cache.get_team_repos(team)
        members = cache.get_team_members(team)
        stats = cache.get_stats()

        # Create table
        table = Table(title=f"Team: {team}")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Repositories", str(len(repos)))
        table.add_row("Team Members", str(len(members)))
        table.add_row("L2 Cache Keys", str(stats.l2_keys))
        table.add_row("Cache Hit Rate", f"{stats.hit_rate}%")
        table.add_row("Memory Used", f"{stats.memory_used_mb:.2f} MB")

        console.print("\n")
        console.print(table)
        console.print("\n")

        if repos:
            console.print("[bold]Repositories:[/bold]")
            for repo in repos:
                # Get repo cache size
                repo_size = cache.get_repo_cache_size(repo)
                console.print(f"  • {repo} ({repo_size / 1024 / 1024:.2f} MB)")

        if members:
            console.print("\n[bold]Members:[/bold]")
            for member in members:
                console.print(f"  • {member}")

        console.print("\n")

        # Show detailed statistics if requested
        if detailed:
            console.print("[bold cyan]Detailed Team Statistics[/bold cyan]\n")

            # Show member table with stats
            if members:
                member_table = Table(title="Team Members")
                member_table.add_column("Member", style="bold")
                member_table.add_column("Role")
                member_table.add_column("L1 Cache", justify="right")
                member_table.add_column("Status")

                for member in members:
                    # Get member's L1 cache size
                    cache_size = cache.get_user_cache_size(member)
                    cache_mb = cache_size / 1024 / 1024

                    # Get role
                    role_bytes = cache.redis.hget(f"team:{team}:roles", member)
                    role = role_bytes.decode() if role_bytes else "member"

                    member_table.add_row(
                        member, role.capitalize(), f"{cache_mb:.2f} MB", "Active"
                    )

                console.print(member_table)
                console.print("\n")

            # Calculate team savings
            if repos:
                console.print("[bold cyan]Token Savings Analysis[/bold cyan]\n")

                # Calculate total cached data
                total_cached = sum(cache.get_repo_cache_size(r) for r in repos)
                num_members = len(members) if members else 1

                # Without sharing: Each member caches individually
                without_sharing = total_cached * num_members

                # With sharing: Cached once in L2
                with_sharing = total_cached

                # Calculate token savings (1MB ≈ 250K tokens)
                tokens_without = (without_sharing / 1024 / 1024) * 250000
                tokens_with = (with_sharing / 1024 / 1024) * 250000
                tokens_saved = tokens_without - tokens_with
                savings_pct = (
                    (tokens_saved / tokens_without * 100) if tokens_without > 0 else 0
                )

                # Cost calculation ($0.015 per 1K tokens)
                cost_without = (tokens_without / 1000) * 0.015
                cost_with = (tokens_with / 1000) * 0.015
                cost_saved = cost_without - cost_with

                savings_panel = Panel.fit(
                    f"[bold green]Team Sharing Benefits[/bold green]\n\n"
                    f"Without L2 sharing:\n"
                    f"  • Tokens: {tokens_without:,.0f}\n"
                    f"  • Cost: ${cost_without:.2f}/month\n\n"
                    f"With L2 sharing:\n"
                    f"  • Tokens: {tokens_with:,.0f}\n"
                    f"  • Cost: ${cost_with:.2f}/month\n\n"
                    f"[bold]Team Savings:[/bold]\n"
                    f"  • {savings_pct:.1f}% token reduction\n"
                    f"  • ${cost_saved:.2f}/month saved",
                    border_style="green",
                )

                console.print(savings_panel)
                console.print("\n")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@repo.command()
@click.option("--team", required=True, help="Team ID")
@click.option("--add", help="Add member (user_id)")
@click.option("--remove", help="Remove member (user_id)")
@click.option("--role", default="member", help="Role: member, admin, viewer")
@click.option("--list", "list_members", is_flag=True, help="List all members")
def members(team: str, add: str, remove: str, role: str, list_members: bool):
    """Manage team members and roles"""

    try:
        # Import UnifiedCacheManager
        sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_server"))
        from unified_cache_manager import UnifiedCacheManager

        cache = UnifiedCacheManager()

        # Add member
        if add:
            success = cache.add_member_to_team(team, add, role)

            if success:
                console.print(
                    f"[green]✅ Added {add} to team '{team}' as {role}[/green]"
                )

                # Show team size
                members = cache.get_team_members(team)
                console.print(f"[dim]   Team now has {len(members)} member(s)[/dim]")
            else:
                console.print(f"[red]✗ Failed to add {add} to team[/red]")
                sys.exit(1)

        # Remove member
        elif remove:
            try:
                # Remove from members set
                cache.redis.srem(f"team:{team}:members", remove)
                # Remove from roles hash
                cache.redis.hdel(f"team:{team}:roles", remove)

                console.print(f"[green]✅ Removed {remove} from team '{team}'[/green]")

                members = cache.get_team_members(team)
                console.print(f"[dim]   Team now has {len(members)} member(s)[/dim]")

            except Exception as e:
                console.print(f"[red]✗ Failed to remove {remove}: {e}[/red]")
                sys.exit(1)

        # List members (default if no action specified)
        elif list_members or (not add and not remove):
            members = cache.get_team_members(team)

            if not members:
                console.print(f"\n[yellow]No members in team '{team}'[/yellow]")
                console.print(
                    f"\nAdd members with: omni repo members --team {team} --add <user_id>"
                )
                sys.exit(0)

            table = Table(title=f"Team '{team}' Members")
            table.add_column("User ID", style="bold")
            table.add_column("Role")
            table.add_column("Cache Size")

            for member in members:
                role_bytes = cache.redis.hget(f"team:{team}:roles", member)
                member_role = role_bytes.decode() if role_bytes else "member"

                cache_size = cache.get_user_cache_size(member)
                cache_mb = cache_size / 1024 / 1024

                table.add_row(member, member_role.capitalize(), f"{cache_mb:.2f} MB")

            console.print("\n")
            console.print(table)
            console.print("\n")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
