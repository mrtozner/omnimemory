"""
Admin CLI for OmniMemory API Key Management

Provides commands for managing API keys, viewing usage, and managing users.

Usage:
    python -m src.admin_cli create-key --tier free --email user@example.com
    python -m src.admin_cli verify-key <api-key>
    python -m src.admin_cli usage-stats <api-key>
    python -m src.admin_cli list-users
    python -m src.admin_cli reset-usage
"""

import sys
import os
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click not installed. Install with: pip install click>=8.1.0")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    print("Error: tabulate not installed. Install with: pip install tabulate>=0.9.0")
    sys.exit(1)

from .auth import APIKeyAuth, User
from .usage_tracker import UsageTracker
from .rate_limiter import RateLimiter


@click.group()
def cli():
    """OmniMemory API Key Management CLI

    Admin commands for managing API keys, users, and quotas.
    """
    pass


@cli.command()
@click.option(
    "--tier",
    type=click.Choice(["free", "pro", "enterprise"]),
    default="free",
    help="Subscription tier",
)
@click.option(
    "--user-id", required=True, help="User identifier (email, username, etc.)"
)
@click.option("--email", help="User email address")
@click.option("--company", help="Company name")
def create_key(tier: str, user_id: str, email: Optional[str], company: Optional[str]):
    """Create a new API key

    Examples:
        # Create free tier key
        python -m src.admin_cli create-key --user-id user123 --tier free

        # Create pro tier key with email
        python -m src.admin_cli create-key --user-id user123 --tier pro --email user@example.com

        # Create enterprise key
        python -m src.admin_cli create-key --user-id acme-corp --tier enterprise --company "Acme Corp"
    """
    try:
        auth = APIKeyAuth()
        api_key = auth.create_api_key(user_id=user_id, tier=tier)

        # Get the created user info
        user = auth.verify_api_key(api_key)

        click.echo("\n" + "=" * 70)
        click.echo("✅ API KEY CREATED SUCCESSFULLY")
        click.echo("=" * 70)
        click.echo(f"\nAPI Key:      {click.style(api_key, fg='green', bold=True)}")
        click.echo(f"User ID:      {user_id}")
        click.echo(f"Tier:         {tier}")
        click.echo(f"Email:        {email or 'N/A'}")
        click.echo(f"Company:      {company or 'N/A'}")
        click.echo(f"Monthly Limit: {user.monthly_limit:,} tokens")
        click.echo("\n" + "=" * 70)
        click.echo("⚠️  IMPORTANT: Save this key - it won't be shown again!")
        click.echo("=" * 70 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Error creating API key: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("api_key")
def verify_key(api_key: str):
    """Verify an API key and show user information

    Args:
        api_key: The API key to verify

    Examples:
        python -m src.admin_cli verify-key om_free_abc123...
    """
    try:
        auth = APIKeyAuth()
        user = auth.verify_api_key(api_key)

        if user:
            click.echo("\n" + "=" * 70)
            click.echo("✅ VALID API KEY")
            click.echo("=" * 70)
            click.echo(f"\nUser ID:       {user.user_id}")
            click.echo(f"Tier:          {user.tier}")
            click.echo(f"Monthly Limit: {user.monthly_limit:,} tokens")
            click.echo(f"Current Usage: {user.current_usage:,} tokens")
            click.echo(
                f"Remaining:     {user.monthly_limit - user.current_usage:,} tokens"
            )

            usage_percent = (
                (user.current_usage / user.monthly_limit * 100)
                if user.monthly_limit > 0
                else 0
            )
            click.echo(f"Usage:         {usage_percent:.1f}%")

            # Show usage bar
            bar_length = 40
            filled = int(bar_length * usage_percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            click.echo(f"\n[{bar}] {usage_percent:.1f}%")
            click.echo("=" * 70 + "\n")
        else:
            click.echo("\n❌ Invalid API key\n")
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error verifying API key: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("api_key")
@click.option("--days", default=30, help="Number of days to show (default: 30)")
@click.option("--detailed", is_flag=True, help="Show detailed breakdown")
def usage_stats(api_key: str, days: int, detailed: bool):
    """Get usage statistics for an API key

    Args:
        api_key: The API key to check

    Examples:
        # Basic stats
        python -m src.admin_cli usage-stats om_free_abc123...

        # Last 7 days
        python -m src.admin_cli usage-stats om_free_abc123... --days 7

        # Detailed breakdown
        python -m src.admin_cli usage-stats om_free_abc123... --detailed
    """
    try:
        auth = APIKeyAuth()
        tracker = UsageTracker()

        # Verify key exists
        user = auth.verify_api_key(api_key)
        if not user:
            click.echo("\n❌ Invalid API key\n")
            sys.exit(1)

        # Get usage statistics
        stats = tracker.get_usage_stats(api_key=api_key)

        click.echo("\n" + "=" * 70)
        click.echo("📊 USAGE STATISTICS")
        click.echo("=" * 70)
        click.echo(f"\nUser ID:              {user.user_id}")
        click.echo(f"Tier:                 {user.tier}")
        click.echo(f"\nTotal Compressions:   {stats['total_compressions']:,}")
        click.echo(f"Original Tokens:      {stats['total_original_tokens']:,}")
        click.echo(f"Compressed Tokens:    {stats['total_compressed_tokens']:,}")
        click.echo(f"Tokens Saved:         {stats['total_tokens_saved']:,}")
        click.echo(f"\nAvg Compression:      {stats['avg_compression_ratio']:.2%}")
        click.echo(f"Avg Quality Score:    {stats['avg_quality_score']:.2%}")

        if stats["first_used"]:
            click.echo(f"\nFirst Used:           {stats['first_used']}")
            click.echo(f"Last Used:            {stats['last_used']}")

        # Show model breakdown
        if stats.get("by_model"):
            click.echo("\n" + "-" * 70)
            click.echo("BY MODEL:")
            click.echo("-" * 70)

            model_data = []
            for model_stats in stats["by_model"]:
                model_data.append(
                    [
                        model_stats["model_id"],
                        f"{model_stats['count']:,}",
                        f"{model_stats['tokens_saved']:,}",
                    ]
                )

            click.echo(
                tabulate(
                    model_data,
                    headers=["Model", "Requests", "Tokens Saved"],
                    tablefmt="simple",
                )
            )

        # Show recent usage if detailed
        if detailed:
            recent = tracker.get_recent_usage(api_key=api_key, limit=10)

            if recent:
                click.echo("\n" + "-" * 70)
                click.echo("RECENT USAGE (Last 10 requests):")
                click.echo("-" * 70)

                recent_data = []
                for record in recent:
                    recent_data.append(
                        [
                            record["timestamp"][:19],  # Truncate timestamp
                            record["model_id"],
                            f"{record['original_tokens']:,}",
                            f"{record['compressed_tokens']:,}",
                            f"{record['compression_ratio']:.1%}"
                            if record["compression_ratio"]
                            else "N/A",
                        ]
                    )

                click.echo(
                    tabulate(
                        recent_data,
                        headers=[
                            "Timestamp",
                            "Model",
                            "Original",
                            "Compressed",
                            "Ratio",
                        ],
                        tablefmt="simple",
                    )
                )

        click.echo("=" * 70 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Error getting usage stats: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--tier", help="Filter by tier")
@click.option(
    "--limit", default=50, help="Maximum number of users to show (default: 50)"
)
def list_users(tier: Optional[str], limit: int):
    """List all API keys and users

    Examples:
        # List all users
        python -m src.admin_cli list-users

        # List only pro tier users
        python -m src.admin_cli list-users --tier pro

        # List first 10 users
        python -m src.admin_cli list-users --limit 10
    """
    try:
        auth = APIKeyAuth()

        # Query database directly for listing
        import sqlite3

        conn = sqlite3.connect(auth.db_path)
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT user_id, tier, monthly_limit, current_usage,
                   created_at, last_used_at, is_active
            FROM api_keys
            WHERE 1=1
        """
        params = []

        if tier:
            query += " AND tier = ?"
            params.append(tier)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        users = cursor.fetchall()
        conn.close()

        if not users:
            click.echo("\n📋 No users found\n")
            return

        click.echo("\n" + "=" * 70)
        click.echo("👥 API KEYS & USERS")
        click.echo("=" * 70 + "\n")

        # Prepare table data
        table_data = []
        for user in users:
            (
                user_id,
                tier,
                monthly_limit,
                current_usage,
                created_at,
                last_used_at,
                is_active,
            ) = user

            usage_percent = (
                (current_usage / monthly_limit * 100) if monthly_limit > 0 else 0
            )
            status = "✅" if is_active else "❌"

            table_data.append(
                [
                    user_id[:20],  # Truncate long IDs
                    tier,
                    f"{monthly_limit:,}",
                    f"{current_usage:,}",
                    f"{usage_percent:.1f}%",
                    status,
                    created_at[:10] if created_at else "N/A",
                ]
            )

        click.echo(
            tabulate(
                table_data,
                headers=["User ID", "Tier", "Limit", "Usage", "%", "Active", "Created"],
                tablefmt="grid",
            )
        )

        click.echo(f"\nShowing {len(users)} of {len(users)} users")
        click.echo("=" * 70 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Error listing users: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.confirmation_option(
    prompt="Are you sure you want to reset monthly usage for ALL users?"
)
def reset_usage():
    """Reset monthly usage for all users

    This should typically be run monthly via cron job.

    ⚠️  WARNING: This will reset current_usage to 0 for all users!

    Examples:
        python -m src.admin_cli reset-usage
    """
    try:
        auth = APIKeyAuth()
        auth.reset_monthly_usage()

        click.echo("\n✅ Monthly usage reset for all users\n")

    except Exception as e:
        click.echo(f"\n❌ Error resetting usage: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("api_key")
def deactivate_key(api_key: str):
    """Deactivate an API key (without deleting it)

    Args:
        api_key: The API key to deactivate

    Examples:
        python -m src.admin_cli deactivate-key om_free_abc123...
    """
    try:
        auth = APIKeyAuth()

        # Verify key exists
        user = auth.verify_api_key(api_key)
        if not user:
            click.echo("\n❌ Invalid API key\n")
            sys.exit(1)

        # Deactivate
        import sqlite3

        conn = sqlite3.connect(auth.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE api_keys SET is_active = 0 WHERE api_key = ?", (api_key,)
        )

        conn.commit()
        conn.close()

        click.echo(f"\n✅ API key deactivated for user: {user.user_id}\n")

    except Exception as e:
        click.echo(f"\n❌ Error deactivating key: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("api_key")
def activate_key(api_key: str):
    """Reactivate a deactivated API key

    Args:
        api_key: The API key to activate

    Examples:
        python -m src.admin_cli activate-key om_free_abc123...
    """
    try:
        auth = APIKeyAuth()

        # Check if key exists (even if deactivated)
        import sqlite3

        conn = sqlite3.connect(auth.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, is_active FROM api_keys WHERE api_key = ?", (api_key,)
        )
        result = cursor.fetchone()

        if not result:
            click.echo("\n❌ API key not found\n")
            conn.close()
            sys.exit(1)

        user_id, is_active = result

        if is_active:
            click.echo(f"\n✅ API key is already active for user: {user_id}\n")
            conn.close()
            return

        # Activate
        cursor.execute(
            "UPDATE api_keys SET is_active = 1 WHERE api_key = ?", (api_key,)
        )

        conn.commit()
        conn.close()

        click.echo(f"\n✅ API key activated for user: {user_id}\n")

    except Exception as e:
        click.echo(f"\n❌ Error activating key: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("api_key")
def rate_limit_info(api_key: str):
    """Show rate limit information for an API key

    Args:
        api_key: The API key to check

    Examples:
        python -m src.admin_cli rate-limit-info om_free_abc123...
    """
    try:
        auth = APIKeyAuth()
        limiter = RateLimiter()

        # Verify key
        user = auth.verify_api_key(api_key)
        if not user:
            click.echo("\n❌ Invalid API key\n")
            sys.exit(1)

        # Get rate limit info
        quota = limiter.get_remaining_quota(api_key, user.tier)

        click.echo("\n" + "=" * 70)
        click.echo("⏱️  RATE LIMIT INFORMATION")
        click.echo("=" * 70)
        click.echo(f"\nUser ID:          {user.user_id}")
        click.echo(f"Tier:             {user.tier}")
        click.echo(f"\nTokens Available: {quota['tokens_available']:,}")
        click.echo(f"Tokens Capacity:  {quota['tokens_capacity']:,}")
        click.echo(f"Refill Rate:      {quota['refill_rate']:.2f} tokens/sec")
        click.echo(f"Usage:            {quota['usage_percent']:.1f}%")

        # Show rate limits for tier
        request_limits = {
            "free": "1 req/sec",
            "pro": "10 req/sec",
            "enterprise": "100 req/sec",
        }
        click.echo(f"\nRequest Limit:    {request_limits.get(user.tier, 'Unknown')}")

        click.echo("=" * 70 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Error getting rate limit info: {e}", err=True)
        sys.exit(1)


@cli.command()
def database_info():
    """Show database information and statistics

    Examples:
        python -m src.admin_cli database-info
    """
    try:
        auth = APIKeyAuth()
        tracker = UsageTracker()

        import sqlite3

        # Get API keys stats
        conn = sqlite3.connect(auth.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM api_keys")
        total_keys = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
        active_keys = cursor.fetchone()[0]

        cursor.execute("SELECT tier, COUNT(*) FROM api_keys GROUP BY tier")
        tier_counts = cursor.fetchall()

        conn.close()

        # Get usage stats
        conn = sqlite3.connect(tracker.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM usage_records")
        total_records = cursor.fetchone()[0]

        conn.close()

        click.echo("\n" + "=" * 70)
        click.echo("💾 DATABASE INFORMATION")
        click.echo("=" * 70)
        click.echo(f"\nAPI Keys Database:    {auth.db_path}")
        click.echo(f"Usage Tracking DB:    {tracker.db_path}")
        click.echo(f"\nTotal API Keys:       {total_keys}")
        click.echo(f"Active Keys:          {active_keys}")
        click.echo(f"Total Usage Records:  {total_records:,}")

        click.echo("\n" + "-" * 70)
        click.echo("KEYS BY TIER:")
        click.echo("-" * 70)
        for tier, count in tier_counts:
            click.echo(f"  {tier:12} {count} keys")

        click.echo("=" * 70 + "\n")

    except Exception as e:
        click.echo(f"\n❌ Error getting database info: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
