"""Configuration management for OmniMemory services.

Uses Pydantic Settings for type-safe environment variable management.
All configuration can be overridden via environment variables or .env file.

Example:
    from mcp_server.config import settings

    print(settings.metrics_service_url)  # http://localhost:8003
    print(settings.allowed_origins)      # ['http://localhost:3000', ...]
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import validator, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All settings can be overridden by setting environment variables.
    For example: export METRICS_SERVICE_URL=http://production:8003
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ============================================================================
    # Service URLs
    # ============================================================================

    embeddings_service_url: str = Field(
        default="http://localhost:8000",
        description="URL for OmniMemory embeddings/search service",
    )

    compression_service_url: str = Field(
        default="http://localhost:8001",
        description="URL for OmniMemory compression service",
    )

    procedural_service_url: str = Field(
        default="http://localhost:8002",
        description="URL for OmniMemory procedural memory service",
    )

    metrics_service_url: str = Field(
        default="http://localhost:8003",
        description="URL for OmniMemory metrics service",
    )

    dashboard_url: str = Field(
        default="http://localhost:8004",
        description="URL for OmniMemory dashboard UI",
    )

    gateway_url: str = Field(
        default="http://localhost:8009", description="URL for OmniMemory gateway API"
    )

    qdrant_url: str = Field(
        default="http://localhost:6333", description="URL for Qdrant vector database"
    )

    redis_url: str = Field(
        default="redis://localhost:6379", description="URL for Redis cache"
    )

    # ============================================================================
    # Database
    # ============================================================================

    database_url: str = Field(
        default="postgresql://omnimemory:omnimemory_dev_pass@localhost:5432/omnimemory",
        description="PostgreSQL database URL",
    )

    sqlite_db_path: str = Field(
        default=os.path.expanduser("~/.omnimemory/dashboard.db"),
        description="SQLite database path for metrics (local mode)",
    )

    # ============================================================================
    # Security
    # ============================================================================

    api_key_secret: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for API key generation (MUST be changed in production)",
    )

    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8004,http://localhost:3301",
        description="Comma-separated list of allowed CORS origins",
    )

    # ============================================================================
    # Monitoring & Telemetry
    # ============================================================================

    prometheus_port: int = Field(
        default=9090,
        description="Port for Prometheus metrics endpoint",
        ge=1024,
        le=65535,
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    enable_telemetry: bool = Field(
        default=True, description="Enable OpenTelemetry instrumentation"
    )

    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint (for SigNoz)",
    )

    deployment_environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)",
    )

    # ============================================================================
    # WebSocket (Optional for future use)
    # ============================================================================

    websocket_max_connections: int = Field(
        default=1000, description="Maximum concurrent WebSocket connections", ge=1
    )

    websocket_ping_interval: int = Field(
        default=30, description="WebSocket ping interval in seconds", ge=5
    )

    # ============================================================================
    # Rate Limiting
    # ============================================================================

    rate_limit_per_minute: int = Field(
        default=100, description="Maximum requests per minute per IP", ge=1
    )

    rate_limit_burst: int = Field(
        default=20, description="Burst allowance for rate limiting", ge=1
    )

    # ============================================================================
    # Feature Flags
    # ============================================================================

    enable_dashboard: bool = Field(
        default=False, description="Enable metrics dashboard (opt-in)"
    )

    enable_mcp_tools: bool = Field(
        default=True, description="Enable MCP tool endpoints"
    )

    enable_agent_api: bool = Field(
        default=True, description="Enable REST API for autonomous agents (n8n, etc.)"
    )

    # ============================================================================
    # Privacy Settings
    # ============================================================================

    enable_path_anonymization: bool = Field(
        default=True,
        description="Anonymize file paths in API responses for cloud privacy",
    )

    path_anonymization_mode: str = Field(
        default="relative",
        description="Path anonymization mode (relative, hashed, virtual)",
    )

    project_root: str = Field(
        default=os.getcwd(),
        description="Project root directory for relative path calculation",
    )

    # ============================================================================
    # Performance
    # ============================================================================

    max_file_size_mb: int = Field(
        default=10, description="Maximum file size for compression (MB)", ge=1, le=100
    )

    compression_timeout_seconds: int = Field(
        default=30, description="Timeout for compression operations", ge=5, le=300
    )

    cache_ttl_seconds: int = Field(
        default=3600, description="Default cache TTL in seconds (1 hour)", ge=60
    )

    # ============================================================================
    # Validators
    # ============================================================================

    @validator("allowed_origins")
    def parse_allowed_origins(cls, v):
        """Parse comma-separated origins into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @validator("api_key_secret")
    def validate_api_key_secret(cls, v, values):
        """Ensure API key secret is set in production."""
        # Check environment variable directly since deployment_environment
        # might not be in values yet (field order issue)
        env = os.environ.get("DEPLOYMENT_ENVIRONMENT", "development").lower()

        if env == "production":
            if not v or v == "dev-secret-key-change-in-production":
                raise ValueError(
                    "API_KEY_SECRET must be set to a secure value in production. "
                    "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )

            if len(v) < 32:
                raise ValueError(
                    "API_KEY_SECRET must be at least 32 characters in production"
                )

        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()

        if v_upper not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")

        return v_upper

    @validator("deployment_environment")
    def validate_environment(cls, v):
        """Validate deployment environment."""
        valid_envs = ["development", "staging", "production"]
        v_lower = v.lower()

        if v_lower not in valid_envs:
            raise ValueError(
                f"DEPLOYMENT_ENVIRONMENT must be one of: {', '.join(valid_envs)}"
            )

        return v_lower

    # ============================================================================
    # Computed Properties
    # ============================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.deployment_environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.deployment_environment == "development"

    @property
    def redis_host(self) -> str:
        """Extract Redis host from URL."""
        # redis://localhost:6379 -> localhost
        return self.redis_url.split("//")[1].split(":")[0]

    @property
    def redis_port(self) -> int:
        """Extract Redis port from URL."""
        # redis://localhost:6379 -> 6379
        url_parts = self.redis_url.split("//")[1].split(":")
        return int(url_parts[1]) if len(url_parts) > 1 else 6379

    @property
    def should_anonymize_paths(self) -> bool:
        """Auto-enable path anonymization in production/staging."""
        if self.is_production or self.deployment_environment == "staging":
            return True
        return self.enable_path_anonymization


# ============================================================================
# Global Settings Instance
# ============================================================================

settings = Settings()


# ============================================================================
# Utility Functions
# ============================================================================


def get_settings() -> Settings:
    """Get the global settings instance.

    This function is useful for dependency injection in FastAPI:

    Example:
        from fastapi import Depends
        from mcp_server.config import get_settings, Settings

        @app.get("/config")
        async def show_config(settings: Settings = Depends(get_settings)):
            return {"metrics_url": settings.metrics_service_url}
    """
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables.

    Useful for testing or when environment changes at runtime.

    Example:
        import os
        from mcp_server.config import reload_settings

        os.environ["METRICS_SERVICE_URL"] = "http://new-host:8003"
        settings = reload_settings()
    """
    global settings
    settings = Settings()
    return settings


def print_settings():
    """Print all current settings (for debugging).

    Masks sensitive values like API keys.

    Example:
        from mcp_server.config import print_settings
        print_settings()
    """
    print("\n" + "=" * 60)
    print("OmniMemory Configuration")
    print("=" * 60)

    # Group settings
    groups = {
        "Service URLs": [
            "embeddings_service_url",
            "compression_service_url",
            "procedural_service_url",
            "metrics_service_url",
            "dashboard_url",
            "gateway_url",
            "qdrant_url",
            "redis_url",
        ],
        "Database": ["database_url"],
        "Security": ["api_key_secret", "allowed_origins"],
        "Privacy": [
            "enable_path_anonymization",
            "path_anonymization_mode",
            "should_anonymize_paths",
        ],
        "Monitoring": [
            "log_level",
            "enable_telemetry",
            "otel_exporter_otlp_endpoint",
            "deployment_environment",
        ],
        "Rate Limiting": ["rate_limit_per_minute", "rate_limit_burst"],
        "Feature Flags": ["enable_dashboard", "enable_mcp_tools", "enable_agent_api"],
        "Performance": [
            "max_file_size_mb",
            "compression_timeout_seconds",
            "cache_ttl_seconds",
        ],
    }

    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            value = getattr(settings, key)

            # Mask sensitive values
            if (
                "secret" in key.lower()
                or "password" in key.lower()
                or "key" in key.lower()
            ):
                if isinstance(value, str) and len(value) > 8:
                    value = value[:4] + "****" + value[-4:]

            # Format database URL
            if "database_url" in key and isinstance(value, str):
                # Hide password in URL
                if "@" in value:
                    parts = value.split("@")
                    creds_parts = parts[0].split("//")
                    if len(creds_parts) > 1:
                        user_pass = creds_parts[1].split(":")
                        if len(user_pass) > 1:
                            value = f"{creds_parts[0]}//{user_pass[0]}:****@{parts[1]}"

            print(f"  {key}: {value}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_settings()
