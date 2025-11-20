"""Comprehensive integration tests for configuration system.

Tests the configuration management in mcp_server/config.py including:
- Environment variable loading
- Pydantic validators
- Computed properties
- Type safety
- Production security requirements
"""

import pytest
import os
from pydantic import ValidationError
from mcp_server.config import Settings, reload_settings, get_settings


class TestConfigEnvironmentLoading:
    """Test configuration loading from environment variables."""

    def test_default_values_when_no_env_vars(self, monkeypatch):
        """Test that defaults are loaded when no environment variables are set."""
        # Clear all OmniMemory-related env vars
        env_prefixes = [
            "METRICS_",
            "GATEWAY_",
            "EMBEDDINGS_",
            "COMPRESSION_",
            "PROCEDURAL_",
            "DASHBOARD_",
            "QDRANT_",
            "REDIS_",
            "DATABASE_",
            "API_KEY_",
            "ALLOWED_",
            "PROMETHEUS_",
            "LOG_",
            "ENABLE_",
            "OTEL_",
            "DEPLOYMENT_",
            "WEBSOCKET_",
            "RATE_LIMIT_",
            "MAX_",
            "CACHE_",
        ]

        for key in list(os.environ.keys()):
            if any(key.startswith(prefix) for prefix in env_prefixes):
                monkeypatch.delenv(key, raising=False)

        settings = Settings()

        # Verify service URLs
        assert settings.metrics_service_url == "http://localhost:8003"
        assert settings.embeddings_service_url == "http://localhost:8000"
        assert settings.compression_service_url == "http://localhost:8001"
        assert settings.gateway_url == "http://localhost:8009"
        assert settings.qdrant_url == "http://localhost:6333"

        # Verify rate limiting
        assert settings.rate_limit_per_minute == 100
        assert settings.rate_limit_burst == 20

        # Verify logging
        assert settings.log_level == "INFO"
        assert settings.deployment_environment == "development"

        # Verify feature flags
        assert settings.enable_dashboard is False
        assert settings.enable_mcp_tools is True

    def test_environment_variable_overrides_defaults(self, monkeypatch):
        """Test that environment variables override default values."""
        monkeypatch.setenv("METRICS_SERVICE_URL", "http://prod-metrics:9000")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "500")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENABLE_DASHBOARD", "true")

        settings = Settings()

        assert settings.metrics_service_url == "http://prod-metrics:9000"
        assert settings.rate_limit_per_minute == 500
        assert settings.log_level == "DEBUG"
        assert settings.enable_dashboard is True

    def test_multiple_env_vars_simultaneously(self, monkeypatch):
        """Test setting multiple environment variables at once."""
        monkeypatch.setenv("EMBEDDINGS_SERVICE_URL", "http://embed:8000")
        monkeypatch.setenv("COMPRESSION_SERVICE_URL", "http://compress:8001")
        monkeypatch.setenv("GATEWAY_URL", "http://gateway:8009")
        monkeypatch.setenv("PROMETHEUS_PORT", "9095")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "7200")

        settings = Settings()

        assert settings.embeddings_service_url == "http://embed:8000"
        assert settings.compression_service_url == "http://compress:8001"
        assert settings.gateway_url == "http://gateway:8009"
        assert settings.prometheus_port == 9095
        assert settings.cache_ttl_seconds == 7200

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case-insensitive."""
        # Pydantic Settings supports case-insensitive by default
        monkeypatch.setenv("metrics_service_url", "http://lowercase:8003")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "250")

        settings = Settings()

        assert settings.metrics_service_url == "http://lowercase:8003"
        assert settings.rate_limit_per_minute == 250

    def test_reload_settings_function(self, monkeypatch):
        """Test that reload_settings() picks up new environment variables."""
        # First load with default
        settings1 = Settings()
        original_url = settings1.metrics_service_url

        # Change environment
        monkeypatch.setenv("METRICS_SERVICE_URL", "http://reloaded:8003")

        # Reload settings
        settings2 = reload_settings()

        assert settings2.metrics_service_url == "http://reloaded:8003"
        assert settings2.metrics_service_url != original_url


class TestConfigValidators:
    """Test Pydantic validators for configuration."""

    def test_api_key_validation_production_rejects_default(self, monkeypatch):
        """Test that production rejects the default API key."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY_SECRET", "dev-secret-key-change-in-production")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "API_KEY_SECRET must be set to a secure value" in str(exc_info.value)

    def test_api_key_validation_production_rejects_short_key(self, monkeypatch):
        """Test that production rejects short API keys."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY_SECRET", "short_key_12345")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "must be at least 32 characters" in str(exc_info.value)

    def test_api_key_validation_production_accepts_long_key(self, monkeypatch):
        """Test that production accepts long secure API keys."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
        secure_key = "a" * 32  # 32 character key
        monkeypatch.setenv("API_KEY_SECRET", secure_key)

        settings = Settings()

        assert settings.api_key_secret == secure_key
        assert settings.is_production is True

    def test_log_level_validation_accepts_valid_levels(self, monkeypatch):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            monkeypatch.setenv("LOG_LEVEL", level)
            settings = Settings()
            assert settings.log_level == level

        # Test lowercase (should be converted to uppercase)
        monkeypatch.setenv("LOG_LEVEL", "debug")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_log_level_validation_rejects_invalid_level(self, monkeypatch):
        """Test that invalid log levels are rejected."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "LOG_LEVEL must be one of" in str(exc_info.value)

    def test_deployment_environment_validation_accepts_valid(self, monkeypatch):
        """Test that valid deployment environments are accepted."""
        valid_envs = ["development", "staging", "production"]

        for env in valid_envs:
            if env == "production":
                # Need secure API key for production
                monkeypatch.setenv("API_KEY_SECRET", "a" * 32)
            else:
                monkeypatch.delenv("API_KEY_SECRET", raising=False)

            monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", env)
            settings = Settings()
            assert settings.deployment_environment == env

    def test_deployment_environment_validation_rejects_invalid(self, monkeypatch):
        """Test that invalid deployment environments are rejected."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "invalid_env")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "DEPLOYMENT_ENVIRONMENT must be one of" in str(exc_info.value)

    def test_allowed_origins_csv_parsing(self, monkeypatch):
        """Test that allowed_origins are parsed from CSV string to list."""
        origins_csv = (
            "http://localhost:3000,http://localhost:8004,https://production.com"
        )
        monkeypatch.setenv("ALLOWED_ORIGINS", origins_csv)

        settings = Settings()

        assert isinstance(settings.allowed_origins, list)
        assert len(settings.allowed_origins) == 3
        assert "http://localhost:3000" in settings.allowed_origins
        assert "http://localhost:8004" in settings.allowed_origins
        assert "https://production.com" in settings.allowed_origins

    def test_allowed_origins_strips_whitespace(self, monkeypatch):
        """Test that allowed_origins strips whitespace from CSV."""
        origins_csv = (
            "http://localhost:3000 , http://localhost:8004  ,  https://production.com"
        )
        monkeypatch.setenv("ALLOWED_ORIGINS", origins_csv)

        settings = Settings()

        # All origins should be stripped of whitespace
        assert "http://localhost:3000" in settings.allowed_origins
        assert "http://localhost:8004" in settings.allowed_origins
        assert "https://production.com" in settings.allowed_origins
        # Should not contain whitespace
        assert all(" " not in origin for origin in settings.allowed_origins)


class TestComputedProperties:
    """Test computed properties of Settings."""

    def test_is_production_property(self, monkeypatch):
        """Test is_production property returns correct value."""
        # Test production
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")
        monkeypatch.setenv("API_KEY_SECRET", "a" * 32)
        settings = Settings()
        assert settings.is_production is True

        # Test non-production
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "development")
        settings = Settings()
        assert settings.is_production is False

    def test_is_development_property(self, monkeypatch):
        """Test is_development property returns correct value."""
        # Test development
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "development")
        settings = Settings()
        assert settings.is_development is True

        # Test non-development
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "staging")
        settings = Settings()
        assert settings.is_development is False

    def test_redis_host_extraction(self, monkeypatch):
        """Test redis_host property extracts host from redis_url."""
        # Test default
        settings = Settings()
        assert settings.redis_host == "localhost"

        # Test custom host
        monkeypatch.setenv("REDIS_URL", "redis://prod-redis:6379")
        settings = Settings()
        assert settings.redis_host == "prod-redis"

        # Test with different host
        monkeypatch.setenv("REDIS_URL", "redis://10.0.0.5:6379")
        settings = Settings()
        assert settings.redis_host == "10.0.0.5"

    def test_redis_port_extraction(self, monkeypatch):
        """Test redis_port property extracts port from redis_url."""
        # Test default
        settings = Settings()
        assert settings.redis_port == 6379

        # Test custom port
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6380")
        settings = Settings()
        assert settings.redis_port == 6380

        # Test different port
        monkeypatch.setenv("REDIS_URL", "redis://redis-server:7000")
        settings = Settings()
        assert settings.redis_port == 7000


class TestTypeSafety:
    """Test type validation for configuration fields."""

    def test_integer_fields_type_validation(self, monkeypatch):
        """Test that integer fields validate types correctly."""
        # Valid integer
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "500")
        settings = Settings()
        assert isinstance(settings.rate_limit_per_minute, int)
        assert settings.rate_limit_per_minute == 500

        # Invalid integer should raise error
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "not_a_number")
        with pytest.raises(ValidationError):
            Settings()

    def test_boolean_fields_type_validation(self, monkeypatch):
        """Test that boolean fields validate types correctly."""
        # Test various boolean representations
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("0", False),
        ]

        for value, expected in test_cases:
            monkeypatch.setenv("ENABLE_DASHBOARD", value)
            settings = Settings()
            assert isinstance(settings.enable_dashboard, bool)
            assert settings.enable_dashboard == expected

    def test_port_range_validation(self, monkeypatch):
        """Test that port numbers are validated within valid range."""
        # Valid port
        monkeypatch.setenv("PROMETHEUS_PORT", "9095")
        settings = Settings()
        assert settings.prometheus_port == 9095

        # Port too low (< 1024)
        monkeypatch.setenv("PROMETHEUS_PORT", "80")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "greater than or equal to 1024" in str(exc_info.value)

        # Port too high (> 65535)
        monkeypatch.setenv("PROMETHEUS_PORT", "70000")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "less than or equal to 65535" in str(exc_info.value)

    def test_timeout_range_validation(self, monkeypatch):
        """Test that timeout fields are validated within valid range."""
        # Valid timeout
        monkeypatch.setenv("COMPRESSION_TIMEOUT_SECONDS", "60")
        settings = Settings()
        assert settings.compression_timeout_seconds == 60

        # Timeout too low (< 5)
        monkeypatch.setenv("COMPRESSION_TIMEOUT_SECONDS", "2")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "greater than or equal to 5" in str(exc_info.value)

        # Timeout too high (> 300)
        monkeypatch.setenv("COMPRESSION_TIMEOUT_SECONDS", "500")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "less than or equal to 300" in str(exc_info.value)

    def test_file_size_range_validation(self, monkeypatch):
        """Test that max_file_size_mb is validated within valid range."""
        # Valid file size
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "50")
        settings = Settings()
        assert settings.max_file_size_mb == 50

        # File size too low (< 1)
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "0")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "greater than or equal to 1" in str(exc_info.value)

        # File size too high (> 100)
        monkeypatch.setenv("MAX_FILE_SIZE_MB", "200")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "less than or equal to 100" in str(exc_info.value)


class TestProductionSecurity:
    """Test production security requirements."""

    def test_production_environment_security_requirements(self, monkeypatch):
        """Test that production environment enforces security requirements."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "production")

        # Should reject default key
        monkeypatch.setenv("API_KEY_SECRET", "dev-secret-key-change-in-production")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "API_KEY_SECRET must be set to a secure value" in str(exc_info.value)

        # Should reject short key
        monkeypatch.setenv("API_KEY_SECRET", "short")
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "must be at least 32 characters" in str(exc_info.value)

        # Should accept long secure key
        monkeypatch.setenv("API_KEY_SECRET", "a" * 32)
        settings = Settings()
        assert settings.is_production is True

    def test_development_allows_default_key(self, monkeypatch):
        """Test that development environment allows default API key."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "development")
        monkeypatch.setenv("API_KEY_SECRET", "dev-secret-key-change-in-production")

        # Should not raise error in development
        settings = Settings()
        assert settings.api_key_secret == "dev-secret-key-change-in-production"
        assert settings.is_development is True

    def test_staging_security_requirements(self, monkeypatch):
        """Test that staging environment has appropriate security."""
        monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "staging")

        # Staging allows default key (not as strict as production)
        monkeypatch.setenv("API_KEY_SECRET", "dev-secret-key-change-in-production")
        settings = Settings()
        assert settings.deployment_environment == "staging"
        assert not settings.is_production
        assert not settings.is_development

    def test_get_settings_function(self):
        """Test get_settings() returns the global settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.metrics_service_url is not None


class TestConfigUtilityFunctions:
    """Test utility functions for configuration management."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings() returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert hasattr(settings, "metrics_service_url")
        assert hasattr(settings, "deployment_environment")

    def test_reload_settings_creates_new_instance(self, monkeypatch):
        """Test that reload_settings() creates a new Settings instance."""
        # Get initial settings
        settings1 = get_settings()
        initial_url = settings1.metrics_service_url

        # Change environment
        monkeypatch.setenv("METRICS_SERVICE_URL", "http://changed:8003")

        # Reload
        settings2 = reload_settings()

        # Should have new value
        assert settings2.metrics_service_url == "http://changed:8003"
        assert settings2.metrics_service_url != initial_url

    def test_websocket_settings_validation(self, monkeypatch):
        """Test WebSocket-related settings validation."""
        # Valid websocket settings
        monkeypatch.setenv("WEBSOCKET_MAX_CONNECTIONS", "2000")
        monkeypatch.setenv("WEBSOCKET_PING_INTERVAL", "60")

        settings = Settings()
        assert settings.websocket_max_connections == 2000
        assert settings.websocket_ping_interval == 60

        # Invalid: max_connections must be >= 1
        monkeypatch.setenv("WEBSOCKET_MAX_CONNECTIONS", "0")
        with pytest.raises(ValidationError):
            Settings()

        # Invalid: ping_interval must be >= 5
        monkeypatch.setenv("WEBSOCKET_MAX_CONNECTIONS", "1000")
        monkeypatch.setenv("WEBSOCKET_PING_INTERVAL", "2")
        with pytest.raises(ValidationError):
            Settings()

    def test_print_settings_function(self, capsys):
        """Test print_settings() utility function."""
        from mcp_server.config import print_settings

        # Call print_settings
        print_settings()

        # Capture output
        captured = capsys.readouterr()

        # Verify output contains expected sections
        assert "OmniMemory Configuration" in captured.out
        assert "Service URLs" in captured.out
        assert "Database" in captured.out
        assert "Security" in captured.out
        assert "Monitoring" in captured.out
        assert "Rate Limiting" in captured.out
        assert "Feature Flags" in captured.out
        assert "Performance" in captured.out

        # Verify sensitive values are masked
        assert "****" in captured.out  # API key should be masked

    def test_print_settings_masks_sensitive_data(self, monkeypatch, capsys):
        """Test that print_settings() masks sensitive information."""
        from mcp_server.config import print_settings

        # Set a long API key to test masking
        monkeypatch.setenv("API_KEY_SECRET", "test-secret-key-1234567890-very-long")
        reload_settings()

        # Call print_settings
        print_settings()

        # Capture output
        captured = capsys.readouterr()

        # Verify the full API key is NOT in the output
        assert "test-secret-key-1234567890-very-long" not in captured.out
        # Verify it's been masked
        assert "test****long" in captured.out or "****" in captured.out

    def test_allowed_origins_handles_empty_string(self, monkeypatch):
        """Test that allowed_origins handles empty strings in CSV."""
        monkeypatch.setenv(
            "ALLOWED_ORIGINS", "http://localhost:3000,,http://localhost:8004"
        )

        settings = Settings()

        # Empty string should be filtered out
        assert len(settings.allowed_origins) == 2
        assert "http://localhost:3000" in settings.allowed_origins
        assert "http://localhost:8004" in settings.allowed_origins
        assert "" not in settings.allowed_origins
