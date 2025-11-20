"""Comprehensive tests for health check endpoints.

Tests cover:
- /health endpoint: Basic health check
- /ready endpoint: Readiness check with dependency verification
- /metrics endpoint: Metrics reporting
"""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import httpx


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(app):
    """Test /health endpoint returns 200 status code."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_correct_structure(app):
    """Test /health endpoint returns correct JSON structure."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        data = response.json()

        # Verify all required fields are present
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_endpoint_values(app):
    """Test /health endpoint returns correct values."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        data = response.json()

        # Verify values
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

        # Verify timestamp is valid ISO format
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert isinstance(timestamp, datetime)


@pytest.mark.asyncio
async def test_health_endpoint_response_time(app):
    """Test /health endpoint responds quickly (< 100ms)."""
    import time

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_time = time.time()
        response = await client.get("/health")
        end_time = time.time()

        assert response.status_code == 200
        response_time_ms = (end_time - start_time) * 1000
        assert (
            response_time_ms < 100
        ), f"Response time {response_time_ms}ms exceeds 100ms"


# /ready endpoint tests


@pytest.mark.asyncio
async def test_ready_all_services_healthy(app):
    """Test /ready endpoint when all dependencies are available."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Mock successful responses for both services
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.elapsed = MagicMock()
        mock_response.elapsed.total_seconds.return_value = 0.05

        mock_client.get.return_value = mock_response

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True
            assert "dependencies" in data
            assert data["dependencies"]["metrics_service"]["status"] == "ok"
            assert data["dependencies"]["qdrant"]["status"] == "ok"
            assert "timestamp" in data
            assert "errors" not in data or len(data["errors"]) == 0


@pytest.mark.asyncio
async def test_ready_metrics_service_down(app):
    """Test /ready endpoint when metrics service is down."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call (metrics service) fails, second call (qdrant) succeeds
        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 503
        mock_response_fail.elapsed = MagicMock()
        mock_response_fail.elapsed.total_seconds.return_value = 0.05

        mock_response_ok = AsyncMock()
        mock_response_ok.status_code = 200
        mock_response_ok.elapsed = MagicMock()
        mock_response_ok.elapsed.total_seconds.return_value = 0.05

        mock_client.get.side_effect = [mock_response_fail, mock_response_ok]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["metrics_service"]["status"] == "error"
            assert data["dependencies"]["qdrant"]["status"] == "ok"
            assert "errors" in data
            assert any("Metrics service" in err for err in data["errors"])


@pytest.mark.asyncio
async def test_ready_qdrant_down(app):
    """Test /ready endpoint when Qdrant is down."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call (metrics service) succeeds, second call (qdrant) fails
        mock_response_ok = AsyncMock()
        mock_response_ok.status_code = 200
        mock_response_ok.elapsed = MagicMock()
        mock_response_ok.elapsed.total_seconds.return_value = 0.05

        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 503
        mock_response_fail.elapsed = MagicMock()
        mock_response_fail.elapsed.total_seconds.return_value = 0.05

        mock_client.get.side_effect = [mock_response_ok, mock_response_fail]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["metrics_service"]["status"] == "ok"
            assert data["dependencies"]["qdrant"]["status"] == "error"
            assert "errors" in data
            assert any("Qdrant" in err for err in data["errors"])


@pytest.mark.asyncio
async def test_ready_both_services_down(app):
    """Test /ready endpoint when both services are down."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Both calls fail
        mock_response_fail = AsyncMock()
        mock_response_fail.status_code = 503
        mock_response_fail.elapsed = MagicMock()
        mock_response_fail.elapsed.total_seconds.return_value = 0.05

        mock_client.get.return_value = mock_response_fail

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["metrics_service"]["status"] == "error"
            assert data["dependencies"]["qdrant"]["status"] == "error"
            assert "errors" in data
            assert len(data["errors"]) >= 2


@pytest.mark.asyncio
async def test_ready_timeout_error(app):
    """Test /ready endpoint handles timeout errors correctly."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call times out, second call succeeds
        mock_client.get.side_effect = [
            httpx.TimeoutException("Connection timeout"),
            AsyncMock(status_code=200, elapsed=MagicMock(total_seconds=lambda: 0.05)),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["metrics_service"]["status"] == "error"
            assert data["dependencies"]["metrics_service"]["error"] == "timeout"
            assert "errors" in data
            assert any("timeout" in err.lower() for err in data["errors"])


@pytest.mark.asyncio
async def test_ready_connection_error(app):
    """Test /ready endpoint handles connection errors correctly."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call raises connection error, second call succeeds
        mock_client.get.side_effect = [
            Exception("Connection refused"),
            AsyncMock(status_code=200, elapsed=MagicMock(total_seconds=lambda: 0.05)),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["metrics_service"]["status"] == "error"
            assert "error" in data["dependencies"]["metrics_service"]
            assert (
                "Connection refused" in data["dependencies"]["metrics_service"]["error"]
            )


@pytest.mark.asyncio
async def test_ready_response_structure(app):
    """Test /ready endpoint returns correct response structure."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.elapsed = MagicMock()
        mock_response.elapsed.total_seconds.return_value = 0.05
        mock_client.get.return_value = mock_response

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")
            data = response.json()

            # Verify required fields
            assert "ready" in data
            assert "dependencies" in data
            assert "timestamp" in data

            # Verify dependencies structure
            assert "metrics_service" in data["dependencies"]
            assert "qdrant" in data["dependencies"]

            # Each dependency should have status and url
            for dep_name, dep_data in data["dependencies"].items():
                assert "status" in dep_data
                assert "url" in dep_data

                # If status is ok, should have response_time_ms
                if dep_data["status"] == "ok":
                    assert "response_time_ms" in dep_data
                    assert isinstance(dep_data["response_time_ms"], int)


@pytest.mark.asyncio
async def test_ready_timestamp_format(app):
    """Test /ready endpoint timestamp is in correct ISO format."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.elapsed = MagicMock()
        mock_response.elapsed.total_seconds.return_value = 0.05
        mock_client.get.return_value = mock_response

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")
            data = response.json()

            # Verify timestamp is valid ISO format
            timestamp = datetime.fromisoformat(data["timestamp"])
            assert isinstance(timestamp, datetime)


@pytest.mark.asyncio
async def test_ready_qdrant_timeout_error(app):
    """Test /ready endpoint handles Qdrant timeout errors correctly."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call (metrics service) succeeds, second call (qdrant) times out
        mock_client.get.side_effect = [
            AsyncMock(status_code=200, elapsed=MagicMock(total_seconds=lambda: 0.05)),
            httpx.TimeoutException("Connection timeout"),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["qdrant"]["status"] == "error"
            assert data["dependencies"]["qdrant"]["error"] == "timeout"
            assert "errors" in data
            assert any("Qdrant timeout" in err for err in data["errors"])


@pytest.mark.asyncio
async def test_ready_qdrant_connection_error(app):
    """Test /ready endpoint handles Qdrant connection errors correctly."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # First call (metrics service) succeeds, second call (qdrant) raises connection error
        mock_client.get.side_effect = [
            AsyncMock(status_code=200, elapsed=MagicMock(total_seconds=lambda: 0.05)),
            Exception("Connection refused"),
        ]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is False
            assert data["dependencies"]["qdrant"]["status"] == "error"
            assert "error" in data["dependencies"]["qdrant"]
            assert "Connection refused" in data["dependencies"]["qdrant"]["error"]
            assert "errors" in data
            assert any("Qdrant error" in err for err in data["errors"])


# /metrics endpoint tests


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200(app):
    """Test /metrics endpoint returns 200 status code."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_json_format(app):
    """Test /metrics endpoint returns JSON format (not Prometheus format)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")

        # Should be able to parse as JSON
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_metrics_endpoint_contains_expected_fields(app):
    """Test /metrics endpoint contains expected metric fields."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")
        data = response.json()

        # Verify all expected fields are present
        assert "omnimemory_uptime_seconds" in data
        assert "omnimemory_version" in data
        assert "omnimemory_start_time" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_metrics_endpoint_correct_types(app):
    """Test /metrics endpoint returns correct data types."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")
        data = response.json()

        # Verify types
        assert isinstance(data["omnimemory_uptime_seconds"], int)
        assert isinstance(data["omnimemory_version"], str)
        assert isinstance(data["omnimemory_start_time"], (int, float))
        assert isinstance(data["timestamp"], str)

        # Verify values
        assert data["omnimemory_version"] == "2.0.0"
        assert data["omnimemory_uptime_seconds"] >= 0

        # Verify timestamp is valid ISO format
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert isinstance(timestamp, datetime)


@pytest.mark.asyncio
async def test_metrics_endpoint_uptime_increases(app):
    """Test /metrics endpoint uptime increases over time."""
    import asyncio

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # First call
        response1 = await client.get("/metrics")
        data1 = response1.json()
        uptime1 = data1["omnimemory_uptime_seconds"]

        # Wait a bit
        await asyncio.sleep(1)

        # Second call
        response2 = await client.get("/metrics")
        data2 = response2.json()
        uptime2 = data2["omnimemory_uptime_seconds"]

        # Uptime should have increased
        assert uptime2 >= uptime1
