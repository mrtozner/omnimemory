"""OpenTelemetry instrumentation for OmniMemory Gateway.

This module sets up automatic instrumentation for FastAPI, providing:
- Distributed tracing (see request flow across services)
- Metrics (request count, latency, error rate)
- Logs (structured logging with trace context)

All telemetry data is sent to SigNoz via OTLP protocol.
"""

import os
import logging
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import (
    Resource,
    SERVICE_NAME,
    SERVICE_VERSION,
    DEPLOYMENT_ENVIRONMENT,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)


def setup_telemetry(
    app,
    service_name: str = "omnimemory-gateway",
    service_version: str = "2.0.0",
    environment: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
):
    """
    Set up OpenTelemetry instrumentation for FastAPI application.

    Args:
        app: FastAPI application instance
        service_name: Name of the service (shown in SigNoz)
        service_version: Version of the service
        environment: Deployment environment (development, staging, production)
        otlp_endpoint: OpenTelemetry collector endpoint (default: localhost:4317)

    Example:
        from fastapi import FastAPI
        from telemetry import setup_telemetry

        app = FastAPI()
        setup_telemetry(app)
    """

    # Get configuration from environment
    environment = environment or os.getenv("DEPLOYMENT_ENVIRONMENT", "development")
    otlp_endpoint = otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    # Check if telemetry is enabled
    telemetry_enabled = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"

    if not telemetry_enabled:
        logger.info("Telemetry is disabled via ENABLE_TELEMETRY environment variable")
        return

    logger.info(
        f"Setting up OpenTelemetry for {service_name} v{service_version} ({environment})"
    )

    # Create resource attributes
    resource = Resource(
        attributes={
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            DEPLOYMENT_ENVIRONMENT: environment,
            "service.namespace": "omnimemory",
        }
    )

    # Setup Tracing
    trace_provider = TracerProvider(resource=resource)

    # Create OTLP span exporter
    span_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True,  # Use insecure for local development
    )

    # Add span processor
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    # Set global trace provider
    trace.set_tracer_provider(trace_provider)

    # Setup Metrics
    metric_exporter = OTLPMetricExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )

    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=60000,  # Export every 60 seconds
    )

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )

    metrics.set_meter_provider(meter_provider)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumented with OpenTelemetry")

    # Instrument HTTPx (for outgoing HTTP requests)
    HTTPXClientInstrumentor().instrument()
    logger.info("HTTPx instrumented with OpenTelemetry")

    # Instrument logging (adds trace context to logs)
    LoggingInstrumentor().instrument(set_logging_format=True)
    logger.info("Logging instrumented with OpenTelemetry")

    logger.info(f"✅ OpenTelemetry setup complete. Sending telemetry to {otlp_endpoint}")
    logger.info(f"📊 View traces and metrics at: http://localhost:3301")


def get_tracer(name: str):
    """
    Get a tracer for manual instrumentation.

    Args:
        name: Name of the tracer (usually __name__)

    Example:
        from telemetry import get_tracer

        tracer = get_tracer(__name__)

        with tracer.start_as_current_span("compress_file"):
            # Your code here
            pass
    """
    return trace.get_tracer(name)


def get_meter(name: str):
    """
    Get a meter for manual metrics.

    Args:
        name: Name of the meter (usually __name__)

    Example:
        from telemetry import get_meter

        meter = get_meter(__name__)
        compression_counter = meter.create_counter(
            "omnimemory.compression.count",
            description="Number of compressions performed"
        )
        compression_counter.add(1)
    """
    return metrics.get_meter(name)
