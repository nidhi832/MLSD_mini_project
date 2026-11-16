"""
Distributed tracing with OpenTelemetry, exported to Jaeger over OTLP (D5).

One trace covers a whole run: the API submission span, the classifier agent span
in the classifier container, and the reasoning agent span (with one child span per
graph node and per tool call) in the agent-worker container. The trace context
travels between containers inside the queued task ("trace_context"), and every
span carries the run's correlation_id and run_id as attributes, so a run can be
found in Jaeger by either.

If OTEL_EXPORTER_OTLP_ENDPOINT is unset or Jaeger is down, spans are dropped
silently; tracing never blocks or fails a run.
"""

import os
from contextlib import contextmanager
from typing import Dict, Any, Optional

_initialised = False
_tracer = None


def init_tracing(service_name: Optional[str] = None):
    """Configure the global tracer provider once per process."""
    global _initialised, _tracer
    if _initialised:
        return _tracer
    _initialised = True
    service = service_name or os.getenv("OTEL_SERVICE_NAME", "claim-triage-service")
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        if endpoint and os.getenv("OTEL_SDK_DISABLED", "").lower() != "true":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("claim-triage")
    except Exception:
        _tracer = None
    return _tracer


def get_tracer():
    return _tracer if _initialised else init_tracing()


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None, context=None):
    """Start a span; a no-op when tracing is unavailable."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    attrs = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
             for k, v in (attributes or {}).items() if v is not None}
    with tracer.start_as_current_span(name, context=context, attributes=attrs) as s:
        yield s


def inject_context() -> Dict[str, str]:
    """Serialise the current trace context so it can travel inside a queued task."""
    carrier: Dict[str, str] = {}
    try:
        from opentelemetry.propagate import inject
        inject(carrier)
    except Exception:
        pass
    return carrier


def extract_context(carrier: Optional[Dict[str, str]]):
    """Rebuild a trace context received from another service."""
    if not carrier:
        return None
    try:
        from opentelemetry.propagate import extract
        return extract(carrier)
    except Exception:
        return None
