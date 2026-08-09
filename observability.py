import logging
import os

_log = logging.getLogger("rag")
_configured = False

def setup(app=None):
    """Wire Application Insights if configured, else console logs only."""
    global _configured
    if _configured:
        return
    _configured = True
    logging.basicConfig(level=logging.INFO)

    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not conn:
        _log.info("observability: app insights disabled, console logging only")
        return
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=conn)
        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        _log.info("observability: app insights enabled")
    except Exception as exc:
        _log.warning("observability: app insights setup failed: %s", type(exc).__name__)
