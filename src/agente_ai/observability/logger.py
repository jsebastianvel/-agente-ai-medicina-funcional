import contextvars
import functools
import json
import logging
import time
import uuid

from agente_ai import config

logger = logging.getLogger("agente_ai.trace")

_configured = False
_trace_var: contextvars.ContextVar[tuple[str, list[dict]] | None] = contextvars.ContextVar(
    "agente_ai_trace", default=None
)


def setup_logging() -> None:
    """Stdout structured JSON logging always works. Local file logging is
    best-effort: LOG_DIR defaults outside OneDrive-synced Documents (see
    CHROMA_DATA_DIR for why) but if it still fails for any reason, the app
    should keep running on stdout-only logging rather than crash."""
    global _configured
    if _configured:
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_DIR / "agent.jsonl", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not set up file logging at %s; stdout only.", config.LOG_DIR)

    _configured = True


def start_trace() -> tuple[str, list[dict]]:
    """Starts a fresh trace for one graph.invoke() call: call before invoking
    the graph, then read the returned list after invoke() completes to render
    a trace in the UI. Each @log_node-wrapped node appends its own record."""
    run_id = uuid.uuid4().hex[:8]
    trace: list[dict] = []
    _trace_var.set((run_id, trace))
    return run_id, trace


def _summarize(value, max_len: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def log_node(node_name: str):
    """Decorator for graph node functions: times execution, logs one
    structured JSON record per call (stdout + optional local file), and
    appends it to the current trace (if any) for the UI's debug view."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            setup_logging()
            ctx = _trace_var.get()
            run_id = ctx[0] if ctx else None
            start = time.perf_counter()
            status, error, result = "ok", None, None
            try:
                result = fn(state)
                return result
            except Exception as exc:
                status, error = "error", str(exc)
                raise
            finally:
                record = {
                    "run_id": run_id,
                    "node": node_name,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "output": _summarize(result) if status == "ok" else None,
                    "error": error,
                }
                logger.info(json.dumps(record, ensure_ascii=False))
                if ctx:
                    ctx[1].append(record)

        return wrapper

    return decorator
