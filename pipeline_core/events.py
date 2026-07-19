"""pipeline_core/events.py — Structured pipeline events (JSON Lines)."""

import os, json, time


class PipelineEvent:
    """A single structured event with monotonic ID."""

    _counter = 0

    def __init__(self, event: str, step: str = None, *, run_id: str = None,
                 current: int = None, total: int = None, unit: str = None,
                 name: str = None, value=None, message: str = None,
                 duration_seconds: float = None, details: str = None,
                 progress_mode: str = None, timestamp: float = None):
        PipelineEvent._counter += 1
        self.event_id = PipelineEvent._counter
        self.event = event
        self.step = step
        self.run_id = run_id
        self.timestamp = timestamp or time.time()
        # Progress
        self.current = current
        self.total = total
        self.unit = unit
        self.progress_mode = progress_mode
        # Metric
        self.name = name
        self.value = value
        # Message
        self.message = message
        self.details = details
        # Duration
        self.duration_seconds = duration_seconds

    def to_dict(self):
        d = {
            "event_id": self.event_id,
            "event": self.event,
            "timestamp": self.timestamp,
        }
        for k in ("step", "run_id", "current", "total", "unit", "progress_mode",
                  "name", "value", "message", "details", "duration_seconds"):
            v = getattr(self, k, None)
            if v is not None:
                d[k] = v
        return d

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self):
        return f"<PipelineEvent #{self.event_id} {self.event} step={self.step}>"


# Factory helpers
def step_started(step: str, **kw):
    return PipelineEvent("step_started", step=step, **kw)

def step_completed(step: str, duration_seconds: float = None, **kw):
    return PipelineEvent("step_completed", step=step, duration_seconds=duration_seconds, **kw)

def step_failed(step: str, message: str = None, details: str = None, **kw):
    return PipelineEvent("step_failed", step=step, message=message, details=details, **kw)

def progress(step: str, current: int, total: int, unit: str = "page", **kw):
    return PipelineEvent("progress", step=step, current=current, total=total, unit=unit, **kw)

def metric(step: str, name: str, value, **kw):
    return PipelineEvent("metric", step=step, name=name, value=value, **kw)

def log_msg(level: str, message: str, step: str = None, **kw):
    return PipelineEvent(level, step=step, message=message, **kw)

def warning(message: str, step: str = None, **kw):
    return PipelineEvent("warning", step=step, message=message, **kw)

def error(message: str, step: str = None, details: str = None, **kw):
    return PipelineEvent("error", step=step, message=message, details=details, **kw)


class EventLogger:
    """Writes structured events to a JSON Lines file + optionally prints human-readable."""

    def __init__(self, path: str = None, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self._file = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")

    def emit(self, ev: PipelineEvent):
        line = ev.to_json()
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        if self.verbose:
            self._print_human(ev)

    def _print_human(self, ev: PipelineEvent):
        ts = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
        parts = [f"[{ts}]"]
        if ev.step:
            parts.append(f"[{ev.step}]")
        if ev.event == "step_started":
            parts.append(f"Starting {ev.step}...")
        elif ev.event == "step_completed":
            dur = f" ({ev.duration_seconds:.1f}s)" if ev.duration_seconds else ""
            parts.append(f"Completed {ev.step}{dur}")
        elif ev.event == "step_failed":
            parts.append(f"Failed {ev.step}: {ev.message or ''}")
        elif ev.event == "progress":
            parts.append(f"{ev.step}: {ev.current}/{ev.total} {ev.unit or ''}")
        elif ev.event == "metric":
            parts.append(f"{ev.step}: {ev.name}={ev.value}")
        elif ev.event in ("warning", "error"):
            parts.append(f"[{ev.event.upper()}] {ev.message or ''}")
        else:
            parts.append(str(ev.message or ev.event))
        print(" ".join(parts))

    def close(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StructuredEventHandler:
    """Adapter that takes a raw log line and emits structured events when patterns match."""

    PAGE_PROGRESS_RE = r"(?:page|Page|trang)\s*(\d+)\s*(?:/|of|trên)\s*(\d+)"
    COMPLETED_RE = r"(?:Completed|✅|Hoàn tất|Done)"
    ERROR_RE = r"(?:Error|❌|Lỗi|failed|Failed)"
    WARNING_RE = r"(?:Warning|⚠️|Cảnh báo)"

    def __init__(self, logger: EventLogger, step: str = None):
        self.logger = logger
        self.step = step

    def handle(self, text: str, step: str = None):
        import re
        step = step or self.step
        if re.search(self.ERROR_RE, text):
            self.logger.emit(error(text, step=step))
        elif re.search(self.WARNING_RE, text):
            self.logger.emit(warning(text, step=step))
        elif re.search(self.PAGE_PROGRESS_RE, text):
            m = re.search(self.PAGE_PROGRESS_RE, text)
            if m:
                self.logger.emit(progress(step, current=int(m.group(1)), total=int(m.group(2))))
        elif re.search(self.COMPLETED_RE, text):
            self.logger.emit(log_msg("info", text, step=step))
        else:
            self.logger.emit(log_msg("info", text, step=step))
