from contextvars import ContextVar

current_monitor: ContextVar = ContextVar("current_monitor", default=None)