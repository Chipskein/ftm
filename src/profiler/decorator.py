import functools

from contextvars import ContextVar
from .context import current_monitor

def step(label_or_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = current_monitor.get()

            if monitor is None:
                return func(*args, **kwargs)

            old = getattr(monitor, "label", None)

            try:
                if callable(label_or_func):
                    label = label_or_func(*args, **kwargs)
                else:
                    label = label_or_func

                if label is not None:
                    monitor.set_label(label)

                return func(*args, **kwargs)

            finally:
                if old is not None:
                    monitor.set_label(old)

        return wrapper
    return decorator