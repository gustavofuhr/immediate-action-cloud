# logger_setup.py
import logging, os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
base_logger = logging.getLogger("event_ai")
base_logger.setLevel(LOG_LEVEL)

# Only add handler if none exist (avoid duplicates in Lambda warm starts)
if not base_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(message)s"))
    base_logger.addHandler(_h)


class CtxAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs
