# logging_setup.py
import logging, sys, json, time

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": round(time.time(), 3),         # unix timestamp
            "level": record.levelname,          # INFO / ERROR
            "msg": record.getMessage(),         # main message
            "logger": record.name,              # logger name
        }
        # Attach extras (if provided via logger.info(..., extra={"key": "value"}))
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "model_version"):
            payload["model_version"] = record.model_version
        return json.dumps(payload, ensure_ascii=False)

logger = logging.getLogger("rentapp")
logger.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.propagate = False
