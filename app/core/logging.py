"""
공통 로깅 — 콘솔 + 일별 파일(service.yyyy-mm-dd.log).
"""
import logging
import sys
from datetime import date
from pathlib import Path

from app.core.config import get_settings


def _daily_file_path(log_dir: Path, base_name: str) -> Path:
    return log_dir / f"{base_name}.{date.today():%Y-%m-%d}.log"


class DailyFileHandler(logging.FileHandler):
    """날짜가 바뀌면 해당 날짜 파일로 전환하는 파일 핸들러."""

    def __init__(self, log_dir: Path, base_name: str, encoding: str = "utf-8"):
        self._log_dir = Path(log_dir)
        self._base_name = base_name
        self._current_date: date | None = None
        self._encoding = encoding
        path = _daily_file_path(self._log_dir, self._base_name)
        super().__init__(path, encoding=encoding)
        self._current_date = date.today()

    def emit(self, record: logging.LogRecord) -> None:
        today = date.today()
        if self._current_date != today:
            self.close()
            path = _daily_file_path(self._log_dir, self._base_name)
            self.baseFilename = str(path)
            self.stream = self._open()
            self._current_date = today
        super().emit(record)


def _setup_service_logger(
    name: str,
    *,
    file_base_name: str | None = None,
) -> logging.Logger:
    s = get_settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_dir = Path(s.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = file_base_name or name
    file_handler = DailyFileHandler(log_dir, base, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_agent_logger() -> logging.Logger:
    return _setup_service_logger("agent", file_base_name=get_settings().log_file_agent)


def get_manager_logger() -> logging.Logger:
    return _setup_service_logger("manager", file_base_name=get_settings().log_file_manager)


def get_exception_location(exc: BaseException) -> str:
    """예외 발생 코드 위치 (파일경로:라인)."""
    tb = getattr(exc, "__traceback__", None)
    while tb and getattr(tb, "tb_next", None):
        tb = tb.tb_next
    if tb:
        return f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
    return "unknown"
