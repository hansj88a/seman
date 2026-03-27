"""프롬프트 파일 로더 — PROMPTS_DIR 미설정 시 앱 내장 app/prompts 사용."""
from pathlib import Path

from app.core.config import get_settings


def _prompts_root() -> Path:
    s = get_settings()
    if s.prompts_dir and s.prompts_dir.strip():
        return Path(s.prompts_dir).resolve()
    return Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(service: str, name: str, *, fallback: str = "") -> str:
    """
    service/name.txt 로드. 없거나 읽기 실패 시 fallback 반환.
    예: load_prompt("agent", "query_reconstructor_system")
    """
    root = _prompts_root()
    path = root / service / f"{name}.txt"
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return fallback
