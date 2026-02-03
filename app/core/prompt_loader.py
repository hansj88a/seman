"""
공통: 프롬프트 파일 로더 (서비스별·이름별 파일 분리, 프로덕션 환경 대응)
PROMPTS_DIR 미설정 시 앱 번들 prompts 사용, 설정 시 해당 경로 우선.
"""
from pathlib import Path

from app.config import get_settings


def _resolve_prompts_base() -> Path:
    """프롬프트 루트 경로: PROMPTS_DIR 설정 시 해당 경로, 아니면 app/prompts."""
    s = get_settings()
    if s.prompts_dir and str(s.prompts_dir).strip():
        return Path(s.prompts_dir).resolve()
    # 앱 패키지 내 prompts 디렉터리
    return Path(__file__).resolve().parent.parent / "prompts"


def get_prompt_path(service: str, name: str, ext: str = "txt") -> Path:
    """서비스·이름에 해당하는 프롬프트 파일 경로 반환."""
    base = _resolve_prompts_base()
    return base / service / f"{name}.{ext}"


def load_prompt(service: str, name: str, fallback: str = "") -> str:
    """
    프롬프트 파일 로드. 없으면 fallback 반환.
    service: "agent" | "manager", name: "query_classifier_system" 등
    """
    path = get_prompt_path(service, name)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return fallback.strip() if fallback else ""
