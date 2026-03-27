"""텍스트 정규화 — Unicode, 노이즈 제거, (선택) 맞춤법·띄어쓰기."""
import re
import unicodedata

from app.core.logging import get_agent_logger

logger = get_agent_logger()

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
KEEP_PATTERN = re.compile(r"[^\w\s.\-,?!]", re.UNICODE)
MULTI_SPACE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def remove_noise(text: str) -> str:
    if not text:
        return ""
    s = EMOJI_PATTERN.sub("", str(text))
    s = KEEP_PATTERN.sub("", s)
    s = MULTI_SPACE.sub(" ", s)
    return s.strip()


def step1_normalize(query: str) -> str:
    """1단계: Unicode 정규화 + 노이즈 제거만 (검색 파이프라인용)."""
    if not query:
        return ""
    return remove_noise(normalize_unicode(query))
