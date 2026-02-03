"""
agent 전용: 사용자 질문 텍스트 정규화
1. Unicode 정규화 (unicodedata)
2. 노이즈 제거 (의미 없는 특수문자, 이모지, 중복 공백)
3. 맞춤법·띄어쓰기 교정 (한글: pykospacing + kiwipiepy, 영문: pyspellchecker)
"""
import re
import unicodedata
from app.logging_config import get_agent_logger

logger = get_agent_logger()

# ---- 1. Unicode 정규화 ----
def normalize_unicode(text: str) -> str:
    """
    Unicode 정규화 (NFKC: 호환 문자 정규화 + 조합).
    전각/반각 통일, 유사 문자 통합 등으로 검색 품질 향상.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text))


# ---- 2. 노이즈 제거 ----
# 이모지 및 기타 기호 블록 (의미 없는 문자 제거용)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # 이모티콘
    "\U0001F300-\U0001F5FF"  # 기호 & 픽토그램
    "\U0001F680-\U0001F6FF"  # 교통 & 맵
    "\U0001F1E0-\U0001F1FF"  # 국기
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # 보조 기호
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

# 의미 없는 특수문자 제거. 유지: 한글·영문·숫자·공백·구두점(.-,?!)
# \w(Unicode) = 글자·숫자·_, 이모지는 위에서 제거
KEEP_PATTERN = re.compile(r"[^\w\s.\-,?!]", re.UNICODE)
MULTI_SPACE = re.compile(r"\s+")


def remove_noise(text: str) -> str:
    """
    노이즈 제거: 이모지 제거, 의미 없는 특수문자 제거, 중복 공백을 하나로.
    한글/영문/숫자/공백/일부 구두점(.-,?!)만 유지.
    """
    if not text:
        return ""
    s = str(text)
    s = EMOJI_PATTERN.sub("", s)
    s = KEEP_PATTERN.sub("", s)
    s = MULTI_SPACE.sub(" ", s)
    return s.strip()


# ---- 3. 맞춤법·띄어쓰기 교정 (한글: pykospacing + kiwipiepy) ----
def _has_hangul(s: str) -> bool:
    return bool(s and re.search(r"[\u3131-\u318E\uAC00-\uD7A3\u1100-\u11FF]", s))


def _correct_korean(text: str) -> str:
    """한글 띄어쓰기·정규화: pykospacing(띄어쓰기) + kiwipiepy(형태소 분석 후 재결합)."""
    if not text or not text.strip():
        return text
    s = text.strip()
    # 1) pykospacing: 띄어쓰기 교정
    try:
        from pykospacing import Spacing

        spacing = Spacing()
        s = spacing(s)
    except Exception as e:
        logger.debug("pykospacing 띄어쓰기 스킵: %s", e)
    # 2) kiwipiepy: 형태소 분석 후 토큰 재결합 (일관된 띄어쓰기·정규화)
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        tokens = kiwi.tokenize(s)
        parts = []
        for t in tokens:
            form = getattr(t, "form", t[0] if isinstance(t, (list, tuple)) else "")
            if form and str(form).strip():
                parts.append(str(form).strip())
        if parts:
            s = " ".join(parts)
    except Exception as e:
        logger.debug("kiwipiepy 정규화 스킵: %s", e)
    return s.strip()


def _correct_english(text: str) -> str:
    """영문 맞춤법 교정 (pyspellchecker). 단어 단위로 교정."""
    try:
        from spellchecker import SpellChecker

        spell = SpellChecker()
        words = text.split()
        corrected = []
        for w in words:
            if w.isascii() and w.isalpha():
                cor = spell.correction(w)
                corrected.append(cor if cor is not None else w)
            else:
                corrected.append(w)
        return " ".join(corrected)
    except Exception as e:
        logger.debug("영문 맞춤법 교정 스킵: %s", e)
        return text


def correct_spelling(text: str) -> str:
    """
    맞춤법·띄어쓰기 교정.
    한글이 포함되면 한글 교정(띄어쓰기·맞춤법), 이어서 영문 단어만 영문 맞춤법 교정.
    """
    if not text or not text.strip():
        return text
    s = text.strip()
    if _has_hangul(s):
        s = _correct_korean(s)
    s = _correct_english(s)
    return s.strip()


# ---- 1단계 전용: Unicode + 특수문자 제거 (형태소 분석 전) ----
def step1_normalize(text: str) -> str:
    """
    1단계 텍스트 정규화: Unicode 정규화 + 특수문자·이모지·중복 공백 제거만.
    (맞춤법 교정 없음. 형태소 분석 전용.)
    """
    if not text:
        return ""
    s = str(text).strip()
    if not s:
        return s
    s = normalize_unicode(s)
    s = remove_noise(s)
    return s.strip() or str(text).strip()


# ---- 통합 파이프라인 ----
def normalize_query(text: str) -> str:
    """
    사용자 질문 텍스트 정규화 파이프라인.
    1. Unicode 정규화  2. 노이즈 제거  3. 맞춤법·띄어쓰기 교정
    """
    if not text:
        return ""
    s = str(text).strip()
    if not s:
        return s
    s = normalize_unicode(s)
    s = remove_noise(s)
    if not s:
        return s
    s = correct_spelling(s)
    return s.strip() or str(text).strip()
