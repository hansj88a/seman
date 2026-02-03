"""
agent 전용: 형태소 분석 — Kiwi/MeCab 명사 추출
"""
from app.config import get_settings
from app.logging_config import get_agent_logger

logger = get_agent_logger()


def _extract_nouns_kiwi(text: str) -> list[str]:
    """Kiwi(kiwipiepy)로 명사 추출."""
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        # tokenize 후 품사가 NNG, NNP, NNB 등 명사만 수집 (Kiwi 품사 태그: NNG 일반명사, NNP 고유명사 등)
        tokens = kiwi.tokenize(text)
        nouns = []
        for t in tokens:
            form = getattr(t, "form", t[0] if isinstance(t, (list, tuple)) else "")
            tag = getattr(t, "tag", t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else "")
            if tag in ("NNG", "NNP", "NNB", "NP", "NR") and form and len(str(form).strip()) > 0:
                nouns.append(str(form).strip())
        return list(dict.fromkeys(nouns))  # 순서 유지 중복 제거
    except Exception as e:
        logger.debug("Kiwi 명사 추출 스킵: %s", e)
        return []


def _extract_nouns_mecab(text: str) -> list[str]:
    """MeCab(python-mecab-ko)로 명사 추출."""
    try:
        from mecab import MeCab

        mecab = MeCab()
        nouns = mecab.nouns(text) or []
        return list(dict.fromkeys(nouns))
    except Exception as e:
        logger.debug("MeCab 명사 추출 스킵: %s", e)
        return []


def extract_nouns(text: str) -> list[str]:
    """
    2단계: 형태소 분석으로 명사 추출.
    .env MORPH_ENGINE=kiwi|mecab 에 따라 Kiwi 또는 MeCab 사용.
    """
    if not text or not str(text).strip():
        return []

    s = str(text).strip()
    settings = get_settings()
    engine = (settings.morph_engine or "kiwi").lower()

    if engine == "mecab":
        nouns = _extract_nouns_mecab(s)
    else:
        nouns = _extract_nouns_kiwi(s)

    if not nouns:
        # fallback: 공백으로 분리한 단어 중 2글자 이상
        nouns = [w.strip() for w in s.split() if len(w.strip()) >= 2]
        nouns = list(dict.fromkeys(nouns))

    logger.info("형태소 분석(명사 추출) engine=%s nouns=%s", engine, nouns[:20])
    return nouns
