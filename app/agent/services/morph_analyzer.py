"""형태소 분석 — Kiwi/MeCab 명사 추출."""
from app.core.config import get_settings
from app.core.logging import get_agent_logger
  
logger = get_agent_logger()


def _extract_nouns_kiwi(text: str) -> list[str]:
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        tokens = kiwi.tokenize(text)
        nouns = []
        for t in tokens:
            form = getattr(t, "form", t[0] if isinstance(t, (list, tuple)) else "")
            tag = getattr(t, "tag", t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else "")
            if tag in ("NNG", "NNP", "NNB", "NP", "NR") and form and len(str(form).strip()) > 0:
                nouns.append(str(form).strip())
        return list(dict.fromkeys(nouns))
    except Exception as e:
        logger.debug("Kiwi 명사 추출 스킵: %s", e)
        return []


def _extract_nouns_mecab(text: str) -> list[str]:
    try:
        from mecab import MeCab
        mecab = MeCab()
        nouns = mecab.nouns(text) or []
        return list(dict.fromkeys(nouns))
    except Exception as e:
        logger.debug("MeCab 명사 추출 스킵: %s", e)
        return []  


def extract_nouns(text: str) -> list[str]:
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
        nouns = [w.strip() for w in s.split() if len(w.strip()) >= 2]
        nouns = list(dict.fromkeys(nouns))
    logger.info("형태소 분석(명사) engine=%s nouns=%s", engine, nouns[:20])
    return nouns
