from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def claim_hash(competitor: str, dimension: str, text: str) -> str:
    payload = f"{normalize_text(competitor)}|{normalize_text(dimension)}|{normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def quote_is_substring(quote: str, source_text: str) -> bool:
    """Validate that quote is a verbatim substring of source text.

    Allows whitespace normalization only — never invents content.
    """
    if not quote or not source_text:
        return False
    if quote in source_text:
        return True
    # Fallback: collapse whitespace on both sides and compare
    q = re.sub(r"\s+", " ", quote).strip()
    s = re.sub(r"\s+", " ", source_text)
    return q in s
