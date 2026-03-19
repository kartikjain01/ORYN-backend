from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GrammarConfig:
    enabled: bool = True
    # If you later install language_tool_python, you can enable it.
    use_language_tool_if_available: bool = False


def basic_grammar_cleanup(text: str) -> str:
    """
    Day-1 safe cleanup (does NOT rewrite meaning):
      - fix spacing before punctuation
      - collapse repeated punctuation
      - ensure space after punctuation
    """
    s = text or ""

    # Remove space before punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    # Ensure one space after punctuation when followed by a letter/number
    s = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", s)

    # Collapse repeated punctuation like "!!!" -> "!"
    s = re.sub(r"([!?.,])\1{1,}", r"\1", s)

    # Cleanup double spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def grammar_correct(text: str, cfg: GrammarConfig = GrammarConfig()) -> str:
    if not cfg.enabled:
        return text or ""

    s = basic_grammar_cleanup(text)

    # Optional: LanguageTool (not required)
    if cfg.use_language_tool_if_available:
        try:
            import language_tool_python  # type: ignore
            tool = language_tool_python.LanguageToolPublicAPI("en-US")
            s = tool.correct(s)
        except Exception:
            pass

    return s