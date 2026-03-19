# backend/services/text_intel/normalize_text.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class NormalizeConfig:
    expand_abbreviations: bool = True
    normalize_numbers: bool = True


# ----------------------------
# 1) Whitespace cleanup
# ----------------------------
_whitespace_re = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    # Convert all whitespace (tabs/newlines/multiple spaces) -> single space
    text = _whitespace_re.sub(" ", text)
    return text.strip()


# ----------------------------
# 2) Basic abbreviations
# ----------------------------
# Keep small and safe (day-1 baseline)
_ABBR_MAP: Dict[str, str] = {
    r"\bDr\.\b": "Doctor",
    r"\bMr\.\b": "Mister",
    r"\bMrs\.\b": "Misses",
    r"\bMs\.\b": "Miss",
    r"\betc\.\b": "etcetera",
    r"\bvs\.\b": "versus",
    r"\bi\.e\.\b": "that is",
    r"\be\.g\.\b": "for example",
}


def expand_abbreviations(text: str) -> str:
    out = text
    for pattern, repl in _ABBR_MAP.items():
        out = re.sub(pattern, repl, out)
    return out


# ----------------------------
# 3) Simple number normalization
# ----------------------------
_UNITS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def _int_to_words(n: int) -> str:
    # Day-1: handle 0..999 only (safe + predictable)
    if n < 0:
        return "minus " + _int_to_words(-n)

    if n < 20:
        return _UNITS[n]

    if n < 100:
        tens = (n // 10) * 10
        rest = n % 10
        return _TENS[tens] if rest == 0 else f"{_TENS[tens]} {_UNITS[rest]}"

    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        return (
            f"{_UNITS[hundreds]} hundred"
            if rest == 0
            else f"{_UNITS[hundreds]} hundred {_int_to_words(rest)}"
        )

    # If bigger than 999, do not convert in v1 (avoid wrong reads)
    return str(n)


# patterns: $50, 10%, plain integers
_money_re = re.compile(r"(?<!\w)\$(\d{1,6})(?!\w)")
_percent_re = re.compile(r"(?<!\w)(\d{1,6})%(?!\w)")
_int_re = re.compile(r"(?<!\w)(\d{1,3})(?!\w)")  # 0..999 only


def normalize_numbers_simple(text: str) -> str:
    out = text

    # $50 -> fifty dollars
    def money_sub(m: re.Match) -> str:
        n = int(m.group(1))
        w = _int_to_words(n)
        return f"{w} dollars"

    out = _money_re.sub(money_sub, out)

    # 10% -> ten percent
    def percent_sub(m: re.Match) -> str:
        n = int(m.group(1))
        w = _int_to_words(n)
        return f"{w} percent"

    out = _percent_re.sub(percent_sub, out)

    # 21 -> twenty one (only 0..999)
    def int_sub(m: re.Match) -> str:
        n = int(m.group(1))
        return _int_to_words(n)

    out = _int_re.sub(int_sub, out)

    return out


# ----------------------------
# Main entry
# ----------------------------
def normalize_text(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> str:
    out = normalize_whitespace(text)

    if cfg.expand_abbreviations:
        out = expand_abbreviations(out)

    if cfg.normalize_numbers:
        out = normalize_numbers_simple(out)

    # Final whitespace pass (abbrev expansions can introduce doubles)
    out = normalize_whitespace(out)
    return out


# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    raw = "  Dr. Smith paid $50   for 10% off.\n\nThis   is  21  tests.  "
    print(normalize_text(raw))