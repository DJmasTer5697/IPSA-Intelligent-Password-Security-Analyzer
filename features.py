"""
IPSA feature extraction module.
No plaintext password is persisted by this module.
"""

import math
import re
import string

COMMON_WORDS = {
    "password","pass","admin","administrator","welcome","qwerty","letmein",
    "login","user","root","guest","hello","love","football","dragon",
    "monkey","master","superman","princess","sunshine","iloveyou",
    "india","india123","computer","internet","google","facebook","twitter",
    "abc","abcd","test","testing","secret","security","access","default",
    "changeme","password1","password123"
}

KEYBOARD_ROWS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "1234567890", "qwertyuiop[]", "asdfghjkl;'",
    "zxcvbnm,./"
]

LEET_MAP = str.maketrans({
    "@":"a", "4":"a", "3":"e", "1":"i", "!":"i", "0":"o",
    "$":"s", "5":"s", "7":"t", "+":"t"
})

def _has_sequence(s: str, min_run: int = 3) -> int:
    s = s.lower()
    if len(s) < min_run:
        return 0
    run = 1
    for i in range(1, len(s)):
        if ord(s[i]) - ord(s[i-1]) in (1, -1):
            run += 1
            if run >= min_run:
                return 1
        else:
            run = 1
    return 0

def _has_keyboard_pattern(s: str, min_run: int = 3) -> int:
    s = s.lower()
    for row in KEYBOARD_ROWS:
        for n in range(len(row) - min_run + 1):
            part = row[n:n+min_run]
            if part in s or part[::-1] in s:
                return 1
    return 0

def _has_repetition(s: str) -> int:
    return int(bool(re.search(r"(.)\1{2,}", s)))

def _dictionary_hit(s: str, words=None) -> int:
    words = words or COMMON_WORDS
    normalized = s.lower().translate(LEET_MAP)
    for w in words:
        if len(w) >= 3 and w in normalized:
            return 1
    return 0

def extract_features(password: str, dictionary=None) -> dict:
    p = str(password)
    n = len(p)

    upper = sum(c.isupper() for c in p)
    lower = sum(c.islower() for c in p)
    digits = sum(c.isdigit() for c in p)
    special = sum(c in string.punctuation for c in p)
    unique = len(set(p))
    diversity = unique / n if n else 0.0

    pool = 0
    if upper: pool += 26
    if lower: pool += 26
    if digits: pool += 10
    if special: pool += len(string.punctuation)
    entropy = n * math.log2(pool) if n and pool else 0.0

    repeated = _has_repetition(p)
    sequential = _has_sequence(p)
    keyboard = _has_keyboard_pattern(p)
    dictionary = _dictionary_hit(p, dictionary)

    # A simple structural indicator: number of transitions between character classes.
    def cls(c):
        if c.isupper(): return "U"
        if c.islower(): return "L"
        if c.isdigit(): return "D"
        if c in string.punctuation: return "S"
        return "O"

    transitions = sum(cls(p[i]) != cls(p[i-1]) for i in range(1, n)) if n > 1 else 0
    repetition_ratio = 1 - diversity

    return {
        "length": n,
        "uppercase": upper,
        "lowercase": lower,
        "digits": digits,
        "special": special,
        "unique_chars": unique,
        "character_diversity": diversity,
        "entropy": entropy,
        "repetition_ratio": repetition_ratio,
        "repeated_pattern": repeated,
        "sequential_pattern": sequential,
        "dictionary_pattern": dictionary,
        "keyboard_pattern": keyboard,
        "structural_transitions": transitions,
    }

FEATURE_COLUMNS = list(extract_features("Abc123!"))
