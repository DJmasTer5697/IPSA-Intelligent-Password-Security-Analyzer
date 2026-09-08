
import math, re
from collections import Counter

COMMON_WORDS = {
    "password","admin","welcome","qwerty","letmein","login","user","guest",
    "dragon","football","monkey","master","hello","iloveyou","princess",
    "sunshine","computer","secret","superman","shadow","access"
}
KEYBOARD_ROWS = ["qwertyuiop","asdfghjkl","zxcvbnm","1234567890"]

def _norm(s):
    s = str(s).lower()
    return (s.replace("@","a").replace("4","a").replace("3","e")
             .replace("1","i").replace("!","i").replace("0","o")
             .replace("$","s").replace("5","s").replace("7","t"))

def _seq(s):
    if len(s) < 3: return 0
    n=0
    for i in range(len(s)-2):
        a,b,c=map(ord,s[i:i+3])
        if (b==a+1 and c==b+1) or (b==a-1 and c==b-1):
            n+=1
    return n

def _keyboard(s):
    s=str(s).lower()
    hits=0
    for row in KEYBOARD_ROWS:
        for i in range(len(row)-2):
            tri=row[i:i+3]
            if tri in s or tri[::-1] in s:
                hits+=1
    return hits

def extract_features(password):
    p=str(password)
    L=len(p)
    cnt=Counter(p)
    unique=len(set(p))
    pool=0
    pool += 26 if any(c.islower() for c in p) else 0
    pool += 26 if any(c.isupper() for c in p) else 0
    pool += 10 if any(c.isdigit() for c in p) else 0
    pool += 32 if any(not c.isalnum() for c in p) else 0
    entropy = L*math.log2(pool) if L and pool else 0.0
    norm=_norm(p)
    return {
        "length":L,
        "uppercase":sum(c.isupper() for c in p),
        "lowercase":sum(c.islower() for c in p),
        "digits":sum(c.isdigit() for c in p),
        "special":sum(not c.isalnum() for c in p),
        "unique_chars":unique,
        "character_diversity":unique/L if L else 0,
        "entropy":entropy,
        "repetition_ratio":1-(unique/L if L else 0),
        "repeated_pattern":1 if re.search(r"(.)\1{2,}",p) else 0,
        "sequential_pattern":_seq(p),
        "keyboard_pattern":_keyboard(p),
        "dictionary_pattern":1 if any(w in norm for w in COMMON_WORDS) else 0,
        "structural_transitions":sum(
            1 for a,b in zip(p,p[1:])
            if (a.isalpha()!=b.isalpha()) or (a.isdigit()!=b.isdigit()) or
               ((not a.isalnum())!=(not b.isalnum()))
        )
    }
