"""Text normalization for Kazakh profanity detection.

Nothing in this file is learned from data: the same fixed rules are applied to
train, validation, test and any new comment. The goal is to make different
spellings of the same word look the same to the model, e.g.

    "Қотааааақ", "котак", "Kotak", "qotaq"  ->  "котак"
    "Блять", "бля*", "Blyat"               ->  "блят", "бля*", "блят"
"""

import re
import unicodedata

# Kazakh Latin letters with diacritics (2018/2021 alphabets) -> Cyrillic.
_LATIN_SPECIAL = {
    "ş": "ш", "ç": "ч", "ğ": "г", "ǵ": "г", "ı": "и", "ń": "н", "ñ": "н",
    "á": "а", "ä": "а", "ó": "о", "ö": "о", "ú": "у", "ü": "у", "ý": "у",
    "é": "е",
}

# Informal Latin transliteration as people actually type it in Kazakhstan.
# Digraphs go first so that "sh" becomes "ш" and not "сх".
_LATIN_DIGRAPHS = [
    ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "е"),
]
_LATIN_SINGLE = str.maketrans({
    "a": "а", "b": "б", "c": "ч", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "ш", "x": "х", "y": "и", "z": "з",
})

# Fold Kazakh-specific letters (and a few Russian ones) to the letter people
# use when they type without a Kazakh keyboard: "қотақ" and "котак" match.
_CYRILLIC_FOLD = str.maketrans({
    "қ": "к", "ғ": "г", "ң": "н", "ә": "а", "ө": "о", "ұ": "у", "ү": "у",
    "һ": "х", "і": "и", "ы": "и", "й": "и", "ё": "е", "э": "е",
    "ь": "", "ъ": "",
})

_MASK_CHARS = re.compile(r"[*#$%]+")        # "с**ка", "б#ә" -> one "*"
_REPEATS = re.compile(r"(.)\1{2,}")         # "нахууууу" -> "наху"
_SPACES = re.compile(r"\s+")


def _strip_punctuation(text: str) -> str:
    """Replace punctuation with spaces, keep letters, digits, emoji and '*'."""
    out = []
    for ch in text:
        if ch == "*":
            out.append(ch)
        elif unicodedata.category(ch).startswith("P"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def basic_clean(text: str) -> str:
    """Minimal cleaning used when normalization is switched off."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _SPACES.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Full normalization: script, spelling, masking and elongation."""
    text = basic_clean(text)
    text = "".join(_LATIN_SPECIAL.get(ch, ch) for ch in text)
    for latin, cyr in _LATIN_DIGRAPHS:
        text = text.replace(latin, cyr)
    text = text.translate(_LATIN_SINGLE)
    text = text.translate(_CYRILLIC_FOLD)
    text = _MASK_CHARS.sub("*", text)
    text = _strip_punctuation(text)
    text = _REPEATS.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Key used to find exact duplicates (case, punctuation, elongation ignored)."""
    return normalize(text)


# Picklable callables used inside the sklearn pipeline.
def normalize_batch(texts):
    return [normalize(t) for t in texts]


def basic_clean_batch(texts):
    return [basic_clean(t) for t in texts]


if __name__ == "__main__":
    for s in ["Қотааааақ!!!", "Kotakkkkkk", "qotaq", "Блять", "с**ка", "ОЙЫН нахууууу",
              "shesheńdi mynaý", "Pizdez oyin", "Маған қатты ұнады 👍👍👍"]:
        print(f"{s!r:30} -> {normalize(s)!r}")
