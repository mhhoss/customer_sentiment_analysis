from pathlib import Path
import re
import pandas as pd
import spacy
import contractions
from typing import Iterable

from spacy.lang.en.stop_words import STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_us_Digital_Software_v1_00.tsv"


_NEGATIONS = {
    "not", "no", "never", "none", "cannot", "can't", "dont", "nothing", "neither", "nor"
}

_EXCEPTIONS = {
    "i", "you", "we", "it", "they"
}

_SPECIAL_TOKENS = {
    "tok_url", "tok_email", "tok_user", "tok_hashtag",
    "tok_num", "tok_num_1", "tok_num_2", "tok_num_3", "tok_num_4", "tok_num_5",
    "tok_excl", "tok_excl_multi", "tok_q", "tok_q_multi", "tok_qex",
    "tok_org"
}

_SCOPE_BREAKERS = {
    "tok_excl", "tok_excl_multi", "tok_q", "tok_q_multi", "tok_qex", "but", "however", "though"
}



_REGEX_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_REGEX_EMAIL = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b")
_REGEX_USER = re.compile(r"@[A-Za-z0-9_]+")
_REGEX_HASHTAG = re.compile(r"#([A-Za-z0-9_]+)")
_REGEX_HTML_TAG = re.compile(r"<[^>]+>")
_REGEX_CONTROL = re.compile(r"[\r\n\t]+")
_REGEX_MULTI_WS = re.compile(r"\s+")
_REGEX_REPEAT_CHAR = re.compile(r"([A-Za-z])\1{2,}")
_REGEX_DOTS = re.compile(r"\.{3,}")
_RE_INT = re.compile(r"^\d+$")



def load_raw_data(nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, sep="\t", encoding="utf-8", nrows=nrows)
    if "review_body" in df.columns:
        df["review_body"] = df["review_body"].fillna("")
    return df


def load_nlp():
    try:
        return spacy.load("en_core_web_md", disable=["ner", "textcat"])
    except OSError as os:
        print(f"The 'en_core_web_md' model is not installed!\nERROR: {os}")
        try:
            return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except Exception:
            return spacy.blank("en")
    except Exception as e:
        print(f"NLP failed to load!\nERROR: {e}")
        return spacy.blank("en")


def load_nlp_ner():
    try:
        return spacy.load("en_core_web_md", disable=["parser", "textcat"])
    except Exception as e:
        print(f"NLP(ner) failed to load!\nERROR: {e}")
        return None


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def normalize_text(text: str) -> str:
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)

    text = expand_contractions(text)

    text = _REGEX_HTML_TAG.sub(" ", text)
    text = _REGEX_URL.sub(" tok_url ", text)
    text = _REGEX_EMAIL.sub(" tok_email ", text)
    text = _REGEX_USER.sub(" tok_user ", text)
    text = _REGEX_HASHTAG.sub(r" tok_hashtag \1 ", text)

    text = _REGEX_CONTROL.sub(" ", text)
    text = _REGEX_DOTS.sub(" ", text)
    text = _REGEX_REPEAT_CHAR.sub(r"\1\1", text)
    text = _REGEX_CONTROL.sub(" ", text)

    text = re.sub(r"\?\!|\!\?", " tok_qex ", text)
    text = re.sub(r"\!{2,}", " tok_excl_multi ", text)
    text = re.sub(r"\?{2,}", " tok_q_multi ", text)
    text = re.sub(r"\!", " tok_excl ", text)
    text = re.sub(r"\?", " tok_q ", text)
    
    text = text.lower()
    text = _REGEX_MULTI_WS.sub(" ", text).strip()
    return text


def map_sentiment_number(token) -> str:
    raw = token.text.strip()
    if _RE_INT.match(raw):
        value = int(raw)
        if 1 <= value <= 5:
            return f"tok_num_{value}"
        return "tok_num"
    if token.like_num:
        return "tok_num"
    return ""


def normalize_token(token, use_lemma: bool = True, remove_stopwords: bool = False) -> str:
    raw = token.text.strip()
    if not raw:
        return ""

    if raw in _SPECIAL_TOKENS:
        return raw
    
    num_token = map_sentiment_number(token)
    if num_token:
        return num_token
    
    if token.is_punct or token.is_space:
        return ""

    normalized = token.lemma_.lower() if (use_lemma and token.lemma_ and token.lemma_ != "-PRON-") else token.lower_
    normalized = normalized.strip("`'\".,;:()[]{}<>|\\/")

    if not normalized:
        return ""
    if len(normalized) < 2 and normalized not in _EXCEPTIONS:
        return ""
    if remove_stopwords and normalized in STOP_WORDS and normalized not in _NEGATIONS:
        return ""

    return normalized


def apply_negation_scope(tokens: list[str], window: int = 3) -> list[str]:
    out: list[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        out.append(tok)

        if tok in _NEGATIONS:
            applied = 0
            j = i + 1
            while j < n and applied < window:
                nxt = tokens[j]
                if nxt in _SCOPE_BREAKERS:
                    break
                if nxt and not nxt.startswith("tok_") and nxt not in _NEGATIONS:
                    out.append(f"neg_{nxt}")
                    applied += 1
                j += 1
        i += 1
    return out


def mask_org_entities_batch(texts, nlp_ner, batch_size=2048):
    if nlp_ner is None:
        return list(texts)

    out = []
    docs = nlp_ner.pipe(texts, batch_size=batch_size)

    for text, doc in zip(texts, docs):
        if not doc.ents:
            out.append(text)
            continue

        parts, last = [], 0
        for ent in doc.ents:
            if ent.label_ != "ORG":
                continue
            parts.append(text[last:ent.start_char])
            parts.append(" tok_org ")
            last = ent.end_char
        parts.append(text[last:])
        out.append("".join(parts))

    return out


def preprocess_one(text: str, nlp, use_lemma: bool = True, remove_stopwords: bool = False, negation_window: int = 3) -> str:
    clean = normalize_text(text)
    if not clean:
        return ""
    doc = nlp(clean)
    tokens = [normalize_token(tok, use_lemma=use_lemma, remove_stopwords=remove_stopwords) for tok in doc]
    tokens = [t for t in tokens if t]
    tokens = apply_negation_scope(tokens, window=negation_window)
    return " ".join(tokens)


def preprocess_series(
    texts: Iterable[str],
    nlp,
    nlp_ner=None,
    batch_size: int = 1024,
    use_lemma: bool = True,
    remove_stopwords: bool = False,
    negation_window: int = 3,
) -> list[str]:
    
    raw_texts = [str(t) for t in texts]

    masked_texts = mask_org_entities_batch(raw_texts, nlp_ner, batch_size=batch_size)

    normalized_texts = (normalize_text(t) for t in masked_texts)

    docs = nlp.pipe(normalized_texts, batch_size=batch_size)

    out: list[str] = []
    for doc in docs:
        tokens = [normalize_token(tok, use_lemma=use_lemma, remove_stopwords=remove_stopwords) for tok in doc]
        tokens = [t for t in tokens if t]
        tokens = apply_negation_scope(tokens, window=negation_window)
        out.append(" ".join(tokens))
    return out


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_body",
    output_col: str = "normalized_review",
    use_lemma: bool = True,
    remove_stopwords: bool = True,
    negation_window: int = 3,
    batch_size: int = 2048,
    nlp=None,
    nlp_ner="auto"
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found.")
    
    if nlp is None:
        nlp = load_nlp()
    if nlp_ner == "auto":
        nlp_ner = load_nlp_ner()

    processed = preprocess_series(
        df[text_col].fillna("").astype(str).values,
        nlp=nlp,
        nlp_ner=nlp_ner,
        batch_size=batch_size,
        use_lemma=use_lemma,
        remove_stopwords=remove_stopwords,
        negation_window=negation_window,
    )
    out_df = df.copy()
    out_df[output_col] = processed
    return out_df
