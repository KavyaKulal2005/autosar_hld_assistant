"""Trained architecture-entity classifier (component / interface / port / signal / other).

Candidate identifiers are CamelCase tokens. Each mention is classified from its
surface shape (suffixes, hump count) and its sentence context (neighbouring words).
The token string itself is NOT used as a feature, so the model can label unseen names.
"""
import re
from pathlib import Path
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LABELS = ["component", "interface", "port", "signal", "other"]
CANDIDATE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b")
WORD = re.compile(r"[A-Za-z0-9_]+")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "entity_model.joblib"


def find_mentions(text):
    """Yield (token, sentence, start_offset_in_sentence) for each CamelCase candidate."""
    for line in text.splitlines():
        for m in CANDIDATE.finditer(line):
            yield m.group(0), line, m.start()


def mention_features(token, sentence, start):
    before = WORD.findall(sentence[:start])
    after = WORD.findall(sentence[start + len(token):])
    low = token.lower()
    feats = {
        "humps": len(re.findall(r"[A-Z]", token)),
        "len": min(len(token), 20) // 4,
        "first_in_sentence": int(not before),
    }
    for n in (2, 3, 4, 5, 6):
        feats[f"suf{n}={low[-n:]}"] = 1
    for n in (3, 4):
        feats[f"pre{n}={low[:n]}"] = 1
    for k in (1, 2, 3):
        if len(before) >= k:
            feats[f"prev{k}={before[-k].lower()}"] = 1
    for k in (1, 2):
        if len(after) >= k:
            feats[f"next{k}={after[k-1].lower()}"] = 1
    if before and after:
        feats[f"ctx={before[-1].lower()}|{after[0].lower()}"] = 1
    return feats


def build_dataset(docs):
    """docs: iterable of (text, labels_dict). Returns (X_feature_dicts, y, tokens)."""
    X, y, toks = [], [], []
    for text, labels in docs:
        for token, sentence, start in find_mentions(text):
            if token in labels:
                X.append(mention_features(token, sentence, start))
                y.append(labels[token])
                toks.append(token)
    return X, y, toks


def make_model():
    return make_pipeline(DictVectorizer(), LogisticRegression(C=5.0, max_iter=2000))


def train(docs, save=True):
    X, y, _ = build_dataset(docs)
    model = make_model().fit(X, y)
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict_entities(text, model):
    """Return {label: sorted unique tokens} for a document's text (majority vote per token)."""
    mentions = list(find_mentions(text))
    out = {k: [] for k in LABELS}
    if not mentions or model is None:
        return out
    probs = model.predict_proba([mention_features(*m) for m in mentions])
    classes = list(model.classes_)
    votes = {}
    for (token, _, _), p in zip(mentions, probs):
        acc = votes.setdefault(token, [0.0] * len(classes))
        for i, v in enumerate(p):
            acc[i] += v
    for token, acc in votes.items():
        out[classes[max(range(len(acc)), key=acc.__getitem__)]].append(token)
    return {k: sorted(v) for k, v in out.items()}
