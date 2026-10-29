"""NLP chatbot: preprocessing (NLTK), intent recognition (scikit-learn) and response generation."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline

try:  # NLTK's tokenizer and stemmer are pure Python and need no corpus downloads.
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import RegexpTokenizer

    _TOKENIZER = RegexpTokenizer(r"[a-z0-9]+(?:'[a-z]+)?")
    _STEMMER = SnowballStemmer("english")
    NLTK_AVAILABLE = True
except ImportError:  # keeps the bot usable (with a simpler stemmer) if NLTK is not installed
    _TOKENIZER = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
    _STEMMER = None
    NLTK_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "intents.json"
KB_PATH = ROOT / "data" / "knowledge_base.txt"
MODEL_PATH = ROOT / "models" / "intent_model.joblib"

# Deliberately short: question words and pronouns ("how", "what", "you") carry the intent of chatbot queries.
STOP_WORDS = frozenset("a an the of to in on for and or is am be been please just really so with at by".split())


def _tokenize(text: str) -> list[str]:
    return _TOKENIZER.tokenize(text) if NLTK_AVAILABLE else _TOKENIZER.findall(text)


def _stem(word: str) -> str:
    if _STEMMER is not None:
        return _STEMMER.stem(word)
    for suffix in ("ing", "ed", "es", "s"):  # crude fallback stemmer
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def normalize(text: str) -> str:
    """Lowercase, strip URLs/punctuation, collapse whitespace (no stemming; used for char n-grams)."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text).lower())
    return " ".join(_tokenize(text.replace("’", "'")))


def preprocess(text: str) -> str:
    """Full pipeline: normalize -> tokenize -> remove stop words -> stem."""
    tokens = normalize(text).split()
    kept = [t for t in tokens if t not in STOP_WORDS] or tokens  # never return an empty query
    return " ".join(_stem(t) for t in kept)


def prepare(text: str) -> str:
    """Model input: stemmed tokens, a separator, then the raw normalized text (feeds both vectorizers)."""
    return f"{preprocess(text)} ||| {normalize(text)}"


def _words(doc: str) -> str:
    return doc.split(" ||| ")[0]


def _chars(doc: str) -> str:
    return doc.split(" ||| ")[-1]


def build_model(C: float = 20.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=_words, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=_chars, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=42)),
    ])


def load_intents(path: Path = DATA_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def training_data(intents: dict[str, Any]) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for item in intents["intents"]:
        for pattern in item["patterns"]:
            texts.append(pattern)
            labels.append(item["tag"])
    return texts, labels


def _signature(intents_path: Path) -> str:
    digest = hashlib.sha256(Path(intents_path).read_bytes())
    digest.update(str(NLTK_AVAILABLE).encode())
    return digest.hexdigest()


class Chatbot:
    def __init__(self, intents_path: Path = DATA_PATH, kb_path: Path = KB_PATH,
                 model_path: Path | None = MODEL_PATH, seed: int | None = None):
        self.intents_path = Path(intents_path)
        self.intents = load_intents(self.intents_path)
        self.intent_map = {i["tag"]: i for i in self.intents["intents"]}
        self.threshold = float(self.intents.get("confidence_threshold", 0.3))
        self.rng = random.Random(seed)
        self._last: dict[str, str] = {}
        self.model = self._load_or_train(model_path)
        self.kb = self._load_kb(Path(kb_path))

    # ---- training -------------------------------------------------
    def _load_or_train(self, model_path: Path | None) -> Pipeline:
        sig = _signature(self.intents_path)
        if model_path and Path(model_path).exists():
            try:
                saved = joblib.load(model_path)
                if saved.get("signature") == sig:
                    return saved["model"]
            except Exception:  # corrupt or incompatible file: retrain below
                pass
        model = build_model()
        texts, labels = training_data(self.intents)
        model.fit([prepare(t) for t in texts], labels)
        if model_path:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "signature": sig}, model_path)
        return model

    # ---- knowledge base ------------------------------------------
    def _load_kb(self, path: Path):
        if not path.exists():
            return None
        chunks = [b.strip() for b in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if b.strip()]
        titles = [c.splitlines()[0].lstrip("# ").strip() for c in chunks]
        bodies = [" ".join(c.splitlines()[1:]).strip() or c for c in chunks]
        vec = TfidfVectorizer(preprocessor=_words, ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([prepare(f"{t} {b}") for t, b in zip(titles, bodies)])
        return {"titles": titles, "bodies": bodies, "vec": vec, "matrix": matrix}

    def retrieve(self, message: str, min_score: float = 0.22):
        if not self.kb:
            return None
        scores = cosine_similarity(self.kb["vec"].transform([prepare(message)]), self.kb["matrix"])[0]
        best = int(np.argmax(scores))
        if scores[best] < min_score:
            return None
        return self.kb["titles"][best], self.kb["bodies"][best], float(scores[best])

    # ---- prediction ----------------------------------------------
    def predict(self, message: str) -> tuple[str, float]:
        proba = self.model.predict_proba([prepare(message)])[0]
        idx = int(np.argmax(proba))
        return str(self.model.classes_[idx]), float(proba[idx])

    def _pick(self, key: str, options: list[str]) -> str:
        choices = [o for o in options if o != self._last.get(key)] or options
        choice = self.rng.choice(choices)
        self._last[key] = choice
        return choice

    def respond(self, message: Any) -> dict[str, Any]:
        text = message if isinstance(message, str) else ""
        text = text.strip()[:500]
        if not normalize(text):
            return {"intent": "empty_input", "confidence": 1.0, "source": None,
                    "response": self.intents.get("empty_response", "Please type a message.")}
        intent, confidence = self.predict(text)
        if confidence >= self.threshold:
            return {"intent": intent, "confidence": confidence, "source": None,
                    "response": self._pick(intent, self.intent_map[intent]["responses"])}
        hit = self.retrieve(text)
        if hit:
            title, body, score = hit
            return {"intent": "knowledge_base", "confidence": confidence, "source": title, "response": body}
        return {"intent": "fallback", "confidence": confidence, "source": None,
                "response": self._pick("fallback", self.intents["fallback_responses"])}
