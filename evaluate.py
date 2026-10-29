"""Evaluate the chatbot's intent classifier honestly and compare the improved pipeline with a baseline.

1. Stratified 5-fold cross-validation on data/intents.json
2. Held-out test on data/test_queries.json (separately written paraphrases, never trained on)
3. Baseline vs improved pipeline, per-intent report, confusion matrix, misclassified phrases
Results are written to reports/evaluation.md and reports/metrics.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from chatbot import ROOT, NLTK_AVAILABLE, build_model, load_intents, normalize, prepare, training_data

intents = load_intents()
texts, labels = training_data(intents)
test = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))["queries"]
test_x, test_y = [q["text"] for q in test], [q["intent"] for q in test]


def baseline():
    """The first version of the project: lowercase + word TF-IDF only, no stemming or char n-grams."""
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])


candidates = {
    "Baseline (lowercase + word TF-IDF)": (baseline, normalize),
    "Improved (NLTK stem + word & char TF-IDF)": (build_model, prepare),
}

results, lines = {}, []
for name, (factory, prep) in candidates.items():
    X = [prep(t) for t in texts]
    cv_pred = cross_val_predict(factory(), X, labels, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    model = factory().fit(X, labels)
    held_pred = model.predict([prep(t) for t in test_x])
    results[name] = {"cv_accuracy": round(accuracy_score(labels, cv_pred), 4),
                     "heldout_accuracy": round(accuracy_score(test_y, held_pred), 4),
                     "heldout_macro_f1": round(f1_score(test_y, held_pred, average="macro"), 4)}

best = "Improved (NLTK stem + word & char TF-IDF)"
model = build_model().fit([prepare(t) for t in texts], labels)
pred = model.predict([prepare(t) for t in test_x])
proba = model.predict_proba([prepare(t) for t in test_x]).max(axis=1)
threshold = float(intents.get("confidence_threshold", 0.3))
classes = list(model.classes_)

wrong = [(t, y, p, c) for t, y, p, c in zip(test_x, test_y, pred, proba) if y != p]
low_conf = sum(c < threshold for c in proba)

lines += ["# Evaluation report", "",
          f"Training examples: {len(texts)} across {len(set(labels))} intents. Held-out test phrases: {len(test_x)}. "
          f"NLTK preprocessing active: {NLTK_AVAILABLE}.", "",
          "## Baseline vs improved", "", "| Pipeline | 5-fold CV accuracy | Held-out accuracy | Held-out macro F1 |", "|---|---|---|---|"]
for n, r in results.items():
    lines.append(f"| {n} | {r['cv_accuracy']:.1%} | {r['heldout_accuracy']:.1%} | {r['heldout_macro_f1']:.3f} |")
lines += ["", f"Confidence threshold {threshold}: {low_conf} of {len(test_x)} held-out phrases would be routed to retrieval/fallback.", "",
          "## Per-intent report (held-out, improved pipeline)", "", "```",
          classification_report(test_y, pred, zero_division=0), "```", "",
          "## Confusion matrix (rows = true, columns = predicted)", "", "```", "labels: " + ", ".join(classes),
          str(confusion_matrix(test_y, pred, labels=classes)), "```", "",
          f"## Misclassified held-out phrases ({len(wrong)})", ""]
lines += [f"- \"{t}\": expected `{y}`, got `{p}` ({c:.0%})" for t, y, p, c in wrong] or ["- none"]
report = "\n".join(lines)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "evaluation.md").write_text(report, encoding="utf-8")
(ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(report)
