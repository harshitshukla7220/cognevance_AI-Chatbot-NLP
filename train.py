"""Train the intent classifier and save it to models/intent_model.joblib."""
import joblib
from chatbot import DATA_PATH, MODEL_PATH, _signature, build_model, load_intents, prepare, training_data, NLTK_AVAILABLE

intents = load_intents()
texts, labels = training_data(intents)
model = build_model()
model.fit([prepare(t) for t in texts], labels)
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "signature": _signature(DATA_PATH)}, MODEL_PATH)
print(f"Trained on {len(texts)} examples, {len(set(labels))} intents (NLTK preprocessing: {NLTK_AVAILABLE}).")
print(f"Saved model to {MODEL_PATH}")
