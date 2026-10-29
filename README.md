# AI Chatbot using NLP

An intent-based chatbot built with Python, **NLTK**, **scikit-learn** and a **Flask** web interface. It recognises what the user wants (intent recognition), answers from curated responses, searches a small knowledge base when unsure, and falls back safely when it doesn't know.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py          # trains and saves models/intent_model.joblib
python app.py            # open http://127.0.0.1:5000
```
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, then the same pip/python commands.

```bash
python evaluate.py                      # accuracy report -> reports/evaluation.md
python -m unittest discover -s tests    # automated tests
```

## How the 7 project tasks are covered
| # | Task | Where |
|---|---|---|
| 1 | Collect / create conversational training data | `data/intents.json`: 16 intents, ~550 labelled example phrases and 3 replies per intent. `data/knowledge_base.txt`: 9 passages for retrieval. |
| 2 | Preprocess text with NLP techniques | `chatbot.py` → `normalize()` / `preprocess()`: lowercase, URL and punctuation removal, **NLTK tokenization**, short stop-word list, **NLTK Snowball stemming**, then TF-IDF with word 1-2-grams and character 2-5-grams (handles typos). |
| 3 | Use NLTK / spaCy / TensorFlow | **NLTK** (`RegexpTokenizer`, `SnowballStemmer`) plus scikit-learn for TF-IDF and the classifier. |
| 4 | Train the chatbot | `train.py`: Logistic Regression on the TF-IDF features, saved with joblib. The model retrains automatically if `intents.json` changes. |
| 5 | Intent recognition + response generation | `Chatbot.respond()`: predicts intent and confidence → picks a reply (never the same one twice in a row). Below the confidence threshold it retrieves the best knowledge-base passage (TF-IDF cosine similarity), otherwise returns a fallback. |
| 6 | Simple UI with Flask or Streamlit | `app.py` + `templates/` + `static/`: chat page with suggestion chips, a toggle showing intent/confidence, dark-mode support, and a `POST /chat` JSON API. |
| 7 | Test accuracy and improve | `evaluate.py` (cross-validation, held-out test, baseline comparison, confusion matrix, misclassified phrases) and 21 unit tests in `tests/`. |

## Evaluation and improvement
`evaluate.py` compares the first-version pipeline (lowercase + word TF-IDF) with the improved one:

| Pipeline | 5-fold CV accuracy | Held-out accuracy |
|---|---|---|
| Baseline | 65.5% | 83.1% |
| Improved (stemming + word & char TF-IDF) | 75.5% | 97.7% |

Notes for an honest reading of these numbers:
- The **held-out set** (`data/test_queries.json`, 130 phrases) was written separately from the training phrases, and a unit test checks that none of them appear in the training data.
- After the first run I reviewed the misclassified phrases and added more varied training examples for the weak intents (greeting vs how_are_you, bot_identity, out_of_scope). That is the improvement loop, but it also means the held-out figure is somewhat optimistic. For a fair final number, write a fresh set of test phrases.
- **Cross-validation is lower** because it holds out phrases from a small, deliberately diverse dataset; it is the more conservative estimate.
- The bundled `reports/evaluation.md` was generated in an environment without NLTK, so it used a simple fallback stemmer. Run `python train.py` and `python evaluate.py` after installing the requirements to regenerate it with NLTK; figures may differ slightly.

## Improving the bot further
Add varied phrases to an intent in `data/intents.json`, run `python train.py`, then `python evaluate.py`. Variety of wording helps more than volume. Check `reports/evaluation.md` for the phrases that are still misclassified.

## Limitations
This is a template-based intent classifier, not a generative model: it can only answer what it was taught, and out-of-domain questions are declined. The dataset is small and hand-written. spaCy or a TensorFlow/Keras LSTM could replace the preprocessing or classifier if you want to extend the project.

## Structure
```
app.py            Flask app           train.py      train + save model
chatbot.py        NLP + chatbot logic evaluate.py   metrics and reports
data/             intents.json, test_queries.json, knowledge_base.txt
templates/ static/  web UI           tests/        unit tests
models/ reports/  generated output
```
