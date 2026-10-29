import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from chatbot import Chatbot, normalize, preprocess, training_data, load_intents

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTests(unittest.TestCase):
    def test_normalize_strips_noise(self):
        self.assertEqual(normalize("  HELLO!!! Visit https://x.com now "), "hello visit now")

    def test_stemming_merges_word_forms(self):
        self.assertEqual(preprocess("training"), preprocess("trained"))

    def test_stop_words_removed_but_never_empty(self):
        self.assertNotIn("the", preprocess("the model").split())
        self.assertTrue(preprocess("the a of"))

    def test_apostrophes_kept(self):
        self.assertIn("what's", normalize("What's up?"))


class DataTests(unittest.TestCase):
    def test_every_intent_has_patterns_and_responses(self):
        for item in load_intents()["intents"]:
            self.assertGreaterEqual(len(item["patterns"]), 20, item["tag"])
            self.assertTrue(item["responses"], item["tag"])

    def test_no_pattern_has_two_labels(self):
        texts, labels = training_data(load_intents())
        seen = {}
        for text, label in zip(texts, labels):
            self.assertEqual(seen.setdefault(text.lower(), label), label, text)

    def test_test_set_not_in_training_set(self):
        texts, _ = training_data(load_intents())
        train = {t.lower() for t in texts}
        queries = json.loads((ROOT / "data" / "test_queries.json").read_text())["queries"]
        self.assertFalse([q["text"] for q in queries if q["text"].lower() in train])


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bot = Chatbot(model_path=Path(cls.tmp.name) / "m.joblib", seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def intent(self, text):
        return self.bot.respond(text)["intent"]

    def test_core_intents(self):
        cases = {"hello": "greeting", "bye": "goodbye", "thank you so much": "thanks",
                 "what is natural language processing": "nlp_definition",
                 "how do you preprocess text": "preprocessing", "which libraries do you use": "libraries",
                 "how were you trained": "training", "how do i run the app": "interface",
                 "how accurate are you": "evaluation", "tell me a joke": "joke",
                 "what is the weather today": "out_of_scope"}
        for text, expected in cases.items():
            self.assertEqual(self.intent(text), expected, text)

    def test_typo_robustness(self):
        self.assertEqual(self.intent("wich libraries do you use"), "libraries")

    def test_empty_and_symbol_only_input(self):
        for text in ["", "   ", "???", None, 42]:
            self.assertEqual(self.bot.respond(text)["intent"], "empty_input")

    def test_gibberish_goes_to_fallback(self):
        result = self.bot.respond("zxqv wplk mmnb")
        self.assertIn(result["intent"], {"fallback", "out_of_scope"})
        self.assertTrue(result["response"])

    def test_knowledge_base_retrieval(self):
        hit = self.bot.retrieve("what is overfitting and generalisation")
        self.assertIsNotNone(hit)
        self.assertIn("overfit", hit[0].lower())

    def test_long_input_is_handled(self):
        self.assertTrue(self.bot.respond("hello " * 1000)["response"])

    def test_responses_vary_but_never_repeat_back_to_back(self):
        replies = [self.bot.respond("hello")["response"] for _ in range(6)]
        self.assertTrue(all(a != b for a, b in zip(replies, replies[1:])))

    def test_model_is_saved_and_reloaded(self):
        path = Path(self.tmp.name) / "m.joblib"
        self.assertTrue(path.exists())
        self.assertEqual(Chatbot(model_path=path).respond("hello")["intent"], "greeting")

    def test_heldout_accuracy_threshold(self):
        queries = json.loads((ROOT / "data" / "test_queries.json").read_text())["queries"]
        correct = sum(self.bot.predict(q["text"])[0] == q["intent"] for q in queries)
        self.assertGreaterEqual(correct / len(queries), 0.90)


class FlaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.client = create_app(Chatbot(model_path=Path(cls.tmp.name) / "m.joblib")).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_index_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"NLP Chatbot", r.data)

    def test_static_assets(self):
        for path in ("/static/chat.js", "/static/style.css"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_chat_endpoint(self):
        r = self.client.post("/chat", json={"message": "hello"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["intent"], "greeting")
        self.assertTrue(data["response"])

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/chat", data="nope").status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"message": 5}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={}).status_code, 400)

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
