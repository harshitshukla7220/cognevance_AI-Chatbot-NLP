"""Flask web interface for the NLP chatbot. Run: python app.py  ->  http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request

from chatbot import Chatbot, NLTK_AVAILABLE


def create_app(bot: Chatbot | None = None) -> Flask:
    app = Flask(__name__)
    app.config["BOT"] = bot or Chatbot()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", nltk=NLTK_AVAILABLE)

    @app.post("/chat")
    def chat():
        payload = request.get_json(silent=True)
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str):
            return jsonify(error="Send JSON like {\"message\": \"hello\"}"), 400
        result = app.config["BOT"].respond(message)
        result["confidence"] = round(result["confidence"], 3)
        return jsonify(result)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
