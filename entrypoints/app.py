import logging

from flask import Flask, request
from flask_cors import CORS

from services.handlers import handle_embed_text, handle_text_query

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route("/query/text", methods=["GET"])
def serve_text_query():
    user: str = request.args["user"]
    text: str = request.args["text"]
    return handle_text_query(user, text)


@app.route("/embed/text", methods=["GET"])
def embed_text():
    text: str = request.args["text"]
    return handle_embed_text(text)


if __name__ == "__main__":
    CORS(app)
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, port=5003)
