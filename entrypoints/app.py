import logging

from flask import Flask, request
from flask_cors import CORS

from bootstrap import bootstrap
from services.handlers import handle_query_text

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route("/query/text", methods=["GET"])
def query_text():
    user: str = request.args["user"]
    text: str = request.args["text"]
    n_cands: int = request.args["n_cands"]
    n_rank: int = request.args["n_rank"]
    return handle_query_text(user, text, n_cands, n_rank)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap()

    CORS(app)
    app.run(port=5003, debug=True)
