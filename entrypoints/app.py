import logging

from flask import Flask, request
from flask_cors import CORS

from bootstrap import bootstrap
from handlers import handle_query_text

logger = logging.getLogger(__name__)


def create_app():
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    CORS(app)
    bootstrap(lazy_load=False)
    return app


app = create_app()


@app.route("/query/text", methods=["GET"])
def query_text():
    user: str = request.args["user"]
    text: str = request.args["text"]
    top_n = int(request.args["top_n"])
    exclude_elems = request.args.get("exclude_elems", [])
    return handle_query_text(user, text, top_n, exclude_elems)


if __name__ == "__main__":
    app.run(port=5003)
