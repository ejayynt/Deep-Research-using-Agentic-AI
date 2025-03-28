import os
import sys
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_deep_research

app = Flask(__name__)
CORS(app)


@app.route("/api/deep-research", methods=["POST"])
def deep_research_endpoint():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Invalid JSON",
                    "details": str(e),
                    "raw_data": request.data.decode("utf-8"),
                }
            ),
            400,
        )

    if not data or "query" not in data or not data["query"]:
        return (
            jsonify(
                {"error": "A non-empty 'query' is required", "received_data": data}
            ),
            400,
        )

    try:
        results = run_deep_research(data["query"])
        clean_results = {
            "query": results.get("query", ""),
            "final_answer": results.get("final_answer", ""),
            "sources": results.get("sources", []),
            "workflow_path": results.get("workflow_path", []),
        }
        return jsonify(clean_results), 200

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "An error occurred during research",
                    "details": str(e),
                    "traceback": traceback.format_exc(),
                }
            ),
            500,
        )


if __name__ == "__main__":
    if "MISTRAL_API_KEY" not in os.environ or "TAVILY_API_KEY" not in os.environ:
        from dotenv import load_dotenv

        load_dotenv()

    app.run(host="0.0.0.0", port=5000, debug=True)
