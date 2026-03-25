import json
import os
import time
from functools import wraps
from datetime import datetime

from flask import Blueprint, request, jsonify
from model.predict import predict_smiles
from utils.reasoning import generate_reasoning, simplify_text, generate_clinical_report

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# ─── Load API Keys ────────────────────────────────────────────
_keys_path = os.path.join(os.path.dirname(__file__), "..", "api_keys.json")
with open(_keys_path) as f:
    _VALID_KEYS = set(json.load(f)["keys"])

# ─── In-memory Rate Limiter (10 req/min per key) ─────────────
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT = 10
RATE_WINDOW = 60  # seconds

def _check_rate(api_key: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW
    hits = _rate_store.get(api_key, [])
    hits = [t for t in hits if t > window_start]
    if len(hits) >= RATE_LIMIT:
        return False
    hits.append(now)
    _rate_store[api_key] = hits
    return True

# ─── Helpers ─────────────────────────────────────────────────
def _ok(data, status=200):
    return jsonify({
        "success": True,
        "version": "v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": data
    }), status

def _err(message, status=400):
    return jsonify({
        "success": False,
        "version": "v1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error": message
    }), status

# ─── Auth Decorator ───────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key not in _VALID_KEYS:
            return _err("Unauthorized: Invalid or missing API key", 401)
        if not _check_rate(key):
            return _err(f"Rate limit exceeded: max {RATE_LIMIT} requests per minute", 429)
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────────

@api_v1.route("/health", methods=["GET"])
def health():
    """Public health check — no auth required."""
    return _ok({"status": "online", "model": "ToxLens GNN + Phi-3 via Ollama"})


@api_v1.route("/predict", methods=["POST"])
@require_api_key
def predict():
    """
    Predict toxicity and interaction risk for two SMILES strings.

    Request body (JSON):
      {
        "smiles1": "<SMILES string for Drug A>",
        "smiles2": "<SMILES string for Drug B>"
      }

    Headers:
      X-API-Key: <your api key>
    """
    body = request.get_json(silent=True)
    if not body or "smiles1" not in body or "smiles2" not in body:
        return _err("Request body must include 'smiles1' and 'smiles2'")

    result = predict_smiles(body["smiles1"], body["smiles2"])
    if "error" in result:
        return _err(result["error"])

    result["mechanistic_interpretation"] = generate_reasoning(result)

    return _ok({
        "drug_a": result["drugA"],
        "drug_b": result["drugB"],
        "interaction": result["interaction"],
        "overall_confidence": result["overall_confidence"],
        "mechanistic_interpretation": result["mechanistic_interpretation"]
    })


@api_v1.route("/simplify", methods=["POST"])
@require_api_key
def simplify():
    """
    Simplify a complex toxicity explanation into plain language.

    Request body (JSON):
      { "text": "<complex explanation text>" }
    """
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return _err("Request body must include 'text'")

    simplified = simplify_text(body["text"])
    return _ok({"simplified_explanation": simplified})


@api_v1.route("/report", methods=["POST"])
@require_api_key
def report():
    """
    Generate a comprehensive clinical toxicity report (Markdown).

    Request body: the full prediction result JSON returned by /api/v1/predict -> data
    """
    body = request.get_json(silent=True)
    if not body:
        return _err("Request body is required")

    report_md = generate_clinical_report(body)
    return _ok({"clinical_report_markdown": report_md})
