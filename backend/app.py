from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from model.predict import predict_smiles
from utils.reasoning import generate_reasoning, simplify_text, generate_clinical_report
from api.v1 import api_v1

app = Flask(__name__)
CORS(app)

# ─── Register Pharma API Blueprint ───────────────────────────
app.register_blueprint(api_v1)

# ─── Swagger / OpenAPI Docs Page ─────────────────────────────
SWAGGER_UI = """<!DOCTYPE html>
<html>
<head>
  <title>ToxLens API v1 — Documentation</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>body{margin:0;background:#060a0f;} .swagger-ui .topbar{background:#0b1118;}</style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({
  spec: {
    openapi: "3.0.0",
    info: {
      title: "ToxLens Pharma API",
      version: "1.0.0",
      description: "Graph Neural Network drug toxicity prediction API for pharmaceutical integration."
    },
    servers: [{ url: "http://127.0.0.1:5000" }],
    components: {
      securitySchemes: {
        ApiKeyAuth: { type: "apiKey", in: "header", name: "X-API-Key" }
      }
    },
    security: [{ ApiKeyAuth: [] }],
    paths: {
      "/api/v1/health": {
        get: {
          summary: "Health Check",
          security: [],
          responses: { "200": { description: "API is online" } }
        }
      },
      "/api/v1/predict": {
        post: {
          summary: "Predict Drug Toxicity & Interaction",
          requestBody: {
            required: true,
            content: {
              "application/json": {
                schema: {
                  type: "object",
                  properties: {
                    smiles1: { type: "string", example: "CCO", description: "SMILES string for Drug A" },
                    smiles2: { type: "string", example: "C1=CC=C(C=C1)[N+](=O)[O-]", description: "SMILES string for Drug B" }
                  },
                  required: ["smiles1", "smiles2"]
                }
              }
            }
          },
          responses: { "200": { description: "Prediction result with mechanistic interpretation" }, "401": { description: "Invalid API key" }, "429": { description: "Rate limit exceeded" } }
        }
      },
      "/api/v1/simplify": {
        post: {
          summary: "Simplify Toxicity Explanation",
          requestBody: {
            required: true,
            content: { "application/json": { schema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } }
          },
          responses: { "200": { description: "Plain-language simplified explanation" } }
        }
      },
      "/api/v1/report": {
        post: {
          summary: "Generate Clinical Report",
          requestBody: {
            required: true,
            content: { "application/json": { schema: { type: "object", description: "Full prediction result from /predict" } } }
          },
          responses: { "200": { description: "Markdown clinical toxicity report" } }
        }
      }
    }
  },
  dom_id: "#swagger-ui",
  presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
  layout: "BaseLayout"
});
</script>
</body>
</html>"""

@app.route("/api/v1/docs")
def api_docs():
    return render_template_string(SWAGGER_UI)

# ─── Original Frontend Routes (kept for UI) ──────────────────
@app.route("/")
def home():
    return "ToxLens Backend Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_smiles(data["smiles1"], data["smiles2"])
    if "error" in result:
        return jsonify(result), 400
    result["reason"] = generate_reasoning(result)
    return jsonify(result)

@app.route("/simplify", methods=["POST"])
def simplify():
    data = request.get_json()
    text = data.get("text", "")
    simple_text = simplify_text(text)
    return jsonify({"simplified": simple_text})

@app.route("/report", methods=["POST"])
def report():
    data = request.get_json()
    report_md = generate_clinical_report(data)
    return jsonify({"report": report_md})

# ⛔ THIS MUST BE LAST
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)