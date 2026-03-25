from flask import Flask, request, jsonify
from flask_cors import CORS

from model.predict import predict_smiles
from utils.reasoning import generate_reasoning

app = Flask(__name__)
CORS(app)

# ✅ ADD HERE
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

# ⛔ THIS MUST BE LAST
if __name__ == "__main__":
    app.run(debug=True)