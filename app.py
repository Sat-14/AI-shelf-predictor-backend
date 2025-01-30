from flask import Flask, jsonify, request
from basic import train_initial_model, predict_sales, add_data, retrain_model, model_path, load_model
import os

app = Flask(__name__)

# Check if model exists, otherwise train it
if not os.path.exists(model_path):
    train_initial_model()
else:
    print("Model loaded from disk.")

# Flask endpoint to train the model
@app.route("/train", methods=["POST"])
def train():
    try:
        train_initial_model()
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask endpoint to predict sales and allocate shelf space
@app.route("/predict", methods=["GET"])
def predict():
    try:
        results = predict_sales()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask endpoint to add new data
@app.route("/add-data", methods=["POST"])
def add_new_data():
    try:
        new_data = request.get_json()
        message = add_data(new_data)
        return jsonify({"message": message}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask endpoint to retrain the model incrementally
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        message = retrain_model()
        return jsonify({"message": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)