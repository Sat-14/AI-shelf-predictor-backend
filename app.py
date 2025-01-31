from flask import Flask, jsonify, request
from basic import train_initial_model, predict_sales, add_data, retrain_model, model_path, load_model
import os
from basic import train_initial_model, predict_sales, add_data, retrain_model, model_path, scaler
import tensorflow as tf 
from tensorflow.keras.models import load_model
from joblib import load
tf.config.run_functions_eagerly(True)

app = Flask(__name__)

# If our model doesn't exist, we play scientist and create it from scratch!
if not os.path.exists(model_path) or not os.path.exists("scaler.pkl"):
    print("Oops, no model found! Time to train one from scratch...")
    train_initial_model()
else:
    print("Good news! Found an existing model. Loading it like a boss...")
    
    model = load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    scaler = load("scaler.pkl")
    from basic import model as basic_model, scaler as basic_scaler
    basic_model = model
    basic_scaler = scaler

# Endpoint to train the model - because even AI needs a workout
@app.route("/train", methods=["POST"])
def train():
    try:
        train_initial_model()
        return jsonify({"message": "The AI has been trained. It now knows more than before!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to predict sales and decide how much shelf space things deserve
@app.route("/predict", methods=["GET"])
def predict():
    try:
        results = predict_sales()
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to add new data - because fresh data is always in style
@app.route("/add-data", methods=["POST"])
def add_new_data():
    try:
        new_data = request.get_json()
        message = add_data(new_data)
        return jsonify({"message": message}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to retrain the model - because AI forgets things too
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        message = retrain_model()
        return jsonify({"message": message}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Running the Flask app - because somebody's gotta do it
if __name__ == "__main__":
    print("Starting the Flask app... Hold on to your hats!")
    app.run(debug=True)
