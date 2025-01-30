from flask import*
import pandas as pd 
import numpy as np 
from pymongo import*
from sklearn.preprocessing import*
import tensorflow as tf
from tensorflow.keras.models import*
from tensorflow.keras.layers import*
import os
from datetime import*
import random

#connecting to mongodb

client=MongoClient("mongodb://localhost:27017")
db=client['store_sales']
collection=db['sales_data']
print("Connected to Database successfully")
#code to encode data name
'''def process_data(df):
    df['name'] = df['name'].apply(lambda x: str(x) if isinstance(x, dict) else x)
    df = pd.get_dummies(df, columns=['name'], drop_first=True)
    
    return df'''
#inserting data into database
#should be ran only once for testing purposes
#creating random 500 items data 
'''def generate_random_items():
    new_data = {}
    for i in range(1, 501):
        item = {
            "name":random.randint(1,5),
            "buy":random.randint(1, 100),
            "profit":random.randint(1, 30),
            "time": random.randint(
                int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()),
                int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
            )
        }
        new_data[f"item_{i}"] = item
    return new_data
new_data = generate_random_items()
# Insert the data into MongoDB
for key, value in new_data.items():
    collection.insert_one({key: value})

print("Data inserted into MongoDB successfully.")'''
#global babies
model_path = "sales_model.h5"

def preprocess_data(df):
    if "_id" in df.columns:
        df.drop("_id", axis=1, inplace=True)
    df.fillna(0, inplace=True)
    X = df[["buy", "profit", "name", "time"]]
    y = df["profit"]
    return X, y

def build_nn_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_initial_model():
    global model, scaler

    # Fetch data from MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Preprocess data
    X, y = preprocess_data(df)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build and train the model
    model = build_nn_model(input_shape=X_scaled.shape[1])
    model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=1)

    # Save the model and scaler
    model.save(model_path)
    print("Model trained and saved!")

def predict_sales():
    global model, scaler

    # Fetch data from MongoDB
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Preprocess data
    X, y = preprocess_data(df)
    X_scaled = scaler.transform(X)

    # Predict sales
    predictions = model.predict(X_scaled)
    total_sales = sum(predictions)

    # Allocate shelf space percentages
    shelf_space = [(pred / total_sales) * 100 for pred in predictions]

    # Combine predictions and shelf space allocation
    results = df.copy()
    results["predicted_sales"] = predictions
    results["shelf_space_percentage"] = shelf_space

    return results.to_dict(orient="records")

def add_data(new_data):
    collection.insert_many(new_data)  # Insert new data into MongoDB
    return "Data added successfully!"

def retrain_model():
    global model, scaler
    new_data = list(collection.find())
    df = pd.DataFrame(new_data)

    # Preprocess the new data
    X, y = preprocess_data(df)
    X_scaled = scaler.transform(X)

    # Incrementally train the model with new data
    model.fit(X_scaled, y, epochs=5, batch_size=16, verbose=1)
    model.save(model_path)

    return "Model retrained successfully!"