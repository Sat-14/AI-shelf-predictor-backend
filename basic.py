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
from joblib import dump, load


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
scaler=None
model=None
model_path = "sales_model.h5"
scaler_path = "scaler.pkl"

def preprocess_data(data):
    if not data:
        print("No data in MongoDB!")
        return None, None
    extracted_data=[]    
   #conversion to pandas dataframe
    for record in data:
      for key,value in record.items():
        if key.startswith("item_") and isinstance(value, dict):
            
            if all(k in value for k in ["buy", "time", "profit", "name"]):

                extracted_data.append(value)
    if not extracted_data:
        print("No valid data in MongoDB!")
        return None, None,None            

    df=pd.DataFrame(extracted_data)
    required_columns = ["buy", "time", "profit", "name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {', '.join(missing_columns)}")
        return None, None       
    
    X=df[["buy", "time", "profit", "name"]]
    y=df["profit"]
    return df,X,y

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

    data = list(collection.find())
    df,X, y = preprocess_data(data)
    if X is None or y is None:
        print("No training data available.")
        return

    # Save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    # Build and train model
    model = build_nn_model(input_shape=X_scaled.shape[1])
    model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=1)
    model.save(model_path,save_format="tf")
    dump(scaler, "scaler.pkl")
    print("Model and scaler saved!")

def predict_sales():
    global model, scaler

    data = list(collection.find())
    df, X, y = preprocess_data(data)  # Update this line

    # Load scaler if missing
    if scaler is None:
        scaler = load(scaler_path)

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled).flatten()
    total_sales = predictions.sum()
    shelf_space = (predictions / total_sales) * 100

    results = df.copy()
    results["predicted_sales"] = predictions
    results["shelf_space_percentage"] = shelf_space
    return results.to_dict(orient="records")

def add_data(new_data):
    collection.insert_many(new_data)  # Insert new data into MongoDB
    return "Data added successfully!"

def retrain_model():
    global model,scaler
    if model is None or scaler is None:
        if os.path.exists(model_path) and os.path.exists("scaler.pkl"):
            model=tf.keras.models.load_model(model_path,custom_objects={"mse":tf.keras.losses.MeanSquaredError()})
            scaler=load("scaler.pkl")
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            return "Error:Model or scaler not found .Train the model first."    
    new_data = list(collection.find())
    df, X, y = preprocess_data(new_data)  # Use updated preprocess_data

    # Ensure scaler exists
    if X is None or y is None:
        return "Error: No valid data for retraining."
        

    X_scaled = scaler.transform(X)
    model.fit(X_scaled, y, epochs=5, batch_size=16, verbose=1)
    model.save(model_path)
    return "Model retrained successfully!"
