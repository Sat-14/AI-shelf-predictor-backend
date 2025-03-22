# 📊 Inventory Sales Predictor  

This project is an AI-powered **inventory sales predictor** that forecasts sales and optimizes shelf space allocation based on real-time data. The model continuously improves with each new data entry.

## 🚀 Features  

✅ **Predict Sales:** Uses a trained neural network to estimate future sales and allocate shelf space accordingly.  
✅ **Incremental Learning:** Continuously retrains the model as new sales data is added.  
✅ **Data Storage:** Utilizes MongoDB for storing sales data.  
✅ **Scalability:** Built with **Flask** and **TensorFlow** for seamless API integration.  
✅ **Visualization:** Generates graphs for **profit per item** and **frequency of sales**.  

---

## 🛠️ Tech Stack  

- **Backend:** Flask  
- **Database:** MongoDB  
- **Machine Learning:** TensorFlow, Keras  
- **Data Processing:** Pandas, NumPy, Scikit-Learn  
- **Visualization:** Matplotlib, Seaborn  

---

## 📞 Installation  

### **Prerequisites**  
Ensure you have the following installed:  
- **Python 3.8+**  
- **MongoDB** (running locally)  
- **Pip**  

### **Steps**  

1️⃣ Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/inventory-sales-predictor.git  
   cd inventory-sales-predictor
   ```

2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3️⃣ Start MongoDB (ensure it's running locally).  

4️⃣ Run the application:  
   ```bash
   python app.py
   ```

---

## 👀 API Endpoints  

### **1️⃣ Train the Model**  
**URL:** `/train`  
**Method:** `POST`  
**Description:** Trains the model from scratch using the existing sales data.  

#### 🔹 Request:  
```bash
curl -X POST http://localhost:5000/train
```

#### 🔹 Response:  
```json
{
  "message": "Model trained successfully!"
}
```

---

### **2️⃣ Retrain the Model with New Data**  
**URL:** `/retrain`  
**Method:** `POST`  
**Description:** Retrains the model incrementally using the latest data.  

#### 🔹 Request:  
```bash
curl -X POST http://localhost:5000/retrain
```

#### 🔹 Response:  
```json
{
  "message": "Model retrained successfully!"
}
```

---

### **3️⃣ Predict Sales and Allocate Shelf Space**  
**URL:** `/predict`  
**Method:** `GET`  
**Description:** Predicts future sales and suggests optimal shelf space allocation.  

#### 🔹 Request:  
```bash
curl -X GET http://localhost:5000/predict
```

#### 🔹 Response:  
```json
[
  {
    "name": 3,
    "buy": 45,
    "profit": 15,
    "time": 1711000000,
    "predicted_sales": 120.5,
    "shelf_space_percentage": 25.4
  },
  {
    "name": 1,
    "buy": 30,
    "profit": 12,
    "time": 1712000000,
    "predicted_sales": 95.2,
    "shelf_space_percentage": 20.1
  }
]
```

---

### **4️⃣ Add New Sales Data**  
**URL:** `/add-data`  
**Method:** `POST`  
**Description:** Adds new sales data to the database.  

#### 🔹 Request:  
```bash
curl -X POST http://localhost:5000/add-data -H "Content-Type: application/json" -d '
[
  {
    "name": 2,
    "buy": 50,
    "profit": 20,
    "time": 1713000000
  },
  {
    "name": 4,
    "buy": 60,
    "profit": 25,
    "time": 1714000000
  }
]'
```

#### 🔹 Response:  
```json
{
  "message": "Data added successfully!"
}
```

---

### **5️⃣ Profit by Item Graph**  
**URL:** `/profit_by_item`  
**Method:** `GET`  
**Description:** Generates and returns a **bar graph** showing profit per item.  

#### 🔹 Request:  
```bash
curl -X GET http://localhost:5000/profit_by_item
```

#### 🔹 Response:  
Returns a **PNG image** of the graph.  

---

### **6️⃣ Frequency of Sale Graph**  
**URL:** `/frequency_of_sale`  
**Method:** `GET`  
**Description:** Generates and returns a **bar graph** showing how often each item is sold.  

#### 🔹 Request:  
```bash
curl -X GET http://localhost:5000/frequency_of_sale
```

#### 🔹 Response:  
Returns a **PNG image** of the graph.  

---

## 🤝 Contributing  
Pull requests are welcome! If you’d like to contribute, open an issue first to discuss your ideas.  

---

## 📜 License  
**MIT License**  

---


