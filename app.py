from flask import Flask,request, jsonify
import joblib
import numpy as np 
from flask import Flask, request, jsonify, render_template

# import os

app =Flask(__name__)

model = joblib.load("model/xgboost_fraud_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    
    try:
        features = np.array(data["features"]).reshape(1,-1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        result = {
            "predictions": int(prediction[0]),
            "fraud_probability":float(probability)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error":str(e)}) 
    
    
    
    
    
    
if __name__ == "__main__":
    app.run(debug=True)
       
       
# print("Current Working Directory:", os.getcwd())
