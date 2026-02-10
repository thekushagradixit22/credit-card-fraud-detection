import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_fraud_model.pkl")

def predict_transaction(transaction_features):
    """
    transaction_features should be a list or numpy array
    containing all 30 features in correct order
    """
    features = np.array(transaction_features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        return f"🚨 FRAUD detected! Probability: {probability:.4f}"
    else:
        return f"✅ Normal transaction. Fraud Probability: {probability:.4f}"


# Example test (replace with real values later)
dummy_input = [0]*30  # temporary dummy transaction

print(predict_transaction(dummy_input))
dummy_input = [-2.31222654,  1.95199201, -1.60985073,  3.99790559, -0.52218786,
       -1.42654532, -2.53738731,  1.39165725, -2.77008928, -2.77227214,
        3.20203321, -2.89990739, -0.59522188, -4.28925378,  0.38972412,
       -1.14074718, -2.83005567, -0.01682247,  0.41695571,  0.12691056,
        0.51723237, -0.03504937, -0.46521108,  0.3201982 ,  0.04451917,
        0.1778398 ,  0.261145  , -0.14327587, -0.30741284, -0.9902137 ]

input_array = np.array(dummy_input).reshape(1, -1)

prediction = model.predict(input_array)
probability = model.predict_proba(input_array)[0][1]

if prediction[0] == 1:
    print(f"🚨 Fraud Detected! Probability: {probability:.4f}")
else:
    print(f"✅ Normal transaction. Fraud Probability: {probability:.4f}")