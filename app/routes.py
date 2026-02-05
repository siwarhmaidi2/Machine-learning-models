from app import app
from flask import render_template,Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/cross')
def cross():
    return render_template('cross.html')
@app.route('/approve')
def approve():
    return render_template('approved.html')
@app.route('/tree')
def tree():
    return render_template('tree.html')
@app.route('/ada')
def ada():
    return render_template('ada.html')
@app.route('/gbm')
def gbm():
    return render_template('gbm.html')
@app.route('/xgb')
def xgb():
    return render_template('xgb.html')
@app.route('/random_forest')
def random_forest():
    return render_template('random_forest.html')
@app.route('/svm')
def svm():
    return render_template('svm.html')
models = {
    "model_knn": joblib.load("models/knn_model_churn.pkl"),
   
}

scaler = joblib.load("models/knn_scaler.pkl") 

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": "Model not found"}), 404
    
    model = models[model_name]

    try:
        # Extract input data from the POST request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        #features = np.array([[1, 300, 10, 0,10,5]])  # Replace with your test data
        scaled_features = scaler.transform(features)
        # Make prediction
        prediction = model.predict(scaled_features)
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


model_tree = joblib.load("models/decision_tree_model.pkl")
selected_features_tree = joblib.load("models/selected_features.pkl")

@app.route('/predict', methods=['POST'])
def predict_tree():
    # Get JSON data from the request
    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "No input data provided or 'features' key missing"}), 400

    # Extract features from the request
    raw_features = data["features"]
    
    # Validate input length
    if len(raw_features) != len(selected_features_tree):
        return jsonify({"error": f"Expected {len(selected_features_tree)} features, got {len(raw_features)}"}), 400

    # Convert the input features to a DataFrame
    input_df = pd.DataFrame([raw_features], columns=selected_features_tree)

    # Ensure proper data formatting
    try:
        input_df = input_df.astype(float)  # Convert numeric fields if needed
    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

    # Convert the DataFrame to a NumPy array for prediction
    input_array = input_df.to_numpy()

    # Make predictions
    probabilities = model_tree.predict_proba(input_array)[:, 1]  # Positive class probabilities
    threshold = 0.65
    predictions = (probabilities >= threshold).astype(int)

    # Return the result
    result = {
        "probabilities": probabilities.tolist(), 
        "predictions": predictions.tolist()
    }
    return jsonify(result)



models_v2 = {
    "model_ada": joblib.load("models/best_ada_model.pkl"),
    "model_random_forest": joblib.load("models/random_forest_model.pkl"),
    "model_gbm": joblib.load("models/random_search_gbm.pkl"),
    "model_xgb": joblib.load("models/xgb_model.pkl"),
   
}

scaler_telecom = joblib.load("models/telecom.pkl") 
selected_features_v2=joblib.load("models/selected_features_forest.pkl") 
@app.route("/predictV2/<model_name>", methods=["POST"])
def predict_v2(model_name):
    if model_name not in models_v2:
        return jsonify({"error": "Model not found"}), 404
    
    model = models_v2[model_name]

    try:
        # Extract input data from the POST request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        #features = np.array([[1, 300, 10, 0,10,5]])  # Replace with your test data
        scaled_features = scaler_telecom.transform(features)
        # Make prediction
        prediction = model.predict(scaled_features)
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400