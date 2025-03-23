from flask import Flask, request, jsonify, make_response
import pickle
import numpy as np
from flask_cors import CORS  # For handling cross-origin requests'
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # This enables CORS for all routes and origins

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Load the model
with open('xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        print("Received OPTIONS request")
        return make_response('', 200)
    
    try:
        print("Received POST request to /predict")
        data = request.json
        print(f"Received data: {data}")
        
        # Create a features array in the correct order expected by the model
        features = []
        
        # Extract each feature in order
        for i in range(1, 34):  # For questions 1 through 33
            # Get value from the request, default to 0 if missing
            value = data.get(str(i), 0)
            
            # Convert to appropriate numeric type
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0
                
            features.append(value)
        
        print(f"Processed features array: {features}")
        
        # Reshape for model prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        print("Making prediction with model...")
        prediction_probabilities = model.predict_proba(features_array)[0]
        prediction = model.predict(features_array)[0]  # Binary prediction (0 or 1)
        risk_score = prediction_probabilities[1] * 100  # Probability of high risk as percentage
        
        print(f"Prediction result: {prediction}")
        print(f"Risk score: {risk_score}")
        
        # Set risk level based on binary prediction
        if prediction == 0:
            risk_level = "low"
            risk_label = "Bugar Bareng Gigin"
        else:  # prediction == 1
            risk_level = "high"
            risk_label = "Siaga Bareng Jajal"
        
        result = {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'prediction': int(prediction),
            'success': True
        }
        
        print(f"Sending response: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)