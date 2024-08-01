from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model and label encoder
model = joblib.load('vertikaltest.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract the sensor readings
        sensor_readings = data.get("sensor_readings")

        # Validate the input data
        if not sensor_readings or len(sensor_readings) != 4:
            return jsonify({'error': 'Invalid input. Expecting 4 sensor readings.'}), 400

        # Convert the sensor readings to a DataFrame with the expected feature names
        input_data = pd.DataFrame([sensor_readings], columns=['TDS_ppm', 'TEMPERATURE_c', 'HUMIDITY', 'pH'])

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Decode the prediction back to the original label
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': decoded_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
