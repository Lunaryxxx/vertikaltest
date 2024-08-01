from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import pandas as pd
import os

# Get the current directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and label encoder with absolute paths
model_path = os.path.join(BASE_DIR, 'vertikaltest.pkl')
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

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
            return Response('Invalid input. Expecting 4 sensor readings.', status=400, mimetype='text/plain')

        # Convert the sensor readings to a DataFrame with the expected feature names
        # Ensure the columns are in the correct order that the model expects
        input_data = pd.DataFrame([sensor_readings], columns=['TDS_ppm', 'TEMPERATURE_c', 'HUMIDITY', 'pH'])

        # Reorder the DataFrame columns to match the order used during model training
        input_data = input_data[['TEMPERATURE_c', 'HUMIDITY', 'TDS_ppm', 'pH']]

        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        # Decode the prediction back to the original label
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]

        # Send back the result as plain text with the desired format
        return Response(f"{decoded_prediction}", mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500, mimetype='text/plain')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
