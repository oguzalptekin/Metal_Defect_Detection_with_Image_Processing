import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import run_model as rn



app = Flask(__name__)
HOST_ADDRESS = 'localhost'


# Endpoint to receive the image and return the plot
@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        # Retrieve the image data from the request
        image_data = request.json['image']

        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image (if required) for your machine learning model
        processed_image = rn.preprocess_image(image)

        # Make predictions using your machine learning model
        model = keras.models.load_model("defect_model.h5")
        predictions = model.predict(tf.expand_dims(processed_image, axis=0))

        # Generate a plot using matplotlib
        # For demonstration purposes, we'll plot a simple sine wave
        plt = rn.draw_plot([processed_image], [predictions])

        # Save the plot to a byte buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the plot to base64 encoded string
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Close the plot to release resources
        plt.close()

        # Return the plot data as a response to the React Native app
        return jsonify({'plot': plot_data})
    else:
        # Handle other methods (GET, PUT, etc.) if necessary
        return jsonify({'error': 'Method Not Allowed'}), 405


if __name__ == '__main__':
    app.run(host=HOST_ADDRESS, port=5000)

#done