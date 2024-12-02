import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import os

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
MODEL_PATH = "saved_model/my_trained_cnn_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
def main():
    st.title("Math156_Handwritten Digit Recognition App_Created by Runpeng, William, Jiawei")
    st.write("Please upload an image of a handwritten digit for prediction.")

    # File uploader for digit images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...Please wait a moment.")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        # Display prediction
        st.write(f"The model predicts this digit is: {predicted_digit}")

# Helper function to preprocess image
def preprocess_image(image):
    # Invert image if necessary (assuming handwriting is in black)
    image = ImageOps.invert(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Normalize pixel values (0-1)
    image_array = np.array(image) / 255.0
    # Reshape to add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

if __name__ == "__main__":
    main()
