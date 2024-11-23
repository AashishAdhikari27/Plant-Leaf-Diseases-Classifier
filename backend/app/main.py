from fastapi import FastAPI, UploadFile, File
import uvicorn

from typing import Union
from pydantic import BaseModel

import tensorflow as tf
from PIL import Image

import json

import numpy as np  
import os

app = FastAPI()




model = tf.keras.models.load_model("../plant_disease_prediction_model.h5")




# Load class indices from JSON
with open('../class_indices.json') as f:
    class_indices = json.load(f)

# Reverse mapping for predictions
index_to_class = {int(k): v for k, v in class_indices.items()}

# Image preprocessing function
def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)):
    image = image.resize(target_size)  # Resize the image
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array.astype('float32') / 255.0  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# API endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image = Image.open(file.file)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Get model predictions
        predictions = model.predict(preprocessed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = index_to_class[predicted_index]
        
        # Return the result
        return {"filename": file.filename, "predicted_class": predicted_class}
    except Exception as e:
        return {"error": str(e)}



