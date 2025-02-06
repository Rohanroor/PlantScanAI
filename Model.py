
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
#import cv2  # For image resizing if needed
#bug fix1...
import joblib  # Add this import
#bug fix1...


# Load the .tflite model
try:
    interpreter = tf.lite.Interpreter(model_path="cnn_model.tflite")  # Replace with your .tflite file path
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ==============================================================
    # ADD THIS CODE TO INSPECT INPUT SHAPE
    # ==============================================================
    input_shape = input_details[0]['shape']
    print(f"Model input shape: {input_shape}")  # Debugging line
    # ==============================================================

except ValueError as e:
    print(f"Error loading .tflite model: {e}. Check the file path.")
    exit()  # Or handle the error appropriately

    
'''
# Load the .pkl models
try:
    with open("svm_model.pkl", "rb") as f:  # Replace with your .pkl file path
        model1 = pickle.load(f)
    with open("rf_model.pkl", "rb") as f:  # Replace with your other .pkl file path
        model2 = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading .pkl model: {e}. Check the file paths.")
    exit()
'''    

#bug fix1...

# Load models with joblib
try:
    model1 = joblib.load("svm_model.pkl")
    model2 = joblib.load("rf_model.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
#bug fix1...    

def preprocess_image(image_path, target_height, target_width, has_batch_dim):
    try:
        img = Image.open(image_path).convert("L")  # <-- CHANGE "RGB" to "L" for grayscale
        img = img.resize((target_width, target_height))  # PIL uses (width, height)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array.astype(np.float32)

        # Add channel dimension (grayscale has 1 channel)
        img_array = np.expand_dims(img_array, axis=-1)  # Shape: [H, W, 1]

        # Add batch dimension if required
        if has_batch_dim:
            img_array = np.expand_dims(img_array, axis=0)  # Shape: [1, H, W, 1]

        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_disease(image_path):
    input_shape = input_details[0]['shape']  # Get input shape from tflite model

    # Determine if the model expects a batch dimension
    if len(input_shape) == 4:
        # Model expects a 4D input: [batch, height, width, channels]
        target_height = input_shape[1]
        target_width = input_shape[2]
        has_batch_dim = True
    else:
        # Model expects a 3D input: [height, width, channels]
        target_height = input_shape[0]
        target_width = input_shape[1]
        has_batch_dim = False

    # Preprocess the image
    img_array = preprocess_image(
        image_path,
        target_height=target_height,
        target_width=target_width,
        has_batch_dim=has_batch_dim  # Pass this flag to control batch dimension
    )

    if img_array is None:
        return None

    # TFLite Model Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    tflite_prediction = np.argmax(tflite_output)  # Get predicted class (example)

    # PKL Model Predictions (Adapt as needed - Example below)
    try:
      pkl_prediction1 = model1.predict(img_array.reshape(img_array.shape[0], -1))[0]  # Reshape if needed by model1
      pkl_prediction2 = model2.predict(img_array.reshape(img_array.shape[0], -1))[0]  # Reshape if needed by model2

    except AttributeError as e: # Handle models that don't have a predict method (e.g., some transformers)
        if hasattr(model1, '__call__'):
            pkl_prediction1 = model1(img_array)
        else:
            print(f"Error: model1 does not have a predict or __call__ method: {e}")
            pkl_prediction1 = None

        if hasattr(model2, '__call__'):
            pkl_prediction2 = model2(img_array)
        else:
            print(f"Error: model2 does not have a predict or __call__ method: {e}")
            pkl_prediction2 = None
    except ValueError as e:
        print(f"Error during PKL model prediction (reshape issues?): {e}")
        return None

    # Combine predictions (Example - Adapt to your logic)
    # This is a simple example. You'll likely need a more sophisticated
    # way to combine the predictions based on your model training.
    final_prediction = (tflite_prediction + pkl_prediction1 + pkl_prediction2) // 3 # averaging example.

    return final_prediction


# Example usage
image_path = "pl.jpg"  # Replace with the actual image path
prediction = predict_disease(image_path)

if prediction is not None:
    print(f"Predicted Disease (0-8): {prediction}")
else:
    print("Prediction failed.")


