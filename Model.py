
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

def preprocess_image(image_path, input_shape):  # Add input_shape for tflite
    try:
        img = Image.open(image_path).convert("RGB")  # Ensure RGB for consistency
        # Resize for tflite model if needed.  Use cv2 if PIL resizing causes issues.
        img = img.resize(input_shape[:2]) # Resize to (height, width)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and ensure float32 for tflite
        return img_array
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None
    except Exception as e:  # Catch other potential image processing errors
        print(f"Error preprocessing image: {e}")
        return None


def predict_disease(image_path):
    input_shape = input_details[0]['shape']  # Get input shape from tflite model
    img_array = preprocess_image(image_path, input_shape[1:3])  # Pass height, width

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


