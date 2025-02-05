# Python (for testing/prototyping - not deployment)
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()
print("the updated model is loaded")