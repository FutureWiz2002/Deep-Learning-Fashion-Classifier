import tensorflow as tf
from keras.models import load_model


saved_model = load_model('Model/model weights.h5')
print("Model Loaded")
