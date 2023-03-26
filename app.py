import pandas as pd
import numpy as np
import tensorflow as tf

# classes:
classes = [
'car',
'house',
'wine bottle',
'chair',
'table',
'tree',
'camera',
'fish',
'rain',
'clock',
'hat'
]

# labels :
labels = {
'car': 0,
'house': 1,
'wine bottle': 2,
'chair': 3,
'table': 4,
'tree': 5,
'camera': 6,
'fish': 7,
'rain': 8,
'clock': 9,
'hat': 10
}

num_classes = len(classes)

# load the model:
from keras.models import load_model
model = load_model('sketch_recogination_model_cnn.h5')

# Predict function for interface:
def predict_fn(image):

  # preprocessing the size:
  resized_image = tf.image.resize(image, (28, 28))              # Resize image to (28, 28)
  grayscale_image = tf.image.rgb_to_grayscale(resized_image)    # Convert image to grayscale

  image = np.array(grayscale_image)

  # model requirements:
  image = image.reshape(1,28,28,1)
  label = tf.constant(model.predict(image).reshape(num_classes))   # giving 2D output so 1D

  # predict:
  predicted_index = tf.argmax(label)
  class_name = [name for name, index in labels.items() if predicted_index == index][0]
  return class_name


def main():
        
    # application interface:
    import gradio as gr

    gr.Interface(fn=predict_fn, inputs="paint", outputs="label", height=100).launch(share=True, debug=True)
