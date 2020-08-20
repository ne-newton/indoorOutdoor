import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

#image dimension for model
img_height = 256
img_width = 256

#user input
parser = argparse.ArgumentParser(
    description='Attempts to recognize if a photo depicts an indoor or outdoor environment.')
parser.add_argument('-i', '--image', default=None, type=str,
    help='path to image to open for test. Default=none')
args = parser.parse_args()

#make sure they input something...
if args.image:
    path = args.image
else:
    raise Exception('No image path was provided.')

#convert image for model
img = keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
img = keras.preprocessing.image.img_to_array(img)
img = tf.expand_dims(img, 0)

#load model
model = keras.models.load_model('model.h5')

#predict image based on model
prediction = model.predict(img)

if prediction[0][0] < 0.5:
    print(
        "There is a {:.1f}% chance that this image was taken indoors"
        .format(abs((100 * np.max(prediction[0][0]))-100))
    )
else:
    print(
        "There is a {:.1f}% chance that this image was taken outdoors"
        .format(100 * np.max(prediction[0][0]))
    )