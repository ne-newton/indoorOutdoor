import numpy as np
import os
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import unittest

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

#Set variables in common between training and validation
batch_size = 32
img_height = 256
img_width = 256
val_split = 0.25
rnd_seed = 101
directory = 'images'
label = 'inferred'

#Create datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels=label,
    image_size=(img_height,img_width),
    batch_size=batch_size,
    validation_split=val_split,
    subset='training',
    seed=rnd_seed
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels=label,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    validation_split=val_split,
    subset='validation',
    seed=rnd_seed
)

class TestDS(unittest.TestCase):

    def equal(self):
        '''Checks if training and valuation datasets have identical elements'''
        self.assertEqual(train_ds.element_spec, val_ds.element_spec)

#build model (includes rescalling of pixels to 0-1 from 0-255)
model = Sequential()
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

#Early stop to avoid overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=2
)

#Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=600,
    callbacks=[early_stop],
    verbose=1
)

#Save model
model.save('model.h5', overwrite=True)

#Output plot of loss for training and validation
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.title('Loss - Binary Cross Entropy')
plt.xlabel('epochs')
plt.show()

#Evaluate model
print('\nTesting validation model...')
test_score = model.evaluate(val_ds, verbose=0)
print('validation dataset prediction loss (binary cross entroy):', test_score)
