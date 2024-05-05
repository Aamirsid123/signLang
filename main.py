import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                     Flatten, Conv2D, GlobalAveragePooling2D,
                                     MaxPooling2D, BatchNormalization)
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

data = tf.keras.utils.image_dataset_from_directory(
    os.path.join('Photo', 'archive (4)', 'asl_alphabet_train', 'asl_alphabet_train'),
    batch_size=128,
    image_size=(64, 64),
    color_mode= 'grayscale',
    shuffle= True,
    label_mode= 'categorical'
)


# Map the lambda function to the dataset


train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = len(data) - train_size - val_size

train_images = data.take(train_size)
val_images = data.skip(train_size).take(val_size)
test_images = data.skip(train_size + val_size).take(test_size)

# Calculate the number of classes directly from the dataset
num_classes = len(data.class_names)

model = Sequential([
    Input(shape=(64, 64, 1), name='input_layer'),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax', name='output_layer')
])

model.compile(
    optimizer=Adamax(learning_rate=1e-3),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

model_es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_images, validation_data=val_images, epochs=30, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='Accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_Accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test_images.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
model.save(os.path.join('model', 'model.keras'))
