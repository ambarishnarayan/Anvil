#!/usr/bin/env python
# coding: utf-8

# In[1]:

# !pip install tensorflow
import tensorflow as tf
import json 
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1

# Reshape the data to add a channel dimension (for grayscale images)
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

xshape = x_train.shape[1:4]
# # Build the CNN model
# model = models.Sequential()

# # Convolutional and pooling layers
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

# # Flatten layer to transition from convolutional to fully connected layers
# model.add(layers.Flatten())

# # Fully connected layers
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 units for 10 classes

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),activation=tf.nn.relu,input_shape=xshape),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides=2, padding='same'),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides=2, padding='same'),
        tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides=2, padding='same'),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation=tf.nn.relu),
        tf.keras.layers.Dense(32,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        ])


# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
history_val = model.fit(x_train, y_train, epochs=18, validation_split=0.2)
with open("history_val.json", "w") as fp:
	json.dump(history_val.history, fp)

# Train the model - full data 
history_all = model.fit(x_train, y_train, epochs=18)
with open("history_all.json", "w") as fp:
	json.dump(history_all.history, fp)
model.save("CNN4MNIST.keras")

	
# Test 
def test_evalute():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    loss, accuracy = model.evaluate(x_test,y_test)
    return "Accuracy on the test dataset using CNN is {}".format(accuracy)
test_evalute()






# %%
