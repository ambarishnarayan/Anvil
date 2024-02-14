

# !pip install tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json 
import keras
# %matplotlib inline

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to the [0, 1] range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the images for the Transformer model
x_train_flatten = x_train.reshape((-1, 28*28))  # Flattening the 28x28 images
x_test_flatten = x_test.reshape((-1, 28*28))


n = 7
m = 7
block_size = 16
hidden_dim = 128
num_layers = 4
num_heads = 8
key_dim = 16
mlp_dim = 128
dropout_rate = 0.1
num_classes = 10
ndata_train, ndata_test = x_train.shape[0], x_test.shape[0]


# Initialize arrays for the transformed dataset
x_train_ravel = np.zeros((ndata_train, n*m, block_size))
x_test_ravel = np.zeros((ndata_test, n*m, block_size))

# Transform training data
for img in range(ndata_train):
    ind = 0
    for row in range(n):
        for col in range(m):
            x_train_ravel[img, ind, :] = x_train[img, (row*4):((row+1)*4), (col*4):((col+1)*4)].ravel()
            ind += 1

# Transform test data
for img in range(ndata_test):
    ind = 0
    for row in range(n):
        for col in range(m):
            x_test_ravel[img, ind, :] = x_test[img, (row*4):((row+1)*4), (col*4):((col+1)*4)].ravel()
            ind += 1

class ClassToken(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_token = None

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=(1, 1, input_shape[-1]),  # Adding a class token for each feature dimension
            initializer='zeros',
            trainable=True,
            name='class_token'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token_broadcasted = tf.tile(self.class_token, [batch_size, 1, 1])
        return tf.keras.layers.Concatenate(axis=1)([class_token_broadcasted, inputs])  # Prepending the class token

def build_ViT(n, m, block_size, hidden_dim, num_layers, num_heads, key_dim, mlp_dim, dropout_rate, num_classes):
    # Image input
    inp = tf.keras.layers.Input(shape=(n*m, block_size))
    # Positional encoding input
    pos_inp = tf.keras.layers.Input(shape=(n*m,))

    # Initial transformation
    mid = tf.keras.layers.Dense(hidden_dim)(inp)

    # Use positional encoding input
    emb = tf.keras.layers.Embedding(input_dim=n*m, output_dim=hidden_dim)(pos_inp)
    mid = mid + emb  # Add positional embeddings

    # Append class token
    token = ClassToken()(mid)

    # Transformer blocks
    for _ in range(num_layers):
        ln = tf.keras.layers.LayerNormalization()(token)
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ln, ln, ln)
        add = tf.keras.layers.Add()([token, mha])  # Add skip connection
        ln = tf.keras.layers.LayerNormalization()(add)
        den = tf.keras.layers.Dense(mlp_dim, activation='gelu')(ln)
        den = tf.keras.layers.Dropout(dropout_rate)(den)
        den = tf.keras.layers.Dense(hidden_dim)(den)
        den = tf.keras.layers.Dropout(dropout_rate)(den)
        token = tf.keras.layers.Add()([den, add])  # Add skip connection

    # Classification head
    ln = tf.keras.layers.LayerNormalization()(token)
    fl = ln[:, 0, :]  # Use the class token
    clas = tf.keras.layers.Dense(num_classes, activation='softmax')(fl)

    # Compile the model
    mod = tf.keras.models.Model(inputs=[inp, pos_inp], outputs=clas)
    mod.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return mod


# Build the Vision Transformer model
model_Vit = build_ViT(n,m,block_size,hidden_dim,num_layers,num_heads,key_dim,mlp_dim,dropout_rate,num_classes)


model_Vit.summary()
x_train.shape


pos_feed_train = np.array([list(range(n*m))]*ndata_train)
pos_feed_test = np.array([list(range(n*m))]*ndata_test)


# Train the model with a potentially larger batch size and more epochs
history_val = model_Vit.fit([x_train_ravel,pos_feed_train], y_train, epochs=100, batch_size=2000, validation_split=0.2)
with open("history_val_Vit.json", "w") as fp:
	json.dump(history_val.history, fp)


# Train the model - full data 
history_all = model_Vit.fit([x_train_ravel,pos_feed_train], y_train, epochs=100, batch_size=2000)
with open("history_all.json", "w") as fp:
	json.dump(history_all.history, fp)
     

model_Vit.save("ViT4MNIST.keras")

# Test 
def test_evalute():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    loss, accuracy = model_Vit.evaluate(x_test,y_test)
    return "Accuracy on the test dataset using ViT is {} ".format(accuracy)
test_evalute()