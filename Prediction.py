
# ?!pip install tensorflow
import tensorflow as tf
import json 
from tensorflow.keras import layers, models
import numpy as np
from keras.preprocessing import image
from csv import Error
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np
import json
from io import BytesIO
from PIL import Image
from typing import Union
# from model import VisionTrnasformer
import anvil.server

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
        
# Load the pre-trained CNN model
basicCNN = tf.keras.models.load_model("CNN4MNIST.keras")
ViT =  tf.keras.models.load_model("ViT4MNIST.keras", custom_objects={'ClassToken': ClassToken})

def preprocess_data(data, patch_rows: Union[int, None], patch_columns: Union[int, None]):
    if type(data)!=np.ndarray:
        data = data.numpy()
    
    flatten_images = np.zeros((data.shape[0],patch_rows*patch_columns,
                               int((data.shape[1]*data.shape[1])/(patch_rows*patch_columns))))
    helper = int(data.shape[1]/patch_rows)
    for i in range(data.shape[0]):
        ind = 0
        for row in range(patch_rows):
            for col in range(patch_columns):
                flatten_images[i,ind,:] = data[i,
                                               (row*helper):((row+1)*helper),
                                               (col*helper):((col+1)*helper)].ravel()
                ind += 1
    return tf.convert_to_tensor(flatten_images)


@anvil.server.callable
def convert_image(file):
    print("conv image")
    try:
      file = file.get_bytes()
      im_df = pd.read_csv(BytesIO(file), header=None)
      im_df = np.array(im_df)

      if im_df.shape != (28, 28):
        raise ValueError(f"Expected file of shape(28, 28), but recieved with shape {im_df.shape}")
      
      max_val = np.max(im_df)
      if max_val <= 1:
        im_df = im_df*255.0
      
      image = Image.fromarray(im_df)
      image = image.convert("L")
      bs = BytesIO()
      image.save(bs, format='JPEG')

      return anvil.BlobMedia("image/jpeg", bs.getvalue(), name='input')
    except Error as e:
      return e
    


@anvil.server.callable
def predict(model: str, file):
    try:
       file = file.get_bytes()
       im_df = pd.read_csv(BytesIO(file), header=None)
       im_df = np.array(im_df)
       if im_df.shape != (28, 28):
          raise ValueError(f"Expected file of shape(28, 28), but recieved with shape {im_df.shape}")
       max_val = np.max(im_df)
       if max_val > 1:
        im_df = im_df/255.0
       if model=='basicCNN':
        print("dfdf")
        im_df = np.expand_dims(im_df, axis=0)
        print(im_df.shape)
        x = tf.convert_to_tensor(im_df)
        val = basicCNN.predict(x)
       elif model=='ViT':
           print("vit")
           df = np.expand_dims(im_df, axis=0)
           df = preprocess_data(df, 7, 7)[0]
           pos_feed = np.array(49)
           df = np.expand_dims(df, axis=0)
           pos_feed = np.expand_dims(pos_feed, axis=0)
           print(df.shape, pos_feed.shape)
           val = ViT.predict([df, pos_feed])
           # df = vit.preprocess_data(im_df)[0]
           # val = vit.predict(df)
       else:
        raise ValueError(f"No model found with name {model}")
       print(val)
       return np.argmax(val)
    except Error as e:
      return e

anvil.server.connect("server_A3XFL2BX57XSEH6BD57OEOYI-2HI3P2QB3HI672CB")
anvil.server.wait_forever()
