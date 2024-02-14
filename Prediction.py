
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
import anvil.server


# Load the pre-trained CNN model
basicCNN = tf.keras.models.load_model("CNN4MNIST.keras")
ViT = tf.keras.models.load_model("ViT4MNIST.keras")


# Test
@anvil.server.callable
def test_evalute_CNN():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    loss, accuracy = basicCNN.evaluate(x_test,y_test)
    return "Accuracy on the test dataset using CNN is  {}".format(round(accuracy,4))
print(test_evalute_CNN())

@anvil.server.callable
def test_evalute_ViT():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    loss, accuracy = ViT.evaluate(x_test,y_test)
    return "Accuracy on the test dataset using ViT is  {}".format(round(accuracy,4))
print(test_evalute_ViT())






@anvil.server.callable
def convert_image(file):
    try:
      file = file.get_bytes()
      im_df = pd.read_csv(BytesIO(file), header=None)
      im_df = np.array(im_df)

      if im_df.shape != (28, 28):
        raise ValueError(f"Expected file of shape(28, 28), but recieved with shape {im_df.shape}")
      
      max_val = np.max(im_df)
      if max_val > 1:
        im_df = im_df/255.0
      
      image = Image.fromarray(im_df)
      image = image.convert("L")
      bs = BytesIO()
      image.save(bs, format='JPEG')

      return anvil.BlobMedia("image/jpeg", bs.getvalue(), name='input')
    except Error as e:
      return e
    


@anvil.server.callable
def predict(model: str):
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
        val = basicCNN.predict(im_df)
       elif model=='ViT':
        df = np.expand_dims(im_df, axis=0)
        df =  ViT.preprocess_data(df)[0]
        val = ViT.predict(df)
       else:
        raise ValueError(f"No model found with name {model}")
       return val
    except Error as e:
      return e

anvil.server.connect("")
anvil.server.wait_forever()