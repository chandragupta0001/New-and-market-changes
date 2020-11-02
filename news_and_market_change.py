import numpy as np

import pandas as pd

train = pd.read_csv('dataset/train.csv')

test = pd.read_csv('dataset/test.csv')
data=train
data["Date"]=data["Date"]+" "
data["joined"]=data["Date"]+data["Headline"].astype(str)+data["Article"].astype(str)
X=data["joined"]
y=data["market_change%"]
Y=y.values
import tensorflow as tf
from tensorflow import keras
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(output_sequence_length=1000)
d=tf.data.Dataset.from_tensor_slices(X)
vectorize_layer.adapt(d.batch(64))


class BagOfWords(keras.layers.Layer):
    def __init__(self, n_tokens, dtype=tf.int32, **kwargs):
        super().__init__(dtype=tf.int32, **kwargs)
        self.n_tokens = n_tokens
    def call(self, inputs):
        one_hot = tf.one_hot(inputs, self.n_tokens)
        return tf.reduce_sum(one_hot, axis=1)[:, 1:]

bagofwords=BagOfWords(n_tokens=1002)

from tensorflow.keras.layers import Dense

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(bagofwords)
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer="nadam",metrics=["accuracy"])

epochs = 20
batch_size = 32

history = model.fit(X, Y, epochs=epochs, batch_size=batch_size,validation_split=0.1)
p=model.predict(X[500:510])
print(p)
print(Y[500:502])


data=test
data["Date"]=data["Date"]+" "
data["joined"]=data["Date"]+data["Headline"].astype(str)+data["Article"].astype(str)
X_test=data["joined"]
submission=model.predict(X_test)
s=submission.shape[0]
submission =submission.reshape(s)
df = pd.DataFrame()
df["market_change%"]=submission
df["News_ID"]=data["News_ID"]
df.to_csv('submission.csv',index = False)