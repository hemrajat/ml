import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("/Users/hemraj/coding/datasets/mnist/mnist_train.csv")
x = data.drop("label",axis=1)
y = data["label"]
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.1,random_state=42)
train_x = tf.cast(tf.constant(train_x),tf.float32)
train_y = tf.cast(tf.constant(train_y),tf.int32)
test_x = tf.cast(tf.constant(test_x),tf.float32)
test_y = tf.cast(tf.constant(test_y),tf.int32)

train_x = tf.reshape(train_x,shape=(-1,28,28,1))
test_x = tf.reshape(test_x,shape=(-1,28,28,1))

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2),
    tf.keras.layers.Conv2D(12,(3,3),strides=1,activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(100,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])
model.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(),
             metrics=["accuracy"])

model.summary()

history = model.fit(train_x,train_y,epochs=100)