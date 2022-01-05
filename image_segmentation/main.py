import tensorflow as tf 
import numpy as np
import sys
import matplotlib.pyplot as plt


def load_model():
    return tf.keras.models.load_model('./image_segmentation.h5')

def preprocess(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.image.resize(img,(96,128),method='nearest')
    return img

def create_mask(pred):
    mask = tf.argmax(pred,axis=-1)
    mask = mask[...,tf.newaxis]
    return mask 

def display(image,mask):
    plt.figure(figsize=(15,30))
    plt.subplot(1,2,1)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
    plt.show()
def segment():
    image = preprocess(sys.argv[1])
    model = load_model()
    pred_mask = model.predict(tf.expand_dims(image,axis=0))
    mask = create_mask(pred_mask)
    display(image,tf.squeeze(mask,axis=0))

segment()