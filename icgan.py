import tensorflow
import keras
from keras.layers import Activation, Dense, Input, Lambda, LeakyReLU
from keras.layers import Conv2D, Flatten, AveragePooling2D, Add
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np
from instancenormalization import InstanceNormalization

# Create and return generator model
def get_generator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 100):
    input_img = Input(shape = (im_size, im_size, 3))
    input_labels = Input(shape=[n_labels])
    x = Concatenate()([input_img,input_labels])

    depth = [512, 256, 128, 64]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4,
                               strides=2, padding='SAME')(x)
    x = tf.keras.layers.Activation('tanh')(x)

    generator = Model(input = [input_img, input_labels], outputs = x, name = 'generator')

    return generator

# Create and return discriminator model
def get_discriminator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 64):
    input_img = Input(shape = (im_size, im_size, 3))
    input_labels = Input(shape=[n_labels])

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
                               strides=2, padding='SAME')(input_img)    
    x = tf.keras.layers.LeakyReLU()(x)
    x = Concatenate()([x,input_labels])

    depth = [128,256,512]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)        

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4,
                               strides=1, padding='SAME')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    discriminator = Model(input = [input_img, input_labels], outputs = x, name = 'discriminator')

    return discriminator

def get_encoder_ez(n_labels = 5, im_size = 128):
    x = input_img = Input(shape = (im_size, im_size, 3))


    depth = [32,64,128,256]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=100)(x)

    encoder_ez = Model([input_img], x, name = 'encoder_ez')

    return encoder_ez


def get_encoder_ey(n_labels = 5, im_size = 128):
    x = input_img = Input(shape = (im_size, im_size, 3))


    depth = [32,64,128,256]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=n_labels)(x)

    encoder_ez = Model([input_img], x, name = 'encoder_ez')

    return encoder_ey