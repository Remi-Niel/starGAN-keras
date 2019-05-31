import tensorflow as tf
import numpy as np
from instancenormalization import InstanceNormalization

# Create and return generator model
def get_generator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 100):
    input_img = tf.keras.layers.Input(shape = (1,1,im_size))
    input_labels = tf.keras.layers.Input(shape=[1,1,n_labels])
    x = tf.keras.layers.Concatenate()([input_img,input_labels])
    x = tf.keras.layers.Reshape((1,1,im_size+n_labels))(x)

    x = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4,
                               strides=2, padding='VALID')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    depth = [256, 256, 128, 64]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4,
                               strides=2, padding='SAME')(x)
    x = tf.keras.layers.Activation('tanh')(x)

    generator = tf.keras.Model([input_img, input_labels], x, name = 'generator')

    # print(generator.summary())
    return generator

# Create and return discriminator model
def get_discriminator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 128):
    input_img = tf.keras.layers.Input(shape = (im_size, im_size, 3))
    input_labels = tf.keras.layers.Input(shape=[n_labels])
    labels = tf.keras.layers.Reshape((1,1,n_labels))(input_labels)
    labels = tf.keras.layers.Lambda(tf.keras.backend.tile, arguments = {'n': (1,64,64,1)})(labels)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4,
                               strides=2, padding='SAME')(input_img)    
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Concatenate()([x,labels])

    print(x.shape)
    depth = [128, 256, 256, 512]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)       
        print(x.shape) 

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4,
                               strides=1, padding='VALID', activation='sigmoid')(x)
    # x = tf.keras.layers.Activation('sigmoid')(x)
    # x = tf.keras.activations.sigmoid(x)

    discriminator = tf.keras.Model([input_img, input_labels], x, name = 'discriminator')
    print(discriminator.summary())
    return discriminator

def get_encoder_ez(n_labels = 5, im_size = 128):
    x = input_img = tf.keras.layers.Input(shape = (im_size, im_size, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='SAME')(x)

    depth = [32,64,128,256]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=5,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=100)(x)

    encoder_ez = tf.keras.Model([input_img], x, name = 'encoder_ez')

    return encoder_ez


def get_encoder_ey(n_labels = 5, im_size = 128):
    x = input_img = tf.keras.layers.Input(shape = (im_size, im_size, 3))


    depth = [32,64,128,256]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=5,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=n_labels)(x)

    encoder_ey = tf.keras.Model([input_img], x, name = 'encoder_ey')

    return encoder_ey