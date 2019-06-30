import tensorflow as tf
import numpy as np
from instancenormalization import InstanceNormalization

# Create and return generator model
def get_generator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 400):
    input_img = tf.keras.layers.Input(shape = (1,1,im_size))
    input_img_ = tf.keras.layers.Reshape((1,1,im_size))(input_img)
    input_labels = tf.keras.layers.Input(shape=[1,1,n_labels])
    x = tf.keras.layers.Concatenate()([input_img_,input_labels])
    x = tf.keras.layers.Reshape((1,1,im_size+n_labels))(x)

    x = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=4,
                               strides=2, padding='VALID')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    depth = [512, 256, 256, 128]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4,
                               strides=2, padding='SAME')(x)
    x = tf.keras.layers.Activation('tanh')(x)

    generator = tf.keras.Model([input_img, input_labels], x, name = 'generator')

    print(generator.summary())
    return generator

# Create and return discriminator model
def get_discriminator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 128):
    input_img = tf.keras.layers.Input(shape = (im_size, im_size, 3))
    input_labels = tf.keras.layers.Input(shape=[n_labels])
    labels = tf.keras.layers.Reshape((1,1,n_labels))(input_labels)
    labels = tf.keras.layers.Lambda(tf.keras.backend.tile, arguments = {'n': (1,64,64,1)})(labels)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
                               strides=2, padding='SAME')(input_img)    
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Concatenate()([x,labels])

    depth = [64, 256, 256, 512]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)       

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4,
                               strides=1, padding='VALID', activation='sigmoid')(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Activation('sigmoid')(x)
    # x = tf.keras.activations.sigmoid(x)

    discriminator = tf.keras.Model([input_img, input_labels], x, name = 'discriminator')
    # print(discriminator.summary())
    return discriminator


def get_encoder_comb(n_labels = 5, im_size = 128, output_size = 400):
    x = input_img = tf.keras.layers.Input(shape = (im_size, im_size, 3))


    depth = [32,64,128,256,256]
    for idx, filters in enumerate(depth):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=5,
                                   strides=2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 

    # x = tf.keras.layers.Flatten()(x)

    # ez = tf.keras.layers.Dense(units=1024)(x)
    ez = tf.keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='SAME')(x)
    ez = tf.keras.layers.BatchNormalization()(ez)
    ez = tf.keras.layers.ReLU()(ez)
    ez = tf.keras.layers.Conv2D(filters=output_size, kernel_size=2, strides=1, padding='VALID')(ez)
    ez = tf.keras.layers.Flatten()(ez)
    # ez = tf.keras.layers.Dense(units=output_size)(ez)

    # ey = tf.keras.layers.Dense(units=512)(x)
    ey = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='SAME')(x)
    ey = tf.keras.layers.BatchNormalization()(ey)
    ey = tf.keras.layers.ReLU()(ey)
    ey = tf.keras.layers.Conv2D(filters=n_labels, kernel_size=2, strides=1, padding='VALID')(ey)
    ey = tf.keras.layers.Flatten()(ey)
    # ey = tf.keras.layers.Dense(units=n_labels)(ey)


    encoder_comb = tf.keras.Model([input_img], [ez,ey], name = 'encoder_comb')
    print(encoder_comb.summary())
    return encoder_comb
