import tensorflow
import keras
from keras.layers import Activation, Dense, Input, Lambda, LeakyReLU
from keras.layers import Conv2D, Flatten, AveragePooling2D, Add
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np
from instancenormalization import InstanceNormalization

def ConvBlock(x, n_filters, kernel_size, strides):
    x = Conv2D(filters = n_filters, kernel_size = kernel_size, strides = strides, padding = 'same', use_bias = False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    return x

def ConvTransposeBlock(x,n_filters):
    x = Conv2DTranspose(filters = n_filters, kernel_size = 4, strides = 2, padding = 'same', use_bias = False)(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    return x

def ResidualBlock(x, n_filters):
    y = ConvBlock(x, n_filters = n_filters, kernel_size = 3, strides = 1)
    y = Conv2D(filters = n_filters, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)(y)
    y = InstanceNormalization()(y)
    y = Add()([x,y])
    return Activation('relu')(x)

# Create and return generator model
def get_generator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 128):
    input_img = Input(shape = (im_size, im_size, 3 + n_labels))
    
    x = ConvBlock(input_img, n_filters = n_filters, kernel_size = 7, strides = 1)

    # Down-sampling layers
    curr_filters = n_filters
    for i in range(2):
        x = ConvBlock(x, n_filters = 2 * curr_filters, kernel_size = 4, strides = 2)
        curr_filters *= 2

    # Bottleneck layers
    for i in range(repeat_num):
        x = ResidualBlock(x, curr_filters)

    # Up-sampling layers
    for i in range(2):
        x = ConvTransposeBlock(x, curr_filters // 2)
        curr_filters //= 2

    x = Conv2D(filters = 3, kernel_size = 7, strides = 1, padding = 'same', use_bias = False)(x)
    out_im = Activation('tanh')(x) 

    generator = Model(input_img, out_im, name = 'generator')

    return generator

# Create and return discriminator model
def get_discriminator(n_filters = 64, n_labels = 5, repeat_num = 6, im_size = 128):
    input_img = Input(shape = (im_size, im_size, 3))
    x = Conv2D(filters = n_filters, kernel_size = 4, strides = 2, padding = 'same')(input_img)
    x = LeakyReLU(alpha = 0.01)(x)

    curr_filters = n_filters
    for i in range(1, repeat_num):
        curr_filters *= 2
        x = Conv2D(filters = curr_filters, kernel_size = 4, strides = 2, padding = 'same')(x)
        x = LeakyReLU(alpha = 0.01)(x)

    kernel_size = int(im_size / (2 ** repeat_num))

    out_src = Conv2D(1, kernel_size = 3, strides = 1, padding = 'same', use_bias = False)(x)
    out_src = Flatten()(out_src)
    
    out_cls = Conv2D(n_labels, kernel_size = kernel_size, padding = 'valid', use_bias = False, activation = 'sigmoid')(x)
    out_cls = Flatten()(out_cls)

    discriminator = Model(input_img, [out_src, out_cls], name = 'discriminator')

    return discriminator
