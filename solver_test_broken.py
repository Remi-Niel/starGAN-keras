from model import get_generator, get_discriminator
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
import numpy as np
import os
import time
import random
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import trange
from functools import partial

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class Solver(object):

    def __init__(self, celeba_loader, config):
        self.data_loader = celeba_loader
        self.n_labels = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.g_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta_1 = config.beta1
        self.beta_2 = config.beta2 

        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.n_critic = config.n_critic
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        self.test_iters = config.test_iters
        self.use_tensorboard = config.use_tensorboard

        self.log_dir = "stargan/logs/"
        self.sample_dir = "stargan/samples/"
        self.model_save_dir = "stargan/models/"
        self.result_dir = "stargan/results/"

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

    def wasserstein_loss(self, y_true, y_pred):
        """Calculates the Wasserstein loss for a sample batch.
        The Wasserstein loss function is very simple to calculate. In a standard GAN, the
        discriminator has a sigmoid output, representing the probability that samples are
        real or generated. In Wasserstein GANs, however, the output is linear with no
        activation function! Instead of being constrained to [0, 1], the discriminator wants
        to make the distance between its output for real and generated samples as
        large as possible.
        The most natural way to achieve this is to label generated samples -1 and real
        samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
        outputs by the labels will give you the loss immediately.
        Note that the nature of this loss means that it can be (and frequently will be)
        less than 0."""
        return keras.backend.mean(y_true * y_pred)


    def build_model(self):
        self.G = get_generator(self.g_conv_dim, self.n_labels, self.g_repeat_num, self.image_size)
        input_img, out_src, out_cls = get_discriminator(self.d_conv_dim, self.n_labels, self.d_repeat_num, self.image_size)

        self.D_a = Model(inputs = [input_img], outputs = [out_src, out_cls])

        concatted_input = Input(shape=(self.image_size,self.image_size,3+self.n_labels))
        gen_samples = self.G(concatted_input)

        weighted_avg = RandomWeightedAverage()
        weighted_avg.BATCH_SIZE=self.batch_size
        averaged_samples = weighted_avg([input_img, gen_samples])
        averaged_samples_out = self.D_a(averaged_samples)[0]

        print(averaged_samples_out)
        print(averaged_samples_out.shape)

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = averaged_samples)
        partial_gp_loss.__name__ = 'gradient_penalty'


        self.d_optimizer = keras.optimizers.Adam(lr = self.d_lr, beta_1 = self.beta_1, beta_2 = self.beta_2, decay = 1.0/self.num_iters_decay/5)
        self.g_optimizer = keras.optimizers.Adam(lr = self.g_lr, beta_1 = self.beta_2, beta_2 = self.beta_2, decay = 1.0/self.num_iters_decay/5)


        self.D = Model(inputs =[input_img, concatted_input], outputs=[out_src, out_cls, averaged_samples_out])

        self.D.compile(loss=[self.wasserstein_loss, "binary_crossentropy", partial_gp_loss], loss_weights = [1, self.lambda_cls, self.lambda_gp], optimizer= self.d_optimizer)

        self.D.trainable = False

        input_img = Input(shape = (self.image_size, self.image_size, 3 + self.n_labels))

        reconstr_img = self.G(input_img)
        output_D     = self.D_a(reconstr_img)


        self.combined = Model(inputs = [input_img], outputs = [reconstr_img] + output_D)

        self.combined.compile(loss = ["mae", self.wasserstein_loss, "binary_crossentropy"], loss_weights = [self.lambda_rec, -1, self.lambda_cls], optimizer = self.g_optimizer)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = self.batch_size
        out = np.zeros((batch_size, dim))
        out[np.arange(batch_size), labels.astype(np.int_)] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.copy()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg)
        return c_trg_list

    def denorm(self, x):
        out = (x + 1) / 2
        return np.clip(out,0, 1)


    def train(self):
        self.writer = tf.summary.FileWriter(self.log_dir)

        callbacks = [keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False),
                     keras.callbacks.ModelCheckpoint(self.model_save_dir + "weights.{epoch:03d}.hdf5", verbose = 1, period = 5)]

        data_iter = iter(self.data_loader)
        
        test_imgs, label_test = next(data_iter)
        test_imgs = np.tile(test_imgs, (5,1,1,1))
        c_fixed = np.asarray(self.create_labels(label_test, self.n_labels, self.data_loader, self.selected_attrs))
        c_fixed = np.concatenate(c_fixed, axis = 0)
        labels_fixed = c_fixed.reshape((5 * self.batch_size, 1, 1, 5))
        label_test = np.tile(label_test, (5,1))
        test_imgs_concatted = np.concatenate((test_imgs, np.tile(labels_fixed, (1,self.image_size, self.image_size,1))), axis=3)

        dummy_y = np.zeros((self.batch_size * 2, 4), dtype=np.float32)

        for epoch in trange(0,self.num_iters//self.log_step//5):
            with keras.backend.get_session().as_default():

                outcome = self.G.predict(test_imgs_concatted)
                s = BytesIO()

                left = self.denorm(test_imgs[epoch%80].reshape((128,128,3)))
                right = self.denorm(outcome[epoch%80].reshape((128,128,3)))

                total = np.concatenate((left,right),axis = 1)
                print(total.shape)
                plt.imsave(s, total)
                out = tf.Summary.Image(encoded_image_string = s.getvalue())
                print(label_test.shape)
                print(labels_fixed.shape)
                labels = np.concatenate((label_test[epoch%80].reshape((1,self.n_labels)),labels_fixed[epoch%80].reshape((1,self.n_labels))))
                s = BytesIO()
                plt.imsave(s, labels)
                labels = tf.Summary.Image(encoded_image_string = s.getvalue())

                summary = tf.Summary(value=[tf.Summary.Value(tag = "In->Out", image = out),
                                            tf.Summary.Value(tag = "Labels", image = labels)])
                self.writer.add_summary(summary, epoch)


            d_loss_r = 0
            d_loss_f = 0
            for i in trange(0, self.log_step):
                for j in range(0,5):
                    try:
                        x_real, label_org = next(data_iter)
                    except:
                        data_iter = iter(self.data_loader)
                        x_real, label_org = next(data_iter)

                    label_trg = label_org[np.random.permutation(label_org.shape[0])]

                    c_org = label_org.copy()
                    c_trg = label_trg.copy()

                    labels_trg = c_trg.reshape((self.batch_size,1,1,5))
                    x_concatted = np.concatenate((x_real, np.tile(labels_trg, (1,self.image_size, self.image_size,1))), axis=3)


                    x_fake = self.G.predict(x_concatted)
                    x_concatted_fake = np.concatenate((x_fake, np.tile(labels_trg, (1,self.image_size, self.image_size,1))), axis=3)


                    fake = np.zeros(self.batch_size)
                    real = np.ones(self.batch_size)
                    concatted_bool = np.concatenate((fake,real))
                    concatted_labels = np.concatenate((c_trg,c_org))
                    concatted_imgs = np.concatenate((x_fake, x_real))
                    concatted_gen_input = np.concatenate((x_concatted_fake,x_concatted))

                    d_loss_r = self.D.train_on_batch([concatted_imgs, concatted_gen_input], [np.tile(concatted_bool.reshape(self.batch_size*2,1),(1,4)), concatted_labels, dummy_y])

                g_loss = self.combined.train_on_batch(x_concatted, [x_real, np.tile(fake.reshape(self.batch_size,1),(1,4)), c_trg])

            



