from model import get_generator, get_discriminator
import tensorflow as tf
import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Concatenate
import numpy as np
import os
import time
import random
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import trange
from keras.layers.merge import _Merge
import keras.backend as K

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output-inputs[i]
        return output

def mean_loss(y_true, y_pred):
    return K.mean(y_pred)

def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

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

    #http://shaofanlai.com/post/10
    def build_model(self):
        self.G = get_generator(self.g_conv_dim, self.n_labels, self.g_repeat_num, self.image_size)
        self.D = get_discriminator(self.d_conv_dim, self.n_labels, self.d_repeat_num, self.image_size)

        self.d_optimizer = keras.optimizers.Adam(lr = self.d_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        self.g_optimizer = keras.optimizers.Adam(lr = self.g_lr, beta_1 = self.beta_2, beta_2 = self.beta_2)

        self.D.trainable = False

        input_img = Input(shape = (self.image_size, self.image_size, 3))
        input_orig_labels = Input(shape = (self.image_size, self.image_size, self.n_labels))
        input_target_labels = Input(shape = (self.image_size, self.image_size, self.n_labels))

        concatted_input = Concatenate(axis=3)([input_img, input_target_labels])

        fake_img = self.G(concatted_input)
        concatted_fake_img = Concatenate(axis=3)([fake_img,input_orig_labels])
        reconstr_img = self.G(concatted_fake_img)

        output_D     = self.D(fake_img)

        self.combined = Model(inputs = [input_img, input_orig_labels, input_target_labels], outputs = [reconstr_img] + output_D)

        self.combined.compile(loss = ["mae", "binary_crossentropy", "binary_crossentropy"], loss_weights = [self.lambda_rec, -1, self.lambda_cls], optimizer = self.g_optimizer)

        shape = (self.image_size,self.image_size,3)
        gen_input, interpolation = Input(shape), Input(shape), Input(shape)
        norm = GradNorm()([self.D(interpolation)[0], interpolation])
        output_D = self.D(gen_input)
        self.dis2batch = Model([gen_input, real_input, interpolation], output_D + [norm])
        # self.dis2batch = Model([gen_input], output_D)

        self.D.trainable = True

        self.dis2batch.compile(loss=["binary_crossentropy", "binary_crossentropy", 'mse'], loss_weights = [1, self.lambda_cls, self.lambda_gp], optimizer= self.d_optimizer)

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

        callback = keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False)
        callback.set_model(self.combined)
        dis_names = ['Discriminator Classification loss', 'Discriminator Adversarial loss', 'Gradient Penalty']
        gen_names = ['Cycle loss', 'Generator Adversarial loss', 'Generator Classification loss']

        data_iter = iter(self.data_loader)
        
        test_imgs, label_test = next(data_iter)
        test_imgs = np.tile(test_imgs, (5,1,1,1))
        c_fixed = np.asarray(self.create_labels(label_test, self.n_labels, self.data_loader, self.selected_attrs))
        c_fixed = np.concatenate(c_fixed, axis = 0)
        labels_fixed = c_fixed.reshape((5 * self.batch_size, 1, 1, 5))
        label_test = np.tile(label_test, (5,1))
        test_imgs_concatted = np.concatenate((test_imgs, np.tile(labels_fixed, (1,self.image_size, self.image_size,1))), axis=3)


        batch_id = 0
        for epoch in trange(0,self.num_iters//self.log_step//5):
            with keras.backend.get_session().as_default():

                outcome = self.G.predict(test_imgs_concatted)
                tmp = np.concatenate((outcome, np.tile(label_test.reshape((5*self.batch_size,1,1,5)),(1,self.image_size,self.image_size,1))),axis=3)
                cycled = self.G.predict(tmp)
                s = BytesIO()

                left = self.denorm(test_imgs[epoch%80].reshape((128,128,3)))
                right = self.denorm(outcome[epoch%80].reshape((128,128,3)))
                total = np.concatenate((left,right),axis = 1)

                right = self.denorm(cycled[epoch%80].reshape((128,128,3)))
                total = np.concatenate((total,right), axis = 1)

                plt.imsave(s, total)
                out = tf.Summary.Image(encoded_image_string = s.getvalue())
                labels = np.concatenate((label_test[epoch%80].reshape((1,self.n_labels)),labels_fixed[epoch%80].reshape((1,self.n_labels))))
                s = BytesIO()
                plt.imsave(s, labels)
                labels = tf.Summary.Image(encoded_image_string = s.getvalue())

                summary = tf.Summary(value=[tf.Summary.Value(tag = "In->Out->Cycled", image = out),
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

                    label_trg = np.flip(label_org, axis=0)

                    c_org = label_org.copy()
                    c_trg = label_trg.copy()

                    labels_trg = c_trg.reshape((self.batch_size,1,1,5))
                    x_concatted = np.concatenate((x_real, np.tile(labels_trg, (1,self.image_size, self.image_size,1))), axis=3)


                    x_fake = self.G.predict(x_concatted)


                    fake = np.zeros(self.batch_size)
                    real = np.ones(self.batch_size)
                    concatted_bool = np.concatenate((fake,real))
                    concatted_labels = np.concatenate((c_trg,c_org))
                    concatted_imgs = np.concatenate((x_fake, x_real))

                    concatted_labels_inv = np.flip(concatted_labels, axis = 0)

                    concatted_fake_imgs = self.G.predict(np.concatenate((concatted_imgs, 
                                                                            np.tile(concatted_labels_inv.reshape(self.batch_size*2,1,1,5), 
                                                                                        (1,self.image_size, self.image_size,1))
                                                                                    ), 
                                                                        axis = 3
                                                                        )
                                                        )

                    epsilon = np.random.uniform(0, 1, size = (2 * self.batch_size,1,1,1))
                    interpolation = epsilon * concatted_imgs + (1-epsilon) * concatted_fake_imgs
                    d_logs = self.dis2batch.train_on_batch([concatted_fake_imgs, interpolation], [concatted_labels, np.tile(concatted_bool.reshape(self.batch_size*2,1),(1,4)), np.ones(self.batch_size * 2)])
                    write_log(callback, dis_names, d_logs[1:4], batch_id)
                    batch_id += 1

                tiled_label_org = np.tile(label_org.reshape(self.batch_size,1,1,5),(1,self.image_size,self.image_size,1))
                tiled_label_trg = np.tile(label_trg.reshape(self.batch_size,1,1,5),(1,self.image_size,self.image_size,1))
                g_logs = self.combined.train_on_batch([x_real, tiled_label_org, tiled_label_trg], [x_real, np.tile(fake.reshape(self.batch_size,1),(1,4)), c_trg])
                write_log(callback, gen_names, g_logs[1:4], batch_id)







            



