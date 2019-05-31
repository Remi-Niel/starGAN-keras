from icgan import get_generator, get_discriminator, get_encoder_ez, get_encoder_ey
import tensorflow as tf
import keras
from keras.engine.topology import Layer
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
import pickle
import cv2

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

def neg_mean_loss(y_true, y_pred):
    return tf.math.scalar_mul(-1,K.mean(y_pred))

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


    def custom_bin(self, y_true, y_pred):
        return tf.divide(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred),self.n_labels)

    def isdir(self):
        i=1;
        while(os.path.isdir(self.model_dir)):
            self.model_dir = self.model_save_dir + self.model_sub_dir + '_' + str(i)
            i += 1
        os.makedirs(self.model_dir)

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
        self.model_sub_dir = config.model_save_dir
        self.model_dir = self.model_save_dir + self.model_sub_dir
        self.restore_epoch = config.restore_epoch
        self.result_dir = "stargan/results/"

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()
        if self.restore_epoch == 0:
            self.isdir()
            self.model_dir += '/'
        else:
            self.model_dir += '/'
            self.combined.load_weights(self.model_dir + "combined_weights" + str(self.restore_epoch) + ".h5")
            self.restore_optimizer(self.combined, "combined")
            self.restore_optimizer(self.DIS, "DIS")


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
        self.E = get_encoder_ez(self.n_labels, self.image_size)
        self.G = get_generator(self.g_conv_dim, self.n_labels, self.g_repeat_num, 100)
        self.D = get_discriminator(self.d_conv_dim, self.n_labels, self.d_repeat_num, 128)

        self.d_optimizer = tf.keras.optimizers.Adam(lr = self.d_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        self.g_optimizer = tf.keras.optimizers.Adam(lr = self.g_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        self.e_optimizer = tf.keras.optimizers.Adam(lr = self.d_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)


        self.D.compile(loss='binary_crossentropy', optimizer=self.d_optimizer) # make discriminator
        self.Ez.compile(loss='binary_crossentropy', optimizer=self.ez_optimizer)
        self.Ey.compile(loss='binary_crossentropy', optimizer=self.ey_optimizer)


        img = tf.keras.layers.Input((self.image_size, self.image_size, 3))
        noise = tf.keras.layers.Input((1,1,100))
        orig_labels = tf.keras.layers.Input([self.n_labels])
        target_labels = tf.keras.layers.Input([1,1,self.n_labels])

        fake_image = self.G([noise, target_labels]) # fake image
        self.D.trainable = False
        
        [ez_output,ey_output] = self.E(img) # this gives latent space z and image label y
        
        fake_image_E = self.G([ez_output, ey_output]) 

        [ez_output_rec, ey_output_rec] = self.E(fake_image) # reconstructed image labels

        output_cls = self.D([fake_image, target_labels]) # discriminator output fake image

        self.gan = tf.keras.Model(inputs = [noise, target_labels], outputs = [output_cls])
        self.gan.compile(loss=['binary_crossentropy'],optimizer=self.g_optimizer)
        self.gan.summary()






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

    def store_optimizer(self, model, name):
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(self.model_dir + name + str(self.restore_epoch)+ "_optimizer.h5",'wb') as f:
            pickle.dump(weight_values, f)
            

    def restore_optimizer(self, model, name):
        model._make_train_function()
        with open(self.model_dir + name + str(self.restore_epoch)+ "_optimizer.h5",'rb') as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)




    def train(self):
        callback = tf.keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False)
        callback.set_model(self.gan)

        data_iter = iter(self.data_loader)
        test_imgs, label_test = next(data_iter)
        c_fixed = np.asarray(self.create_labels(label_test, self.n_labels, self.data_loader, self.selected_attrs))
        c_fixed = np.concatenate(c_fixed, axis = 0)
        labels_fixed = c_fixed.reshape((5 * self.batch_size, 1, 1, 5))
        print(labels_fixed.shape)
        print(test_imgs.shape)


        batch_id = 0
        start = 0
        epochs = 10000
        for epoch in trange(start,epochs):               

            with tf.keras.backend.get_session().as_default():
                noise = np.random.uniform(-1., 1., size=[self.batch_size,1,1,100])
                # noise = noise.reshape(self.batch_size,1,1,100)
                f_labels = np.random.normal(0,1,(self.batch_size,1,1,5))
                outcome = self.denorm(self.G.predict([noise,f_labels]))
                s = BytesIO()
                plt.imsave(s, outcome[0])
                out = tf.Summary.Image(encoded_image_string = s.getvalue())
                summary = tf.Summary(value=[tf.Summary.Value(tag = "In->Out->Cycled", image = out)])
                callback.writer.add_summary(summary, epoch)
                callback.writer.flush() 


            for i in trange(0, self.log_step):
                for j in range(0,5):
                    try:
                        x_real, label_org = next(data_iter)
                    except:
                        data_iter = iter(self.data_loader)
                        x_real, label_org = next(data_iter)  


                    label_trg = np.flip(label_org, axis=0)

                    noise = np.random.uniform(-1., 1., size=[self.batch_size,1,1,100])
                    label_trg_ = label_trg.reshape(self.batch_size,1,1,self.n_labels)
                    x_fake = self.G.predict([noise, label_trg_])

                    d_real_labels = np.ones([self.batch_size,1])
                    d_fake_labels = np.zeros([self.batch_size,1])

                    d_loss_real = self.D.train_on_batch([x_real, label_org],[d_real_labels])
                    d_loss_fake = self.D.train_on_batch([x_fake, label_trg],[d_fake_labels])
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    batch_id += 1
                    
                    g_labels = np.ones([self.batch_size,1])
                    g_loss = self.gan.train_on_batch([noise, label_trg_],[g_labels])

                    write_log(callback, ['d_loss'], [d_loss], batch_id)
                    write_log(callback, ['g_loss'], [g_loss], batch_id)







            



