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
import pickle
from PIL import Image


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

def ls_loss(y_true, y_pred):
    return tf.reduce_mean(tf.squared_difference(y_true, y_pred))


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
        grad_norm = tf.norm(tf.layers.flatten(grad))
        return grad_norm

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

class Solver(object):


    def custom_bin(self, y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)

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

        self.test_file = config.test_file
        self.test_labels = config.test_labels
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
        self.G = get_generator(self.g_conv_dim, self.n_labels, self.g_repeat_num, self.image_size)
        self.D = get_discriminator(self.d_conv_dim, self.n_labels, self.d_repeat_num, self.image_size)

        print(self.G.summary())

        self.d_optimizer = keras.optimizers.Adam(lr = self.d_lr, beta_1 = self.beta_1, beta_2 = self.beta_2)
        self.g_optimizer = keras.optimizers.Adam(lr = self.g_lr, beta_1 = self.beta_2, beta_2 = self.beta_2)

        self.D.trainable = False

        combined_real_img = Input(shape = (self.image_size, self.image_size, 3))
        input_orig_labels = Input(shape = (self.image_size, self.image_size, self.n_labels))
        input_target_labels = Input(shape = (self.image_size, self.image_size, self.n_labels))

        concatted_input = Concatenate(axis=3)([combined_real_img, input_target_labels])

        combined_fake_img = self.G(concatted_input)
        output_src, output_cls     = self.D(combined_fake_img)
        concatted_combined_fake_img = Concatenate(axis=3)([combined_fake_img,input_orig_labels])
        reconstr_img = self.G(concatted_combined_fake_img)


        self.combined = Model(inputs = [combined_real_img, input_orig_labels, input_target_labels], outputs = [reconstr_img, output_src, output_cls])

        self.combined.compile(loss = ["mae", neg_mean_loss, self.custom_bin], loss_weights = [self.lambda_rec, 1, self.lambda_cls], optimizer = self.g_optimizer)

        shape = (self.image_size,self.image_size,3)
        fake_input, real_input, interpolation = Input(shape), Input(shape), Input(shape)
        norm = GradNorm()([self.D(interpolation)[0], interpolation])
        fake_output_src, fake_output_cls = self.D(fake_input)
        real_output_src, real_output_cls = self.D(real_input)
        self.DIS = Model([real_input, fake_input, interpolation], [fake_output_src, real_output_src, real_output_cls, norm])
        # self.DIS = Model([gen_input], output_D)

        self.D.trainable = True

        self.DIS.compile(loss=[mean_loss, neg_mean_loss, self.custom_bin, 'mse'], loss_weights = [1, 1, self.lambda_cls, self.lambda_gp], optimizer= self.d_optimizer)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = self.batch_size
        out = np.zeros((batch_size, dim))
        out[np.arange(batch_size), labels.astype(np.int_)] = 1
        return out

    def create_labels(self, c_org, c_dim=5, selected_attrs=None):
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
        callback = keras.callbacks.TensorBoard(log_dir = self.log_dir, write_graph = False)
        callback.set_model(self.combined)
        dis_names = ['Discriminator Adversarial loss', 'Discriminator Classification loss', 'Gradient Penalty']
        gen_names = ['Cycle loss', 'Generator Adversarial loss', 'Generator Classification loss']

        data_iter = iter(self.data_loader)
        
        test_imgs, label_test = next(data_iter)
        test_imgs = np.tile(test_imgs, (5,1,1,1))
        c_fixed = np.asarray(self.create_labels(label_test, self.n_labels, self.selected_attrs))
        c_fixed = np.concatenate(c_fixed, axis = 0)
        labels_fixed = c_fixed.reshape((5 * self.batch_size, 1, 1, 5))
        label_test = np.tile(label_test, (5,1))
        test_imgs_concatted = np.concatenate((test_imgs, np.tile(labels_fixed, (1,self.image_size, self.image_size,1))), axis=3)


        batch_id = 0
        total_steps = self.num_iters//self.log_step//self.n_critic
        steps_before_decay = self.num_iters_decay // self.log_step//self.n_critic

        start = 0
        if self.restore_epoch > 0:
            start = self.restore_epoch * self.model_save_step//self.log_step
            batch_id = self.restore_epoch * self.model_save_step * self.n_critic
            
        for epoch in trange(start,total_steps+1):

            reduction = epoch - steps_before_decay
            if (reduction > 0 == 0):
                ratio = reduction / (total_steps - steps_before_decay)
                lr = (self.g_lr * (1.0-ratio)) 
                print(lr)
                K.set_value(self.combined.optimizer.lr, lr)
                lr = (self.d_lr * (1.0-ratio)) 
                K.set_value(self.DIS.optimizer.lr, lr)

            if (epoch % (self.sample_step / self.log_step) == 0):
                with keras.backend.get_session().as_default():
                    outcome = self.G.predict(test_imgs_concatted)
                    tmp = np.concatenate((outcome, np.tile(label_test.reshape((5*self.batch_size,1,1,5)),(1,self.image_size,self.image_size,1))),axis=3)
                    cycled = self.G.predict(tmp)
                    s = BytesIO()

                    left = self.denorm(test_imgs[epoch%(5*self.batch_size)].reshape((self.image_size,self.image_size,3)))
                    right = self.denorm(outcome[epoch%(5*self.batch_size)].reshape((self.image_size,self.image_size,3)))
                    total = np.concatenate((left,right),axis = 1)

                    right = self.denorm(cycled[epoch%(5*self.batch_size)].reshape((self.image_size,self.image_size,3)))
                    total = np.concatenate((total,right), axis = 1)

                    plt.imsave(s, total)
                    out = tf.Summary.Image(encoded_image_string = s.getvalue())
                    labels = np.concatenate((label_test[epoch%(5*self.batch_size)].reshape((1,self.n_labels)),labels_fixed[epoch%(5*self.batch_size)].reshape((1,self.n_labels))))
                    s = BytesIO()
                    plt.imsave(s, labels)
                    labels = tf.Summary.Image(encoded_image_string = s.getvalue())

                    summary = tf.Summary(value=[tf.Summary.Value(tag = "In->Out->Cycled", image = out),
                                                tf.Summary.Value(tag = "Labels", image = labels)])
                    callback.writer.add_summary(summary, epoch)
                    callback.writer.flush()


            d_loss_r = 0
            d_loss_f = 0
            d_logs = np.zeros(5)
            g_logs = np.zeros(4)
            for i in trange(0, self.log_step):
                for j in range(0,self.n_critic):
                    try:
                        x_real, label_org = next(data_iter)
                    except:
                        data_iter = iter(self.data_loader)
                        x_real, label_org = next(data_iter)

                    label_trg = np.flip(label_org, axis=0)

                    c_org = label_org.copy()
                    c_trg = label_trg.copy()

                    labels_trg = c_trg.reshape((self.batch_size,1,1,self.n_labels))
                    x_concatted = np.concatenate((x_real, np.tile(labels_trg, (1,self.image_size, self.image_size,1))), axis=3)


                    x_fake = self.G.predict(x_concatted)


                    fake = np.zeros(self.batch_size)
                    real = np.ones(self.batch_size)

                    epsilon = np.random.uniform(0, 1, size = (self.batch_size,1,1,1))
                    interpolation = epsilon * x_real + (1-epsilon) * x_fake
                    d_logs += self.DIS.train_on_batch([x_real, x_fake, interpolation], [np.tile(fake.reshape((self.batch_size,1)),(1,4)), np.tile(real.reshape((self.batch_size,1)),(1,4)), c_org, np.ones(self.batch_size)])    
                    batch_id += 1

                tiled_label_org = np.tile(label_org.reshape(self.batch_size,1,1,5),(1,self.image_size,self.image_size,1))
                tiled_label_trg = np.tile(label_trg.reshape(self.batch_size,1,1,5),(1,self.image_size,self.image_size,1))
                g_logs += self.combined.train_on_batch([x_real, tiled_label_org, tiled_label_trg], [x_real, np.tile(real.reshape((self.batch_size,1)),(1,4)), c_trg])

            d_logs = (d_logs / (self.log_step * self.n_critic)).tolist()
            g_logs = (g_logs / self.log_step).tolist()


            write_log(callback, gen_names, g_logs[1:4], batch_id)
            write_log(callback, dis_names, [d_logs[1]+d_logs[2]] +d_logs[3:5], batch_id)

            if (epoch > start and epoch % (self.model_save_step // self.log_step) == 0):
                self.restore_epoch += 1

                self.combined.save_weights(self.model_dir + "combined_weights" + str(self.restore_epoch) + ".h5")

                self.store_optimizer(self.combined, "combined")
                self.store_optimizer(self.DIS, "DIS")

    def test(self):
        if self.restore_epoch == 0:
            print("Choose non zero restore epoch for testing")
            return;

        if self.test_file is None or self.test_labels is None:
            print("--test_file and --test_labels have to be provided")
            return;

        self.test_labels = np.asarray(self.test_labels).astype(np.float32)

        img = Image.open(self.test_file)
        img = img.resize((self.image_size, self.image_size))
        img = np.asarray(img)
        img = img.astype(np.float32)
        img = img / 255

        #normalize:
        img = (img - 0.5) / 0.5 #normalize to [-1,1]


        input_target_labels = np.asarray(self.create_labels(np.asarray([self.test_labels]).reshape(1,self.n_labels), self.n_labels, self.selected_attrs))
        input_target_labels = input_target_labels.reshape((self.n_labels,1,1,self.n_labels))
        

        img_tiled = np.tile(img.reshape(1,self.image_size,self.image_size,3),(self.n_labels,1,1,1))
        concatted_imgs = np.concatenate((img_tiled, np.tile(input_target_labels, (1,self.image_size, self.image_size,1))), axis=3) 

        fake_images = self.G.predict(concatted_imgs)

        fake_images = self.denorm(fake_images)
        fake_images = fake_images * 255

        for i in range(self.n_labels):
            im = Image.fromarray(fake_images[i,:,:,:].astype('uint8'))
            im.save(self.result_dir + str(i)+".png")











            







            



