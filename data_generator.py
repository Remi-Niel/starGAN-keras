import numpy as np
import keras
import random
from PIL import Image
import os


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_dir, attr_path, selected_attrs,  mode, crop_size, image_size, batch_size):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.crop_size = crop_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
        self.on_epoch_end()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]


        self.list_IDs = 0
        if self.mode != 'train':
            self.list_IDs = 2000


        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            #first 2000 files in test_set rest in train_set
            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
                if self.mode == 'train':
                    self.list_IDs += 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.list_IDs / self.batch_size))

    def preprocess_im(self, image):
        width, height = image.size
        if random.randint(0,2):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = image.crop((int(width/2.0 - self.crop_size/2.0),int(height/2.0 - self.crop_size/2.0),int(width/2.0 + self.crop_size/2.0),int(height/2.0 + self.crop_size/2.0)))
        image = image.resize((self.image_size, self.image_size))

        img = np.asarray(image)
        img = img.astype(np.float32)
        img = img / 255

        #normalize:
        img = (img - 0.5) / 0.5 #normalize to [-1,1]

        return img

    def __getsingleitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.preprocess_im(image), np.asarray(label)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        self.on_epoch_end()

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs-1)
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        y = np.empty((self.batch_size, len(self.selected_attrs)), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample and class
            X[i,], y[i,] = self.__getsingleitem__(ID)

        return X, y

def get_generator(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, mode='train'):

    generator = DataGenerator(image_dir, attr_path, selected_attrs,  mode, crop_size, image_size, batch_size)

    return generator