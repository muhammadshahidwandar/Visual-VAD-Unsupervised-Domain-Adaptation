import numpy as np
import cv2
import os
import re
from natsort import natsorted
import random


width_hlf = 64  # from image center to edge length (=width/2)
"""
This code is initialized by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
and a lot of extra functionalities like (class balanced batch generation) and 
data augumentation(taking fixed size chunk from randomly scaled image) are added 

"""

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=True,Path = './DfaulPath/',
                 mean = np.array([128., 128., 128.],float), scale_size=(156, 156),
                 nb_classes = 2):
        
                
        # initialize the parameters
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.basePath =Path
        self.scale_size = scale_size
        self.pointer = 0
        self.read_class_list(class_list)

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.imagesClass1 = []
            self.imagesClass2 = []
            self.labelsClass1 = []
            self.labelsClass2 = []
            for l in lines:
                items = l.split()
                fullPath = self.basePath + items[0] + '/'
                filelist = [file for file in os.listdir(fullPath) if file.endswith('.jpg')]
                filelist = natsorted(filelist)
                for file in filelist:
                     label = int(file.split('_')[1])
                     if (label == 0):
                          self.imagesClass1.append(fullPath + file)
                          self.labelsClass1.append(label)
                     else:
                          self.imagesClass2.append(fullPath + file)
                          self.labelsClass2.append(label)

            
            #Total number of images
            self.LenClass1 = len(self.imagesClass1)
            self.LenClass2 = len(self.imagesClass2)
            self.data_size = self.LenClass1+self.LenClass2
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images[:]
        labels = self.labels[:]
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # np.random.random_integers()
        Class1Idx = np.random.randint(0, self.LenClass1, int((batch_size) / 2), dtype='int')
        Class2Idx = np.random.randint(0, self.LenClass2, int((batch_size) / 2), dtype='int')  #
        # Get next batch of image (path) and labels
        paths = []
        labels = []
        for i in range(int(batch_size / 2)):
            paths.append(self.imagesClass1[Class1Idx[i]])
            paths.append(self.imagesClass2[Class2Idx[i]])
            labels.append(0)
            labels.append(1)

        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        Num = np.random.randint(300, size=(2, 128))

        for i in range(len(paths)):
            # img = data_aug(paths[i])
            img = cv2.imread(paths[i])

            # flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
            img[Num[0, i]:Num[0, i] + width_hlf, Num[1, i]:Num[1, i] + width_hlf, :] = self.mean
            img = img.astype(np.float32)

            # subtract the mean
            img -= self.mean

            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            # print('length of label',len(labels))
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels