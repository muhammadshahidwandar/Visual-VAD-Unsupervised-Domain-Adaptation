"""
With this script you can test Resnet50 as provided in the Resnet.py
on any given dynamic image.
Specify the configuration settings at the beginning according to your
problem.
This script was written for TensorFlow 1.12 

Author: Muhammad Shahid
contact: shhaid.eyecom(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from datageneratorSequenceBatch import ImageDataGeneratorSeq
import cv2

# Path to take the trained model
checkpoint_path = "./model/Test1Ep10/"
Gpu_id = "0"

# Print a path error if it doesn't exist
if not os.path.isdir(checkpoint_path): print('model weights path does not exist',checkpoint_path)

saver = tf.train.import_meta_graph(checkpoint_path+'model_epoch10.ckpt.meta')
img = cv2.imread('./images/DynmcImg_1_2559.jpg')
# rescale image
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32)
[l,m,n]= img.shape
img_batch = np.ndarray(shape=(1,l,m,n), dtype=float)
img_batch[0,:,:,:] = img
# Restore the tensor Graph
graph = tf.get_default_graph()
##### Extract the variable from tensor graph 
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y:0")
is_training = graph.get_tensor_by_name("training:0")
#######################
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

os.environ["CUDA_VISIBLE_DEVICES"] = Gpu_id
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# Start Tensorflow session
with tf.Session() as sess:
    # Restore Weights
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    fnlOut = np.array([0, 0])
    feedDict = {x: img_batch, is_training: False}
    probs = sess.run(y_pred, feed_dict=feedDict)
    lablP = np.argmax(probs,axis=1)
    if(lablP>0):
        print('Speaker Detected with confidence',probs[0][1])
    else:
        print('not Speaker Detected with confidence',probs[0][0])