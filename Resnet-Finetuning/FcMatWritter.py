"""
With this script you can  evaluate the trained Resnet50 model
as provided in the Resnet.py on given dynamic images of any dataset.
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
from datageneratorSequenceBatch import ImageDataGenerator2


"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
val_file = './TrainTestColmb/testColumb5.txt'

# parameters
batch_size = 128
GPU_id = '0'

# Path to store model checkpoints
checkpoint_path = "./PavisTraindModls/PavisTestComplet3/resnet_20190620_131408/checkpoint/"

# print error if check point path doesn't exist
if not os.path.isdir(checkpoint_path): print('model weights path does not exist',checkpoint_path)

saver = tf.train.import_meta_graph(checkpoint_path+'model_epoch10.ckpt.meta')
val_generator = ImageDataGenerator2(val_file, shuffle = False,basePath = './PersonDynamic10/')#PersonDynamic10/')#./PersonOptical/')

# Get the number of training/validation steps per epoch
val_batches_per_epoch = np.ceil(val_generator.data_size / batch_size).astype(np.int16)

# Restore Graph
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y:0")
is_training = graph.get_tensor_by_name("training:0")
#######################
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
# Start Tensorflow session
temp2 = np.ones(shape=[1,2048],dtype= float)
with tf.Session() as sess:
    # Restore Weights
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
       
    
    for epoch in range(1):
        step = 1
       
        test_acc = 0.
        test_count = 0
        temp=np.empty([0, 2048])
        label=np.empty([0,1])
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            fc_val = sess.run(fc1, feed_dict={x: batch_tx,is_training: False})
            for it in range(len(batch_ty)):
                diff = fc_val[it]-temp2
                temp = np.vstack([temp,diff])
                label = np.vstack([label, np.argmax(batch_ty,axis=1)[it]])
           
        data = {}
        #data['FilePath'] = paths
        data['Featurs'] = temp
        data['label'] = label
        savemat('./Test1_Mat/Speakr6ResnetFc.mat', data)
        print('MatSave')
