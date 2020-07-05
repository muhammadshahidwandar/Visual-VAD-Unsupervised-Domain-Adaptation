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
from datagenerator import ImageDataGenerator2


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
with tf.Session() as sess:
    # Restore Weights
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    # Validate the model on the entire validation set
    print("{} Start validation".format(datetime.now()))
    test_acc = 0.
    test_count = 0
    labl = 0;
    label_org = np.array([])
    label_pred = np.array([])
    fnlOut = np.array([0, 0])
    ##################################
    print('started')
    val_generator.reset_pointer()
    TP=0;TN=0;FP=0;FN = 0;TPR =0;FPR=0;Auc = 0
    endl = batch_size#128
    for tp in range(val_batches_per_epoch):
        if(tp==val_batches_per_epoch-1):
            endl= val_generator.data_size%128#val_batches_per_epoch*128-val_generator.data_size
        batch_tx, batch_ty = val_generator.next_batch(batch_size)
        feedDict = {x: batch_tx, is_training: False}
        probs = sess.run(y_pred, feed_dict=feedDict)
        lablP = np.argmax(probs,axis=1)
        true_labels= np.argmax(batch_ty,axis=1);
        results = probs[0:endl,:]
        lablP = np.argmax(results,axis=1)
        true_labels= np.argmax(batch_ty[0:endl,:],axis=1)
        TP = TP+np.sum(np.logical_and(lablP == 1, true_labels == 1))
        TN = TN+np.sum(np.logical_and(lablP == 0, true_labels == 0))
        FP = FP+np.sum(np.logical_and(lablP == 1, true_labels == 0))
        FN = FN+np.sum(np.logical_and(lablP == 0, true_labels == 1))
    #Percision recall and F1 score
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    Auc = (TPR-FPR+1)/2
    print( 'TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
    PRCN = np.float(TP) / np.float(TP + FP+0.00001)
    RCAL = np.float(TP) / np.float(TP + FN+0.00001)
    F1 = 2 * PRCN * RCAL / (PRCN + RCAL+0.00001)
    print( 'RECALL: %f, Percision: %f, F1 Score: %f, Auc: %f' % (RCAL, PRCN, F1, Auc))