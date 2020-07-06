"""
With this script you can Train Resnet50 CNN as provided in the Resnet.py
on any given dataset.
Specify the configuration settings at the beginning according to your
problem.
This script was written for TensorFlow 1.12

Author: Muhammad Shahid
contact: shhaid.eyecom(at)gmail.com
"""
import os, sys
import numpy as np
import tensorflow as tf
import datetime
from Resnet import ResNetModel
from datageneratorBalancedBatch import ImageDataGenerator
from datageneratorSequenceBatch import ImageDataGeneratorSeq

tf.app.flags.DEFINE_float('learning_rate',1e-6, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs',20 , 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc,scale5', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('training_file', '../RealVAD/trainRealVAD1.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file','../RealVAD/testRealVAD1.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', './TrainedModel/', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 5, 'Logging period in terms of iteration')


FLAGS = tf.app.flags.FLAGS


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()
    GPU_id = "0"  #"1"
   # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, None, None, 3],name='x')
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes],name='y')
    is_training = tf.placeholder('bool', [],name='training')
    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(x, is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)
    loss = model.loss(x, y)
    train_op = model.optimize(FLAGS.learning_rate, train_layers)
    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    y_pred = tf.nn.softmax(model.prob,name="y_pred")
    ###################### To access intermediat weights or convolution layer from CNN model
    b = tf.constant(1.00,dtype=tf.float32)#,shape=[128,7,7,2048])
    Conv_Last = tf.multiply(b,model.s5,name = "convLast") # to force the graph to return this value
    c = tf.constant(1.00,dtype=tf.float32,shape=[FLAGS.batch_size,2048])
    fc1_lyr = tf.add(model.avg_pool,c,name = 'Fc_Layer')
    a = tf.constant(1.00,dtype=tf.float32,shape=[2048,2])
    Weight = tf.multiply(a,model.weights,name = "weight")
    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Initalize the data generator seperately for the training and validation set %'./PersonDynamic10/'
    train_generator = ImageDataGenerator(FLAGS.training_file, 
                                     horizontal_flip = True, shuffle = False,Path = '../Columb_Dynamic/')#PersonOptical2/')#PersonOPFDynamic12/')#./PersonOptical/')
    val_generator = ImageDataGeneratorSeq(FLAGS.val_file, shuffle = False,basePath = '../Columb_Dynamic/')#PersonOptical2/')#PersonOPFDynamic12/')#./PersonOptical/')
    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.ceil(val_generator.data_size / FLAGS.batch_size).astype(np.int16)
    print('Total Number of training Batches',train_batches_per_epoch)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers,weight_Path='ResNet-L50.npy')

        # If you want to restore from intermediated trained step, uncoment this following line
        # saver.restore(sess, "trainedModel/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                # Get a batch of images and labels
                batch_xs, batch_ys = train_generator.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, is_training: True})

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1
                remiand = epoch%1 # this will be used for full test set evaluation
            if(epoch>=FLAGS.num_epochs-1):
                # Epoch completed, start validation
                print("{} Start validation".format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_generator.next_batch(FLAGS.batch_size)
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                    test_acc += acc
                    test_count += 1

                test_acc /= test_count
                s = tf.Summary(value=[
                    tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
                ])
                val_writer.add_summary(s, epoch+1)
                fc1_val = sess.run(Weight, feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                print(fc1_val)
                print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
                ###########################F1 Score calculation on full test set#################
            if(remiand==0):
                print('started')
                val_generator.reset_pointer()
                TP=0;TN=0;FP=0;FN = 0;TPR =0;FPR=0;Auc = 0
                endl =FLAGS.batch_size 
                for tp in range(val_batches_per_epoch):
                    if(tp==val_batches_per_epoch-1):
                        endl= val_generator.data_size%128
                    batch_tx, batch_ty = val_generator.next_batch(FLAGS.batch_size)
                    feedDict = {x: batch_tx, is_training: False}
                    weighted_probs = sess.run(y_pred, feed_dict=feedDict)
                    results = weighted_probs[0:endl,:]

                    lablP = np.argmax(results,axis=1)
                    true_labels= np.argmax(batch_ty[0:endl,:],axis=1)
                    TP = TP+np.sum(np.logical_and(lablP == 1, true_labels == 1))
                    TN = TN+np.sum(np.logical_and(lablP == 0, true_labels == 0))
                    FP = FP+np.sum(np.logical_and(lablP == 1, true_labels == 0))
                    FN = FN+np.sum(np.logical_and(lablP == 0, true_labels == 1))
                
                TPR = TP/(TP+FN)
                FPR = FP/(FP+TN)
                Auc = (TPR-FPR+1)/2
            # #Percision recall and F1 score
                print( 'TP: %i, FP: %i, TN: %i, FN: %i' % (TP, FP, TN, FN))
                PRCN = np.float(TP) / np.float(TP + FP+0.00001)
                RCAL = np.float(TP) / np.float(TP + FN+0.00001)
                F1 = 2 * PRCN * RCAL / (PRCN + RCAL+0.00001)
                print( 'RECALL: %f, Percision: %f, F1 Score: %f, Auc: %f' % (RCAL, PRCN, F1, Auc))
            ###############################################################
            # Reset the file pointer of the image data generator
            val_generator.reset_pointer()
            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

if __name__ == '__main__':
    tf.app.run()