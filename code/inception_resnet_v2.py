from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import tflearn.activations as activations
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from tflearn.utils import repeat

# gpu config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
config = tf.ConfigProto(gpu_options=gpu_options)


def read_and_decode(data_path, batch_size): 
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
	
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=
              {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/class/label': tf.FixedLenFeature([], tf.int64)})

    # Convert the image data from string back to the numbers
    img = tf.decode_raw(features['image/encoded'], tf.uint8)

    # Reshape image data into the original shape
    image = tf.reshape(img, [256, 256, 3])
    #image = tf.cast(image, tf.uint8)
									   
    # Cast label data into int32
    label = tf.cast(features['image/class/label'], tf.int32)
	
    # Creates batches by randomly shuffling tensors
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                batch_size=batch_size, capacity=200, num_threads=32,
                min_after_dequeue=16)
	
    return image_batch, label_batch

def block35(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None,name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 32, 3, bias=False, activation=None,name='Conv2d_0b_3x3')))
    tower_conv2_0 = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 48,3, bias=False, activation=None, name='Conv2d_0b_3x3')))
    tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 64,3, bias=False, activation=None, name='Conv2d_0c_3x3')))
    tower_mixed = merge([tower_conv, tower_conv1_1, tower_conv2_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

def block17(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv_1_0 = relu(batch_normalization(conv_2d(net, 128, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv_1_1 = relu(batch_normalization(conv_2d(tower_conv_1_0, 160,[1,7], bias=False, activation=None,name='Conv2d_0b_1x7')))
    tower_conv_1_2 = relu(batch_normalization(conv_2d(tower_conv_1_1, 192, [7,1], bias=False, activation=None,name='Conv2d_0c_7x1')))
    tower_mixed = merge([tower_conv,tower_conv_1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net


def block8(net, scale=1.0, activation="relu"):
    tower_conv = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_1x1')))
    tower_conv1_0 = relu(batch_normalization(conv_2d(net, 192, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
    tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 224, [1,3], bias=False, activation=None, name='Conv2d_0b_1x3')))
    tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 256, [3,1], bias=False, name='Conv2d_0c_3x1')))
    tower_mixed = merge([tower_conv,tower_conv1_2], mode='concat', axis=3)
    tower_out = relu(batch_normalization(conv_2d(tower_mixed, net.get_shape()[3], 1, bias=False, activation=None, name='Conv2d_1x1')))
    net += scale * tower_out
    if activation:
        if isinstance(activation, str):
            net = activations.get(activation)(net)
        elif hasattr(activation, '__call__'):
            net = activation(net)
        else:
            raise ValueError("Invalid Activation.")
    return net

if __name__ == '__main__':
	
    # load data
    data_train = '../dataset/output/train.tfrecords'
    data_test = '../dataset/output/validation.tfrecords'
    batch_train = 2000
    batch_test = 400
    imgs_train, labels_train = read_and_decode(data_train, batch_train)
    imgs_test, labels_test = read_and_decode(data_test, batch_test)
	
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
	    # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
									   		
        images_train, lbls_train = sess.run([imgs_train, labels_train])
        #images = images.astype(np.uint8)
        images_ = tf.reshape(images_train, [batch_train,256,256,3])
        print(images_train.shape)
		
        images_test, lbls_test = sess.run([imgs_test, labels_test])
        #images = images.astype(np.uint8)
        images_ = tf.reshape(images_test, [batch_test,256,256,3])
        print(images_test.shape)
            
        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
        sess.close()
		
        # ont hot
        X = images_train
        testX = images_test
        Y = tflearn.data_utils.to_categorical(lbls_train, 3)
        testY = tflearn.data_utils.to_categorical(lbls_test, 3)
        print(X.shape, testX.shape, Y.shape, testY.shape)
		
		
        # build alexnet network		
        num_classes = 3
        dropout_keep_prob = 0.8

        network = input_data(shape=[None, 256, 256, 3])
        conv1a_3_3 = relu(batch_normalization(conv_2d(network, 32, 3, strides=2, bias=False, padding='VALID',activation=None,name='Conv2d_1a_3x3')))
        conv2a_3_3 = relu(batch_normalization(conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID',activation=None, name='Conv2d_2a_3x3')))
        conv2b_3_3 = relu(batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3')))
        maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
        conv3b_1_1 = relu(batch_normalization(conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID',activation=None, name='Conv2d_3b_1x1')))
        conv4a_3_3 = relu(batch_normalization(conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID',activation=None, name='Conv2d_4a_3x3')))
        maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

        tower_conv = relu(batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1')))

        tower_conv1_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5')))

        tower_conv2_0 = relu(batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1')))
        tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3')))
        tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None,name='Conv2d_5b_b2_0c_3x3')))

        tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
        tower_conv3_1 = relu(batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None,name='Conv2d_5b_b3_0b_1x1')))

        tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

        net = repeat(tower_5b_out, 10, block35, scale=0.17)

        tower_conv = relu(batch_normalization(conv_2d(net, 32, 1, bias=False, strides=2,activation=None, padding='VALID', name='Conv2d_6a_b0_0a_3x3')))
        tower_conv1_0 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_6a_b1_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1_0, 256, 3, bias=False, activation=None, name='Conv2d_6a_b1_0b_3x3')))
        tower_conv1_2 = relu(batch_normalization(conv_2d(tower_conv1_1, 32, 1, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_6a_b1_0c_3x3')))
        tower_pool = max_pool_2d(net, 1, strides=2, padding='VALID',name='MaxPool_1a_3x3')
        net = merge([tower_conv, tower_conv1_2, tower_pool], mode='concat', axis=3)
        net = repeat(net, 20, block17, scale=0.1)

        tower_conv = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, activation=None, name='Conv2d_0a_1x1')))
        tower_conv0_1 = relu(batch_normalization(conv_2d(tower_conv, 384, 1, bias=False, strides=2, padding='VALID', activation=None,name='Conv2d_0a_1x1')))

        tower_conv1 = relu(batch_normalization(conv_2d(net, 256, 1, bias=False, padding='VALID', activation=None,name='Conv2d_0a_1x1')))
        tower_conv1_1 = relu(batch_normalization(conv_2d(tower_conv1,288,1, bias=False, strides=2, padding='VALID',activation=None, name='COnv2d_1a_3x3')))

        tower_conv2 = relu(batch_normalization(conv_2d(net, 256,1, bias=False, activation=None,name='Conv2d_0a_1x1')))
        tower_conv2_1 = relu(batch_normalization(conv_2d(tower_conv2, 288, 1, bias=False, name='Conv2d_0b_3x3',activation=None)))
        tower_conv2_2 = relu(batch_normalization(conv_2d(tower_conv2_1, 320, 1, bias=False, strides=2, padding='VALID',activation=None, name='Conv2d_1a_3x3')))

        tower_pool = max_pool_2d(net, 1, strides=2, padding='VALID', name='MaxPool_1a_3x3')
        net = merge([tower_conv0_1, tower_conv1_1,tower_conv2_2, tower_pool], mode='concat', axis=3)

        net = repeat(net, 9, block8, scale=0.2)
        net = block8(net, activation=None)

        net = relu(batch_normalization(conv_2d(net, 1536, 1, bias=False, activation=None, name='Conv2d_7b_1x1')))
        net = avg_pool_2d(net, net.get_shape().as_list()[1:3],strides=2, padding='VALID', name='AvgPool_1a_8x8')
        net = flatten(net)
        net = dropout(net, dropout_keep_prob)
        loss = fully_connected(net, num_classes,activation='softmax')

        network = tflearn.regression(loss, optimizer='RMSprop',
                  loss='categorical_crossentropy',
                  learning_rate=0.0001)
		
        # training
        if not os.path.isdir('checkpoints'):
            os.makedirs('checkpoints')
        #if not os.path.isdir('model'):
            #os.makedirs('model')

        model = tflearn.DNN(network, checkpoint_path='checkpoints/inception_resnet_v2',
                    max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="tflearn_logs/")
        model.fit(X, Y, n_epoch=200, validation_set=(testX, testY), shuffle=True,
                  show_metric=True, batch_size=64, snapshot_step=200,
                  snapshot_epoch=False, run_id='inception_resnet_v2')		  
        #model.save('model/model_retrained_by_inception_resnet_v2')