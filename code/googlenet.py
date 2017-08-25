from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

# gpu config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.6)
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
		
		
        # build googlenet network
		
        # inception_3a
        network = input_data(shape=[None, 256, 256, 3])
        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')
		
        # merge the inception_3a__
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

        # inception_3b
        inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

        # merge the inception_3b_*
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')

        # inception_4a
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

        # inception_4b
        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')
        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')

        # inception_4c
        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')
        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

        # inception_4d
        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

        # inception_4e
        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')

        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        # inception_5a
        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')

        # inception_5b
        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)
        loss = fully_connected(pool5_7_7, 3,activation='softmax')
        network = regression(loss, optimizer='momentum',
                  loss='categorical_crossentropy', learning_rate=0.001)
		
		
        # training
        if not os.path.isdir('checkpoints'):
            os.makedirs('checkpoints')
        #if not os.path.isdir('model'):
            #os.makedirs('model')
		
        model = tflearn.DNN(network, checkpoint_path='checkpoints/googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)
        model.fit(X, Y, n_epoch=200, validation_set=(testX, testY), shuffle=True,
                  show_metric=True, batch_size=64, snapshot_step=200,
                  snapshot_epoch=False, run_id='googlenet')
        #model.save('model/model_retrained_by_googlenet')