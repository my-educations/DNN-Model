from __future__ import division, print_function, absolute_import
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn

# gpu config
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
config = tf.ConfigProto(gpu_options=gpu_options)

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

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
		
        # Real-time data preprocessing
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True)

        # Real-time data augmentation
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_crop([32, 32], padding=4)

        # Building Residual Network
        net = tflearn.input_data(shape=[None, 256, 256, 3],
                         #data_preprocessing=img_prep,
                         #data_augmentation=img_aug)
								)
        net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
        net = tflearn.residual_block(net, n, 16)
        net = tflearn.residual_block(net, 1, 32, downsample=True)
        net = tflearn.residual_block(net, n-1, 32)
        net = tflearn.residual_block(net, 1, 64, downsample=True)
        net = tflearn.residual_block(net, n-1, 64)
        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)
        # Regression
        net = tflearn.fully_connected(net, 3, activation='softmax')
        mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
		
        # training
        if not os.path.isdir('checkpoints'):
            os.makedirs('checkpoints')
        #if not os.path.isdir('model'):
            #os.makedirs('model')
		
        model = tflearn.DNN(net, checkpoint_path='checkpoints/resnet',
                    max_checkpoints=1, tensorboard_verbose=0, clip_gradients=0.)

        model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=200, show_metric=True, 
				  batch_size=64, shuffle=True, run_id='resnet')
        #model.save('model/model_retrained_by_resnet') 