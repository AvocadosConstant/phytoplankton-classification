import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import os
import numpy as np

"""
    Code originally lifted from
    http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
"""

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(3)

trial_name = "pc_no_unidentified_rs_3.txt"
batch_size = 32

#Prepare input data
classes = ['Asterionella','Aulocoseira','Colonial Cyanobacteria','Cryptomonas','Detritus','Dolichospermum','Filamentous cyanobacteria','Romeria','Staurastrum']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.20
img_size = 256
num_channels = 3
os.chdir('..')
train_path=os.getcwd()
train_path += '/data/416_Station40_09012015_10x/extracted_images/'
# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

keep_prob = tf.placeholder(tf.float32,name='keep_prob')

##Network graph params
filter_size_conv1 = 7
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 2048

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters, name):
    shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters]
    ## We shall define the weights that will be trained using create_weights function.
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05,name="_kernel"))
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME',name = name)
    print(layer)
    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer



def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
             num_inputs,
             num_outputs,
             use_relu=True):

    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1, name = "conv1")
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2, name="conv2")

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3, name = "conv3")

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)
# session.run(tf.global_variables_initializer())
print(layer_fc1)

print(keep_prob)
dropout = tf.nn.dropout(x=layer_fc1,
    keep_prob=keep_prob,
    noise_shape=None,
    seed=3,
    name='dropout')
layer_fc2 = create_fc_layer(input=dropout,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv1')

tf.get_variable_scope().reuse_variables()
    # print(tf.trainable_variables())
weights = vars[0]

#TO-DO: Normalize by channel and image
#x_min = tf.reduce_min(weights)
#x_max = tf.reduce_max(weights)
#weights_0_to_1 = (weights - x_min) / (x_max - x_min)
#weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
weights_transposed = tf.transpose (weights, [3, 0, 1, 2])
#print(weights)
#?summary_writer = tf.summary.FileWriter('summary_dir');
#summary_writer.add_graph(graph=tf.get_default_graph())
session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    log = open(trial_name,"a")
    log.write(msg.format(epoch + 1, acc, val_acc, val_loss))
    log.write("\n")
    log.close()
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob:0.5}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch,
                              keep_prob:1}

        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss= session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))
            #print (vars[0])
            # summary = tf.summary.tensor_summary(name="conv1/filters", tensor = tens_summary, summary_description = "filter images")
            # print(summary)
            #summary_writer.add_summary(img, global_step=i)
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, '/data/szaman5/cnn-save/phytoplankton-model')
    total_iterations += num_iteration
train(num_iteration=1960)
