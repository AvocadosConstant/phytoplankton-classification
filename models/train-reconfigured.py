
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import dataset
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  #input_layer = tf.reshape(features, [-1, 256, 256, 3])
  input_layer = tf.placeholder(tf.float32, shape=[None, 256,256,3], name='x')
  input_layer = features

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 256, 256, 1x3]
  # Output Tensor Shape: [batch_size, 256, 256, 32x3]

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      trainable=True,
      name="conv1")

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 256, 256, 32x3]
  # Output Tensor Shape: [batch_size, 128, 128, 32x3]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 128, 32x3]
  # Output Tensor Shape: [batch_size, 128, 128, 64x3]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 128, 64x3]
  # Output Tensor Shape: [batch_size, 64, 64, 64x3]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 64, 64, 64x3]
  # Output Tensor Shape: [batch_size, 64 * 64 * 64x3]
  layer_s = pool2.get_shape()
  num_features = layer_s[1:4].num_elements()

  pool2_flat = tf.reshape(pool2, [-1, num_features])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=9)
  print(logits)
  print(labels)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  # for i in range(0,32):
  #     print(conv1[0,:28,:28,i])
  #print(tf.trainable_variables())
  with tf.variable_scope('conv1') as scope:
    tf.get_variable_scope().reuse_variables()
    weights = tf.get_variable('kernel')
    x_min = tf.reduce_min(weights)
    x_max = tf.reduce_max(weights)
    weights_0_to_1 = (weights - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)

  # to tf.image_summary format [batch_size, height, width, channels]
    weights_transposed = tf.transpose (weights_0_to_255_uint8, [3, 0, 1, 2])
    print(weights_transposed)
  # this will display random 3 filters from the 64 in conv1
    tf.summary.image('conv1/filters', weights_transposed,max_outputs=32)
    # print (weights)
  print("-----------------------\t Print Here \t----------------------------")
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  labels = tf.argmax(labels,axis=1)
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
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
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data=data.train.images
  #,train_labels = mnist.train.images  # Returns np.array
  train_labels = data.train.labels
  train_labels_cls = data.train.cls
  #np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels_correct = np.asarray(mnist.test.labels, dtype=np.int32)
  eval_data = data.valid.images
  eval_labels = data.valid.labels
  eval_label_cls = data.valid.cls

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="tmp/")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities":"softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=train_data,
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=2000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=eval_data,
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
