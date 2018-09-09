from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import math
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import scipy.io as sio
import random

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 500
NUM_EPOCHS = 100
EVAL_BATCH_SIZE = 500
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
FINE_TURE = False
LAMBDA = 10
FLAGS = None
REMAIN_PERCENTAGE = 0.9
# LAMBDA_POOL = 10.0*np.arange(1.0, 15.0, 1.0, dtype=np.float32)
LAMBDA_POOL = np.arange(10.0, 500.0, 1.0, dtype=np.float32)
EPOCH_STAGE1 = 20


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = np.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=np.float32)
    labels = np.zeros(shape=(num_images,), dtype=np.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def realistic_error_rate(predictions, labels, predicted_hardness):
    """Return the error rate based on dense predictions and sparse labels."""
    # # print (predicted_hardness)
    # predicted_hardness = predicted_hardness / np.sum(predicted_hardness)
    # # print (np.argmax(predictions, 1) == labels)
    # # print (np.multiply(np.argmax(predictions, 1) == labels, np.squeeze(predicted_hardness)))
    # return 100.0 - 100 * np.sum(np.multiply(np.argmax(predictions, 1) == labels, np.squeeze(predicted_hardness)))
    # # return 100.0 - (
    # #     100.0 *
    # #     np.sum(np.argmax(predictions, 1) == labels) /
    # #     predictions.shape[0])
    print (np.sum(predicted_hardness))
    return 100.0 - 100 * (np.sum(np.multiply(np.argmax(predictions, 1) == labels, np.squeeze(predicted_hardness))) / np.sum(predicted_hardness))



def get_train_info(data_dir):
    hardness_scores = []
    with open(data_dir, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            info = line.split()
            hardness_scores.append(float(info[0]))
    return hardness_scores


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def loss_main_network(predicted_labels, predicted_hardness, train_labels_node):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    weighted_loss = tf.multiply(predicted_hardness, loss)
    return tf.reduce_mean(weighted_loss)


    #
    # ones = tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1))
    # ones_subs_probs = tf.subtract(ones, predicted_hardness)
    # focal_factor = tf.square(ones_subs_probs)
    #
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    # loss = tf.reshape(loss, [BATCH_SIZE, 1])
    # term1 = tf.multiply(focal_factor, loss)
    # term2 = tf.multiply(tf.square(predicted_hardness), tf.nn.relu(tf.subtract(tf.constant(M_parameter, dtype=tf.float32, shape=(BATCH_SIZE, 1)), loss)))
    # weighted_loss = tf.reduce_mean(tf.add(term1, term2))
    # return weighted_loss

def loss_main_network2(predicted_labels, predicted_hardness, train_labels_node, predicted_error):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    weighted_loss = tf.multiply(predicted_hardness, loss)
    return tf.reduce_mean(tf.multiply(weighted_loss, tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), predicted_error)))


def loss_adaptive_network(predicted_labels, predicted_hardness, train_labels_node):
    # hard sample with large weight considering classification error
    INCLINATION = 1.0
    PENALTY_ERR = 0.5
    slim_para = 100
    PENALTY_COE = 0.5
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32)))

    # compute 1 - e_i
    max_probs_duplicate = tf.tile(max_probs, multiples=[1, NUM_LABELS])
    prob_diff = tf.subtract(max_probs_duplicate, tf.nn.softmax(predicted_labels, dim=1))
    prob_diff_array = tf.nn.sigmoid(tf.multiply(prob_diff, tf.constant(slim_para, dtype=tf.float32)))
    reduced_prob_diff = tf.reduce_prod(prob_diff_array, axis=1, keep_dims=True)
    reduced_prob_diff = tf.nn.relu(reduced_prob_diff)
    error_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                             tf.multiply(tf.constant(2, dtype=tf.float32), reduced_prob_diff))
    error_loss = tf.nn.relu(error_loss)
    # modulate p_i
    # method 1
    #    max_probs = tf.nn.relu(tf.subtract(max_probs, tf.multiply(tf.constant(PENALTY_ERR, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss)))
    # max_probs = tf.nn.relu(tf.subtract(max_probs,
    #                                    tf.multiply(tf.constant(PENALTY_ERR, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
    #                                                tf.multiply(error_loss, max_probs))))


    # # method 2
    # max_probs_new = tf.add(max_probs, tf.pow(tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
    #                                                      tf.multiply(
    #                                                          tf.constant(2.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
    #                                                          max_probs)), error_loss))
    #
    # max_probs_new = tf.subtract(max_probs_new,
    #                             tf.pow(
    #                                 tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss),
    #                                 error_loss))
    # max_probs_new = tf.minimum(max_probs_new, max_probs)
    # max_probs = max_probs_new

    # method 3
    max_probs_new = tf.add(max_probs, tf.pow(tf.subtract(tf.constant(PENALTY_COE, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                                                         tf.multiply(
                                                             tf.constant(1.0 + PENALTY_COE, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                                                             max_probs)), error_loss))
    max_probs_new = tf.subtract(max_probs_new,
                                tf.pow(
                                    tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss),
                                    error_loss))
    max_probs_new = tf.minimum(max_probs_new, max_probs)
    max_probs = max_probs_new



    term1 = tf.multiply(tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), max_probs),
                        predicted_hardness)
    term2 = tf.multiply(tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), predicted_hardness),
                        max_probs)
    term1 = tf.multiply(tf.constant(INCLINATION, dtype=tf.float32, shape=(BATCH_SIZE, 1)), term1)
    final_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), tf.add(term1, term2))
    return tf.reduce_mean(final_loss)


    # INCLINATION = 1
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    # loss = tf.reshape(loss, [BATCH_SIZE, 1])
    # max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32)))
    # term1 = tf.multiply(tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), max_probs), predicted_hardness)
    # term2 = tf.multiply(tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), predicted_hardness), max_probs)
    # term1 = tf.multiply(tf.constant(INCLINATION, dtype=tf.float32, shape=(BATCH_SIZE, 1)), term1)
    # final_loss = tf.subtract(tf.constant(INCLINATION, dtype=tf.float32, shape=(BATCH_SIZE, 1)), tf.add(term1, term2))
    # return tf.reduce_mean(final_loss)


def compute_p_i(predicted_labels, predicted_hardness, train_labels_node):
    # easy sample with large weight considering classification error
    INCLINATION = 1
    PENALTY_ERR = 0.5
    slim_para = 100
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32)))

    # compute 1 - e_i
    max_probs_duplicate = tf.tile(max_probs, multiples=[1, NUM_LABELS])
    prob_diff = tf.subtract(max_probs_duplicate, tf.nn.softmax(predicted_labels, dim=1))
    prob_diff_array = tf.nn.sigmoid(tf.multiply(prob_diff, tf.constant(slim_para, dtype=tf.float32)))
    reduced_prob_diff = tf.reduce_prod(prob_diff_array, axis=1, keep_dims=True)
    reduced_prob_diff = tf.nn.relu(reduced_prob_diff)
    error_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                             tf.multiply(tf.constant(2, dtype=tf.float32), reduced_prob_diff))
    error_loss = tf.nn.relu(error_loss)
    # # modulate p_i
    #    max_probs = tf.nn.relu(tf.subtract(max_probs, tf.multiply(tf.constant(PENALTY_ERR, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss)))
    # max_probs = tf.nn.relu(tf.subtract(max_probs,
    #                                    tf.multiply(tf.constant(PENALTY_ERR, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
    #                                                tf.multiply(error_loss, max_probs))))

    max_probs_new = tf.add(max_probs, tf.pow(tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                                                         tf.multiply(
                                                             tf.constant(2.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                                                             max_probs)), error_loss))

    max_probs_new = tf.subtract(max_probs_new,
                                tf.pow(
                                    tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss),
                                    error_loss))
    max_probs_new = tf.minimum(max_probs_new, max_probs)

    return max_probs_new


def loss_error(predicted_labels, train_labels_node):
    slim_para = 1000.0
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32, shape=(BATCH_SIZE, 1))))
    max_probs_duplicate = tf.tile(max_probs, multiples=[1, NUM_LABELS])
    prob_diff = tf.subtract(max_probs_duplicate, tf.nn.softmax(predicted_labels, dim=1))
    prob_diff_array = tf.nn.sigmoid(tf.multiply(prob_diff, tf.constant(slim_para, dtype=tf.float32)))
    reduced_prob_diff = tf.reduce_prod(prob_diff_array, axis=1, keep_dims=True)
    error_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                             tf.multiply(tf.constant(2, dtype=tf.float32, shape=(BATCH_SIZE, 1)), reduced_prob_diff))
    return error_loss



def error_weighted_loss(predicted_labels, train_labels_node):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=predicted_labels)

    slim_para = 100
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
    loss = tf.reshape(loss, [BATCH_SIZE, 1])
    max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32, shape=(BATCH_SIZE, 1))))
    max_probs_duplicate = tf.tile(max_probs, multiples=[1, NUM_LABELS])
    prob_diff = tf.subtract(max_probs_duplicate, tf.nn.softmax(predicted_labels, dim=1))
    prob_diff_array = tf.nn.sigmoid(tf.multiply(prob_diff, tf.constant(slim_para, dtype=tf.float32)))
    reduced_prob_diff = tf.reduce_prod(prob_diff_array, axis=1, keep_dims=True)
    error_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                             tf.multiply(tf.constant(2, dtype=tf.float32, shape=(BATCH_SIZE, 1)), reduced_prob_diff))
    one_sub_error_loss = tf.subtract(tf.constant(1, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error_loss)
    return tf.reduce_mean(tf.multiply(cross_entropy, one_sub_error_loss))


def binary_cross_entropy(error, predicted_error):

    term1 = tf.multiply(error, tf.log(predicted_error))
    term2 = tf.multiply(tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error), tf.log(tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), predicted_error)))
    weights_for_balance = tf.multiply(tf.pow(tf.div(tf.reduce_sum(error), BATCH_SIZE), tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error)), tf.pow(1.0 - tf.div(tf.reduce_sum(error), BATCH_SIZE), error))



    return tf.reduce_mean(tf.multiply(weights_for_balance,  tf.multiply(tf.constant(-1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), tf.add(term1, term2))))

# def loss_error(predicted_labels, train_labels_node):
#     slim_para = 1
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node, logits=predicted_labels)
#     loss = tf.reshape(loss, [BATCH_SIZE, 1])
#     max_probs = tf.exp(tf.multiply(loss, tf.constant(-1, dtype=tf.float32)))
#     max_probs_duplicate = tf.tile(max_probs, multiples=[1, NUM_LABELS])
#     prob_diff = tf.subtract(max_probs_duplicate, tf.nn.softmax(predicted_labels, dim=1))
#     prob_diff_array = tf.nn.sigmoid(tf.multiply(prob_diff, tf.constant(slim_para, dtype=tf.float32)))
#     reduced_prob_diff = tf.reduce_prod(prob_diff_array, axis=1, keep_dims=True)
#     error_loss = tf.subtract(tf.constant(1, dtype=tf.float32),
#                              tf.multiply(tf.constant(2, dtype=tf.float32), reduced_prob_diff))
#     return tf.reduce_mean(error_loss)


def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def main(_):
    def train(train_data, train_labels, validation_data, validation_labels, test_data, test_labels, CUR_LAMBDA):
        accuracy_temp = 100
        all_easiness_te_save = np.zeros([10000, 1])
        all_easiness_tr_save = np.zeros([60000, 1])
        train_size = train_labels.shape[0]
        test_size = test_labels.shape[0]

        def main_network(data, train=False):
            """The Model definition."""
            conv1_m = tf.nn.conv2d(data,
                                   conv1_weights_m,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            relu1_m = tf.nn.relu(tf.nn.bias_add(conv1_m, conv1_biases_m))
            conv2_m = tf.nn.conv2d(relu1_m,
                                   conv2_weights_m,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            relu2_m = tf.nn.relu(tf.nn.bias_add(conv2_m, conv2_biases_m))
            pool1_m = tf.nn.max_pool(relu2_m,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
            if train:
                pool1_m = tf.nn.dropout(pool1_m, 0.75, seed=SEED)
            conv3_m = tf.nn.conv2d(pool1_m,
                                   conv3_weights_m,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            relu3_m = tf.nn.relu(tf.nn.bias_add(conv3_m, conv3_biases_m))
            conv4_m = tf.nn.conv2d(relu3_m,
                                   conv4_weights_m,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            relu4_m = tf.nn.relu(tf.nn.bias_add(conv4_m, conv4_biases_m))
            pool2_m = tf.nn.max_pool(relu4_m,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
            if train:
                pool2_m = tf.nn.dropout(pool2_m, 0.75, seed=SEED)
            pool_shape_m = pool2_m.get_shape().as_list()
            reshape_m = tf.reshape(
                pool2_m,
                [pool_shape_m[0], pool_shape_m[1] * pool_shape_m[2] * pool_shape_m[3]])
            hidden_m = tf.nn.relu(tf.matmul(reshape_m, fc1_weights_m) + fc1_biases_m)
            if train:
                hidden_m = tf.nn.dropout(hidden_m, 0.5, seed=SEED)
            return tf.matmul(hidden_m, fc2_weights_m) + fc2_biases_m

        def auxiliary_network(data, train1=False, train2 = False):
            """The Model definition."""
            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv1_a = tf.nn.conv2d(data,
                                   conv1_weights_a,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            # Bias and rectified linear non-linearity.

            #    conv1_bn_a = batch_norm(tf.nn.bias_add(conv1_a, conv1_biases_a), 32, phase_train)
            relu1_a = tf.nn.relu(conv1_a)
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool1_a = tf.nn.max_pool(relu1_a,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
            conv2_a = tf.nn.conv2d(pool1_a,
                                   conv2_weights_a,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
            #    conv2_bn_a = batch_norm(tf.nn.bias_add(conv2_a, conv2_biases_a), 64, phase_train)
            relu2_a = tf.nn.relu(conv2_a)
            pool2_a = tf.nn.max_pool(relu2_a,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape_a = pool2_a.get_shape().as_list()
            reshape_a = tf.reshape(
                pool2_a,
                [pool_shape_a[0], pool_shape_a[1] * pool_shape_a[2] * pool_shape_a[3]])
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            hidden_a = tf.nn.relu(tf.matmul(reshape_a, fc1_weights_a) + fc1_biases_a)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train1:
                hidden_a = tf.nn.dropout(hidden_a, 0.5, seed=SEED)
            pred_a = tf.matmul(hidden_a, fc2_weights_a) + fc2_biases_a

            # Calculate batch mean and variance
            batch_mean1_a, batch_var1_a = tf.nn.moments(pred_a, [0])

            # Apply the initial batch normalizing transform
            epsilon = 1e-3
            pred_hat_a = (pred_a - batch_mean1_a) / tf.sqrt(batch_var1_a + epsilon)
            logits_a = tf.nn.relu(pred_hat_a)
            predicted_hard = tf.nn.sigmoid(tf.matmul(logits_a, fc3_weights_a) + fc3_biases_a)



            # get the e_i predictor
            hidden_ae = tf.nn.relu(tf.matmul(hidden_a, fc4_weights_a) + fc4_biases_a)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train2:
                hidden_ae = tf.nn.dropout(hidden_ae, 0.5, seed=SEED)
            pred_ae = tf.matmul(hidden_ae, fc5_weights_a) + fc5_biases_a

            # Calculate batch mean and variance
            batch_mean1_ae, batch_var1_ae = tf.nn.moments(pred_ae, [0])

            # Apply the initial batch normalizing transform
            epsilon = 1e-3
            pred_hat_ae = (pred_ae - batch_mean1_ae) / tf.sqrt(batch_var1_ae + epsilon)
            logits_ae = tf.nn.relu(pred_hat_ae)
            error_score = tf.nn.sigmoid(tf.matmul(logits_ae, fc6_weights_a) + fc6_biases_a)

            #error_score = tf.nn.sigmoid(100.0 * (error_score - 0.5))
            return predicted_hard, error_score

        with tf.Graph().as_default() as graph:
            train_data_node = tf.placeholder(
                data_type(),
                shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input_node_train')
            train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,), name='gt_labels')
            eval_data = tf.placeholder(
                data_type(),
                shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), name='input_node_val')
            phase_train = tf.placeholder(tf.bool, name='phase_train')
            # main network parameters
            conv1_weights_m = tf.get_variable("conv1_w_m", [3, 3, NUM_CHANNELS, 32], trainable=True)
            conv1_biases_m = tf.get_variable("conv1_b_m", [32], trainable=True)
            conv2_weights_m = tf.get_variable("conv2_w_m", [3, 3, 32, 32], trainable=True)
            conv2_biases_m = tf.get_variable("conv2_b_m", [32], trainable=True)
            conv3_weights_m = tf.get_variable("conv3_w_m", [3, 3, 32, 64], trainable=True)
            conv3_biases_m = tf.get_variable("conv3_b_m", [64], trainable=True)
            conv4_weights_m = tf.get_variable("conv4_w_m", [3, 3, 64, 64], trainable=True)
            conv4_biases_m = tf.get_variable("conv4_b_m", [64], trainable=True)

            fc1_weights_m = tf.get_variable("fc1_w_m", [7 * 7 * 64, 512], trainable=True)
            fc1_biases_m = tf.get_variable("fc1_b_m", [512], trainable=True)
            fc2_weights_m = tf.get_variable("fc2_w_m", [512, NUM_LABELS], trainable=True)
            fc2_biases_m = tf.get_variable("fc2_b_m", [NUM_LABELS], trainable=True)

            # adaptive network parameters
            conv1_weights_a = tf.get_variable("conv1_w_a", [5, 5, NUM_CHANNELS, 32], trainable=True)
            conv1_biases_a = tf.get_variable("conv1_b_a", [32], trainable=True)
            conv2_weights_a = tf.get_variable("conv2_w_a", [5, 5, 32, 64], trainable=True)
            conv2_biases_a = tf.get_variable("conv2_b_a", [64], trainable=True)
            fc1_weights_a = tf.get_variable("fc1_w_a", [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], trainable=True)
            fc1_biases_a = tf.get_variable("fc1_b_a", [512], trainable=True)
            fc2_weights_a = tf.get_variable("fc2_w_a", [512, NUM_LABELS], trainable=True)
            fc2_biases_a = tf.get_variable("fc2_b_a", [NUM_LABELS], trainable=True)
            fc3_weights_a = tf.get_variable("fc3_w_a", [NUM_LABELS, 1], trainable=True)
            fc3_biases_a = tf.get_variable("fc3_b_a", [1], trainable=True)

            fc4_weights_a = tf.get_variable("fc4_w_a", [512, 512], trainable=True)
            fc4_biases_a = tf.get_variable("fc4_b_a", [512], trainable=True)
            fc5_weights_a = tf.get_variable("fc5_w_a", [512, NUM_LABELS], trainable=True)
            fc5_biases_a = tf.get_variable("fc5_b_a", [NUM_LABELS], trainable=True)
            fc6_weights_a = tf.get_variable("fc6_w_a", [NUM_LABELS, 1], trainable=True)
            fc6_biases_a = tf.get_variable("fc6_b_a", [1], trainable=True)

            batch = tf.Variable(0, dtype=data_type())

            # forward
            predicted_labels = main_network(train_data_node, True)
            predicted_hardness,_ = auxiliary_network(train_data_node, train1=True)
            predicted_hardness_te, predicted_error = auxiliary_network(train_data_node, train1=False, train2=True)
            # compute loss
            error = loss_error(predicted_labels, train_labels_node)
            loss_m_1 = loss_main_network(predicted_labels, predicted_hardness, train_labels_node)
            loss_m_2 = loss_main_network2(predicted_labels, predicted_hardness_te, train_labels_node, tf.nn.sigmoid(10000.0 * (predicted_error - 0.5)))
            loss_a = loss_adaptive_network(predicted_labels, predicted_hardness, train_labels_node)


            # p_i
            p_i = compute_p_i(predicted_labels, predicted_hardness, train_labels_node)

            variables_names = [v.name for v in tf.trainable_variables()]
            trained_variables_m = ['conv1_w_m:0', 'conv1_b_m:0', 'conv2_w_m:0', 'conv2_b_m:0', 'conv3_w_m:0',
                                   'conv3_b_m:0',
                                   'conv4_w_m:0', 'conv4_b_m:0', 'fc1_w_m:0', 'fc1_b_m:0', 'fc2_w_m:0', 'fc2_b_m:0']
            trained_variables_a = ['conv1_w_a:0', 'conv1_b_a:0', 'conv2_w_a:0', 'conv2_b_a:0', 'fc1_w_a:0', 'fc1_b_a:0', 'fc2_w_a:0', 'fc2_b_a:0', 'fc3_w_a:0', 'fc3_b_a:0']
            trained_variables_a1 = ['conv1_w_a:0', 'conv1_b_a:0', 'conv2_w_a:0', 'conv2_b_a:0', 'fc1_w_a:0', 'fc1_b_a:0', 'fc2_w_a:0', 'fc2_b_a:0', 'fc3_w_a:0', 'fc3_b_a:0']
            trained_variables_a2 = ['fc4_w_a:0', 'fc4_b_a:0', 'fc5_w_a:0', 'fc5_b_a:0', 'fc6_w_a:0', 'fc6_b_a:0']

            trained_variables_stage2 = ['conv1_w_m:0', 'conv1_b_m:0', 'conv2_w_m:0', 'conv2_b_m:0', 'conv3_w_m:0',
                                   'conv3_b_m:0',
                                   'conv4_w_m:0', 'conv4_b_m:0', 'fc1_w_m:0', 'fc1_b_m:0', 'fc2_w_m:0', 'fc2_b_m:0', 'fc4_w_a:0', 'fc4_b_a:0', 'fc5_w_a:0', 'fc5_b_a:0', 'fc6_w_a:0', 'fc6_b_a:0']

            var_list_m = [v for v in tf.trainable_variables() if v.name in trained_variables_m]
            #    trained_variables_a = ['conv1_w_a', 'conv1_b_a', 'conv2_w_a', 'conv2_b_a', 'fc1_w_a', 'fc1_b_a', 'fc2_w_a', 'fc2_b_a', 'fc3_w_a', 'fc3_b_a']
            var_list_a = [v for v in tf.trainable_variables() if v.name in trained_variables_a]
            var_list_a1 = [v for v in tf.trainable_variables() if v.name in trained_variables_a1]
            var_list_a2 = [v for v in tf.trainable_variables() if v.name in trained_variables_a2]
            var_list_stage2 = [v for v in tf.trainable_variables() if v.name in trained_variables_stage2]

            # L2 regularization for the fully connected parameters.
            regularizers_m = (tf.nn.l2_loss(fc1_weights_m) + tf.nn.l2_loss(fc1_biases_m) +
                              tf.nn.l2_loss(fc2_weights_m) + tf.nn.l2_loss(fc2_biases_m))
            regularizers_a2 = (tf.nn.l2_loss(conv1_weights_a) + tf.nn.l2_loss(conv1_biases_a) +
                               tf.nn.l2_loss(conv2_weights_a) + tf.nn.l2_loss(conv2_biases_a) +
                               tf.nn.l2_loss(fc4_weights_a) + tf.nn.l2_loss(fc4_biases_a) +
                               tf.nn.l2_loss(fc5_weights_a) + tf.nn.l2_loss(fc5_biases_a) +
                               tf.nn.l2_loss(fc6_weights_a) + tf.nn.l2_loss(fc6_biases_a))

            # Add the regularization term to the loss.
            loss_m_1 = loss_m_1 + 1e-3 * regularizers_m




            # l2_regularizer_error = tf.nn.l2_loss(tf.subtract(error, predicted_error)) / BATCH_SIZE

            coefficient_1 = tf.reshape(tf.tile(tf.expand_dims(tf.div(tf.reduce_sum(error), BATCH_SIZE), 0), [BATCH_SIZE]), [BATCH_SIZE,1])
            coefficient_2 = tf.reshape(tf.tile(tf.expand_dims((1.0 - tf.div(tf.reduce_sum(error), BATCH_SIZE)), 0), [BATCH_SIZE]), [BATCH_SIZE,1])
            # coefficient_1 = tf.multiply(tf.constant(0.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), coefficient_1)
            weights_for_balance = tf.multiply(tf.pow(coefficient_1, tf.subtract(
                tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), error)),
                                              tf.pow(coefficient_2, error))
            l1_regularizer_error = tf.reduce_sum(
                tf.multiply(weights_for_balance, tf.abs(tf.subtract(error, predicted_error)))) / BATCH_SIZE
            l2_regularizer_error = tf.reduce_sum(tf.multiply(weights_for_balance, tf.pow(tf.subtract(error, predicted_error), 2.0))) / BATCH_SIZE
            cross_entropy_regularizer_error = binary_cross_entropy(error, predicted_error)





            loss_m_2 = loss_m_2 + 1e-3 * regularizers_m
            # CUR_LAMBDA * tf.reduce_mean(predicted_error) + 10.0 * l2_regularizer_error + 1e-1 * regularizers_a2

            regularizers_a = (tf.nn.l2_loss(conv1_weights_a) + tf.nn.l2_loss(conv1_biases_a) +
                              tf.nn.l2_loss(conv2_weights_a) + tf.nn.l2_loss(conv2_biases_a) +
                              tf.nn.l2_loss(fc1_weights_a) + tf.nn.l2_loss(fc1_biases_a) +
                              tf.nn.l2_loss(fc2_weights_a) + tf.nn.l2_loss(fc2_biases_a) +
                              tf.nn.l2_loss(fc3_weights_a) + tf.nn.l2_loss(fc3_biases_a))

            # l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1, scope=None)
            # regularizers_a = tf.contrib.layers.apply_regularization(l1_regularizer, var_list_a)

            # Add the regularization term to the loss.
            loss_a = loss_a + 1e-3 * regularizers_a
            loss_a_2 = tf.reduce_mean(predicted_error) + CUR_LAMBDA * l1_regularizer_error + 1e-2 * regularizers_a2
            batch_mean_predicted_error, batch_var_predicted_error = tf.nn.moments(predicted_error, [0])
            loss_a_3 = l1_regularizer_error + 1e-1 * regularizers_a2
                       # - 800.0 * batch_var_predicted_error
            # l1_regularizer_error + 1e-2 * regularizers_a2 -

            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate_m = tf.train.exponential_decay(
                0.01,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                2 * train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)

            learning_rate_m2 = tf.train.exponential_decay(
                0.01,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                2 * train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)

            # Use simple momentum for the optimization.
            optimizer_m_1 = tf.train.MomentumOptimizer(learning_rate_m, 0.9).minimize(loss_m_1, global_step=batch,
                                                                                      var_list=var_list_m)
            optimizer_m_2 = tf.train.MomentumOptimizer(learning_rate_m2, 0.9).minimize(loss_m_2, global_step=batch,
                                                                                      var_list=var_list_m)

            learning_rate_a = tf.train.exponential_decay(
                0.0005,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
            learning_rate_a2 = tf.train.exponential_decay(
                0.01,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
            learning_rate_a3 = tf.train.exponential_decay(
                0.001,  # Base learning rate.
                batch * BATCH_SIZE,  # Current index into the dataset.
                train_size,  # Decay step.
                0.95,  # Decay rate.
                staircase=True)


            # Use simple momentum for the optimization.
            optimizer_a = tf.train.MomentumOptimizer(learning_rate_a, 0.9).minimize(loss_a, global_step=batch,
                                                                                    var_list=var_list_a)
            optimizer_a_2 = tf.train.MomentumOptimizer(learning_rate_a2, 0.9).minimize(loss_a_2, global_step=batch,
                                                                                    var_list=var_list_a2)
            optimizer_a_3 = tf.train.MomentumOptimizer(learning_rate_a3, 0.9).minimize(loss_a_3, global_step=batch,
                                                                                       var_list=var_list_a2)
            # Predictions for the current training minibatch.
            train_prediction = tf.nn.softmax(predicted_labels)
            # Predictions for the test and validation, which we'll compute less often.
            eval_prediction = tf.nn.softmax(main_network(eval_data))
            _, predicted_error_te = auxiliary_network(train_data_node, train1=False,
                                                                              train2=False)
            # predicted_error_te = tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)), predicted_error_te)
            predicted_error_te = tf.subtract(tf.constant(1.0, dtype=tf.float32, shape=(BATCH_SIZE, 1)),
                                         tf.nn.sigmoid(10000.0 * (predicted_error_te - 0.5)))

        def eval_in_batches(data, sess):
            """Get all predictions for a dataset by running it in small batches."""
            size = data.shape[0]
            if size < EVAL_BATCH_SIZE:
                raise ValueError("batch size for evals larger than dataset: %d" % size)
            predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
            for begin in xrange(0, size, EVAL_BATCH_SIZE):
                end = begin + EVAL_BATCH_SIZE
                if end <= size:
                    predictions[begin:end, :] = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[begin:end, ...]})
                else:
                    batch_predictions = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                    predictions[begin:, :] = batch_predictions[begin - size:, :]
            return predictions

        def eval_in_batches_predicted_easiness(data, sess):
            """Get all predictions for a dataset by running it in small batches."""
            size = data.shape[0]
            if size < EVAL_BATCH_SIZE:
                raise ValueError("batch size for evals larger than dataset: %d" % size)
            predicted_easiness_test = np.ndarray(shape=(size, 1), dtype=np.float32)
            for begin in xrange(0, size, EVAL_BATCH_SIZE):
                end = begin + EVAL_BATCH_SIZE
                if end <= size:
                    predicted_easiness_test[begin:end, :] = sess.run(predicted_error_te,
                                                                     feed_dict={train_data_node: data[begin:end, ...]})
                else:
                    batch_predicted_easiness_test = sess.run(predicted_error_te,
                                                             feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                    predicted_easiness_test[begin:, :] = batch_predicted_easiness_test[begin - size:, :]
            return predicted_easiness_test

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver()
        # Create a local session to run the training.
        start_time = time.time()

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph, config=config) as sess:

            # with tf.Session(graph=graph) as sess:
            # save main_network and adaptive network parameters
            saver_m = tf.train.Saver(
                {"conv1_w_m": conv1_weights_m, "conv1_b_m": conv1_biases_m, "conv2_w_m": conv2_weights_m,
                 "conv2_b_m": conv2_biases_m,
                 "conv3_w_m": conv3_weights_m, "conv3_b_m": conv3_biases_m, "conv4_w_m": conv4_weights_m, "conv4_b_m": conv4_biases_m, "fc1_w_m": fc1_weights_m,
                 "fc1_b_m": fc1_biases_m, "fc2_w_m": fc2_weights_m, "fc2_b_m": fc2_biases_m})
            saver_a = tf.train.Saver(
                {"conv1_w_a": conv1_weights_a, "conv1_b_a": conv1_biases_a, "conv2_w_a": conv2_weights_a,
                 "conv2_b_a": conv2_biases_a, "fc1_w_a": fc1_weights_a, "fc1_b_a": fc1_biases_a,
                 "fc2_w_a": fc2_weights_a, "fc2_b_a": fc2_biases_a, "fc3_w_a": fc3_weights_a, "fc3_b_a": fc3_biases_a})
            # Run all the initializers to prepare the trainable parameters.
            if FINE_TURE == True:
                saver_m.restore(sess,
                                "/data6/peiwang/projects/experiments/caffe-tensorflow-master/examples/mnist/tmp0to9_main_network/model.ckpt")
                saver_a.restore(sess,
                                "/data6/peiwang/projects/experiments/caffe-tensorflow-master/examples/mnist/tmp0to9_adaptive_network/model.ckpt")
                print("Model restored.")
                initialize_uninitialized_vars(sess)
            else:
                tf.global_variables_initializer().run()
            print('Initialized!')

            # Loop through training steps.
            all_easiness_hist = np.zeros([10000, 1])
            for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * BATCH_SIZE) % (train_size)
                batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
                # This dictionary maps the batch data (as a np array) to the
                # node in the graph it should be fed to.
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels, phase_train: True}
                # Run the optimizer to update weights.
                if step < int(EPOCH_STAGE1 * train_size) // BATCH_SIZE:
                    sess.run(optimizer_m_1, feed_dict=feed_dict)
                    sess.run(optimizer_a, feed_dict=feed_dict)
                    # sess.run(optimizer_a_3, feed_dict=feed_dict)
                else:
                    sess.run(optimizer_m_2, feed_dict=feed_dict)
                    sess.run(optimizer_a_2, feed_dict=feed_dict)




                # if step == int(EPOCH_STAGE1 * train_size) // BATCH_SIZE:
                #     # save the hardest scores
                #     all_easiness_te_view = np.zeros([10000, 1])
                #     for step_te_view in xrange(int(test_size) // BATCH_SIZE):
                #         offset_te_view = (step_te_view * BATCH_SIZE) % (test_size)
                #         batch_data_te_view = test_data[offset_te_view:(offset_te_view + BATCH_SIZE), ...]
                #         batch_labels_te_view = test_labels[offset_te_view:(offset_te_view + BATCH_SIZE)]
                #         feed_dict_te_view = {train_data_node: batch_data_te_view,
                #                         train_labels_node: batch_labels_te_view, phase_train: False}
                #         easiness_view = sess.run(predicted_hardness, feed_dict=feed_dict_te_view)
                #         all_easiness_te_view[offset_te_view:(offset_te_view + BATCH_SIZE), 0] = np.squeeze(easiness_view)
                #
                #     SAVE_PATH_si_test = '/data6/peiwang/projects/experiments/caffe-tensorflow-master/examples/mnist/predicted_hardness_scores_view.txt'
                #     fl_tr = open(SAVE_PATH_si_test, 'w')
                #     for i in range(all_easiness_te_view.shape[0]):
                #         save_info = str(all_easiness_te_view[i, 0])
                #         fl_tr.write(save_info)
                #         fl_tr.write("\n")
                #     fl_tr.close()


                # cur_p_i = sess.run(p_i,feed_dict=feed_dict)
                # print (np.squeeze(cur_p_i))
                # # save train infomation after each iteration
                # all_loss_each_iter_tr = np.zeros([60000,1])
                # all_pit_each_iter_tr = np.zeros([60000,1])
                #
                # if step % 10 == 0 and step > 0:
                # # if step % 100 == 0 or step < 30:
                #     num_judger = num_judger + 1
                #     for step_tr in xrange(int(int(train_size) // BATCH_SIZE)):
                #         offset_tr = (step_tr * BATCH_SIZE) % (train_size)
                #         batch_data_tr = train_data[offset_tr:(offset_tr + BATCH_SIZE), ...]
                #         batch_labels_tr = train_labels[offset_tr:(offset_tr + BATCH_SIZE)]
                #         feed_dict = {train_data_node: batch_data_tr,
                #                      train_labels_node: batch_labels_tr}
                #         # # get gt loss
                #         # cur_loss = sess.run(loss, feed_dict=feed_dict)
                #         # cur_loss = cur_loss
                #         # all_loss_each_iter_tr[step_tr,0] = cur_loss
                #
                #         # get gt p_i_t
                #         cur_prediction = sess.run(train_prediction, feed_dict={train_data_node: batch_data_tr})
                #         # print (cur_prediction)
                #         # print (batch_labels_tr)
                #         cur_pit = [cur_prediction[i][b_i] for i, b_i in enumerate(batch_labels_tr)]
                #         cur_pit = cur_pit
                #         # print (cur_pit)
                #         all_pit_each_iter_tr[offset_tr:(offset_tr + BATCH_SIZE),0] = cur_pit
                #         # print (all_pit_each_iter_tr)
                #
                #
                #     for i in range(all_pit_each_iter_tr.shape[0]):
                #         all_loss_each_iter_tr[i,0] = (-1.0) *math.log(all_pit_each_iter_tr[i,0])
                #
                #     total_all_loss_each_iter_tr = np.concatenate((total_all_loss_each_iter_tr, all_loss_each_iter_tr), axis=1)
                #     total_all_pit_each_iter_tr = np.concatenate((total_all_pit_each_iter_tr, all_pit_each_iter_tr),axis=1)
                #
                #     # total_all_loss_each_iter_tr = total_all_loss_each_iter_tr + all_loss_each_iter_tr
                #     # total_all_pit_each_iter_tr = total_all_pit_each_iter_tr + all_pit_each_iter_tr
                #
                # # save test information after each iteration
                #
                # all_loss_each_iter_te = np.zeros([10000,1])
                # all_pit_each_iter_te = np.zeros([10000,1])
                # if step % 10 == 0 and step > 0:
                # # if step % 100 == 0 or step < 30:
                #     for step_te in xrange(int(test_size) // BATCH_SIZE):
                #
                #         offset_te = (step_te * BATCH_SIZE) % (test_size)
                #         batch_data_te = test_data[offset_te:(offset_te + BATCH_SIZE), ...]
                #         batch_labels_te = test_labels[offset_te:(offset_te + BATCH_SIZE)]
                #         feed_dict = {train_data_node: batch_data_te,
                #                      train_labels_node: batch_labels_te}
                #         cur_prediction = sess.run(train_prediction, feed_dict={train_data_node: batch_data_te})
                #         cur_pit = [cur_prediction[i][b_i] for i, b_i in enumerate(batch_labels_te)]
                #         cur_pit = cur_pit
                #         all_pit_each_iter_te[offset_te:(offset_te + BATCH_SIZE),0] = cur_pit
                #     for i in range(all_pit_each_iter_te.shape[0]):
                #         all_loss_each_iter_te[i, 0] = (-1.0) * math.log(all_pit_each_iter_te[i, 0])
                #
                #
                #     total_all_loss_each_iter_te = np.concatenate((total_all_loss_each_iter_te, all_loss_each_iter_te), axis=1)
                #     total_all_pit_each_iter_te = np.concatenate((total_all_pit_each_iter_te, all_pit_each_iter_te),axis=1)



                # print some extra information once reach the evaluation frequency
                if step % EVAL_FREQUENCY == 0:

                    # if step > int(EPOCH_STAGE1 * train_size) // BATCH_SIZE:
                    print ('offset is', offset)
                    cur_hardness_view = sess.run(predicted_hardness_te, feed_dict=feed_dict)
                    print ('current predicted hardness scores')
                    print (cur_hardness_view.reshape(BATCH_SIZE)[0:20])
                    cur_predicted_error_view = sess.run(predicted_error, feed_dict=feed_dict)
                    print ('current predicted error results')
                    print(cur_predicted_error_view.reshape(BATCH_SIZE)[0:20])
                    cur_fake_error_view, cur_weights_for_balance = sess.run([error, weights_for_balance], feed_dict=feed_dict)
                    print ('true error results')
                    print(cur_fake_error_view.reshape(BATCH_SIZE)[0:20])
                    print(cur_weights_for_balance.reshape(BATCH_SIZE)[0:20])
                    print(sess.run(loss_main_network2(predicted_labels, predicted_hardness_te, train_labels_node, tf.nn.sigmoid(100.0 * (predicted_error - 0.5))), feed_dict=feed_dict))
                    print(sess.run(tf.reduce_mean(predicted_error), feed_dict=feed_dict))
                    print(sess.run(l1_regularizer_error, feed_dict=feed_dict))


                    # print(sess.run(coefficient_1, feed_dict=feed_dict))

                    # fetch some extra nodes' data
                    lr1, lr2, predictions = sess.run(
                        [learning_rate_m, learning_rate_m2, train_prediction],
                        feed_dict=feed_dict)


                    if step < int(EPOCH_STAGE1 * train_size) // BATCH_SIZE:
                        lr1, lr2, predictions, cur_loss_m, cur_loss_a, cur_error, cur_predicted_error = sess.run(
                            [learning_rate_m, learning_rate_a, train_prediction, loss_m_1, loss_a, error, predicted_error_te],
                            feed_dict=feed_dict)

                    else:
                        lr1, lr2, predictions, cur_loss_m, cur_loss_a, cur_error, cur_predicted_error = sess.run(
                            [learning_rate_m2, learning_rate_a2, train_prediction, loss_m_2, loss_m_2, error, predicted_error_te],
                            feed_dict=feed_dict)


                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %
                          (step, float(step) * BATCH_SIZE / train_size,
                           1000 * elapsed_time / EVAL_FREQUENCY))
                    print('Minibatch learning rate1: %.6f, learning rate2: %.6f' % (lr1, lr2))
                    print('current batch loss of main network: %.5f' % cur_loss_m)
                    print('current batch loss of auxiliary: %.5f' % cur_loss_a)
                    print('current batch error: %.5f' % np.sum(cur_error))

                    if step < int(EPOCH_STAGE1 * train_size) // BATCH_SIZE:
                        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                        total_test_error = error_rate(
                            eval_in_batches(validation_data, sess), validation_labels)
                        print('Validation error: %.6f%%' % total_test_error)
                    else:
                        print('Minibatch error: %.1f%%' % realistic_error_rate(predictions, batch_labels, cur_predicted_error))
                        total_test_error = realistic_error_rate(
                            eval_in_batches(validation_data, sess), validation_labels,
                            eval_in_batches_predicted_easiness(validation_data, sess))
                        print('Validation error: %.6f%%' % total_test_error)

                    if total_test_error < accuracy_temp and total_test_error > 0.1:
                        accuracy_temp = total_test_error

                        all_easiness_te = np.zeros([10000, 1])
                        for step_te in xrange(int(test_size) // BATCH_SIZE):
                            offset_te = (step_te * BATCH_SIZE) % (test_size)
                            batch_data_te = test_data[offset_te:(offset_te + BATCH_SIZE), ...]
                            batch_labels_te = test_labels[offset_te:(offset_te + BATCH_SIZE)]
                            feed_dict_te = {train_data_node: batch_data_te,
                                            train_labels_node: batch_labels_te, phase_train: False}
                            easiness = sess.run(predicted_error_te, feed_dict=feed_dict_te)
                            all_easiness_te[offset_te:(offset_te + BATCH_SIZE), 0] = np.squeeze(easiness)

                        all_easiness_tr = np.zeros([60000, 1])
                        for step_tr in xrange(int(train_size) // BATCH_SIZE):
                            offset_tr = (step_tr * BATCH_SIZE) % (train_size)
                            batch_data_tr = train_data[offset_tr:(offset_tr + BATCH_SIZE), ...]
                            batch_labels_tr = train_labels[offset_tr:(offset_tr + BATCH_SIZE)]
                            feed_dict_tr = {train_data_node: batch_data_tr,
                                            train_labels_node: batch_labels_tr, phase_train: False}
                            easiness = sess.run(predicted_error_te, feed_dict=feed_dict_tr)
                            all_easiness_tr[offset_tr:(offset_tr + BATCH_SIZE), 0] = np.squeeze(easiness)

                        all_easiness_te_save = all_easiness_te
                        all_easiness_tr_save = all_easiness_tr
                    sys.stdout.flush()

            final_all_easiness_te = np.zeros([10000, 1])
            for step_te in xrange(int(test_size) // BATCH_SIZE):
                offset_te = (step_te * BATCH_SIZE) % (test_size)
                batch_data_te = test_data[offset_te:(offset_te + BATCH_SIZE), ...]
                batch_labels_te = test_labels[offset_te:(offset_te + BATCH_SIZE)]
                feed_dict_te = {train_data_node: batch_data_te,
                                train_labels_node: batch_labels_te, phase_train: False}
                easiness = sess.run(predicted_error_te, feed_dict=feed_dict_te)
                final_all_easiness_te[offset_te:(offset_te + BATCH_SIZE), 0] = np.squeeze(easiness)

        sess.close()
        return accuracy_temp, all_easiness_te_save, all_easiness_tr_save, final_all_easiness_te, total_test_error

    if FLAGS.self_test:
        print('Running self-test.')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
        test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
        num_epochs = 1
    else:
        # Get the data.
        train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
        train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
        test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
        test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

        # Extract it into numpy arrays.
        train_data = extract_data(train_data_filename, 60000)
        train_labels = extract_labels(train_labels_filename, 60000)
        test_data = extract_data(test_data_filename, 10000)
        test_labels = extract_labels(test_labels_filename, 10000)

        # # # randomly shuffle labels
        # picked_shuffle = train_labels[0:30000]
        # picked_shuffle.tolist()
        # random.shuffle(picked_shuffle)
        # picked_shuffle = np.array(picked_shuffle)
        # train_labels[0:30000] = picked_shuffle
        #
        # train_data_list = []
        # for i in range(train_labels.shape[0]):
        #     train_data_list.append(train_data[i, :, :, :])
        # train_labels = train_labels.tolist()
        # z = zip(train_data_list, train_labels)
        # random.shuffle(z)
        # train_data_list, train_labels = [list(l) for l in zip(*z)]
        # new_train_data = np.zeros((len(train_labels), 28, 28, 1))
        # for i in range(len(train_labels)):
        #     new_train_data[i, :, :, :] = train_data_list[i]
        # train_data = new_train_data
        # train_labels = np.array(train_labels)


        # Generate a validation set.
        validation_data = test_data
        validation_labels = test_labels
        # train_data = train_data[VALIDATION_SIZE:, ...]
        # train_labels = train_labels[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]
    test_size = test_labels.shape[0]
    total_acc = []
    all_tr_easiness = np.zeros([LAMBDA_POOL.shape[0], 5, 60000, 1])
    all_te_easiness = np.zeros([LAMBDA_POOL.shape[0], 5, 10000, 1])
    final_all_te_easiness = np.zeros([LAMBDA_POOL.shape[0], 5, 10000, 1])
    accuracy = np.zeros([LAMBDA_POOL.shape[0], 5])
    final_accuracy = np.zeros([LAMBDA_POOL.shape[0], 5])
    for i in range(LAMBDA_POOL.shape[0]):
        CUR_LAMBDA = LAMBDA_POOL[i]
        print('current lambda is ', CUR_LAMBDA)
        for j in range(5):
            accuracy_tmp, all_easiness_te, all_easiness_tr, final_all_easiness_te, final_te_acc = train(train_data, train_labels, validation_data,
                                                                   validation_labels, test_data, test_labels,
                                                                   CUR_LAMBDA)
            accuracy[i, j] = accuracy_tmp
            final_accuracy[i, j] = final_te_acc
            all_tr_easiness[i, j, :, :] = all_easiness_tr
            all_te_easiness[i, j, :, :] = all_easiness_te
            final_all_te_easiness[i, j, :, :] = final_all_easiness_te
            sio.savemat('all_tr_easiness_save_different_lambda.mat', {'all_tr_easiness': all_tr_easiness})
            sio.savemat('all_te_easiness_save_different_lambda.mat', {'all_te_easiness': all_te_easiness})
            sio.savemat('final_all_te_easiness_save_different_lambda.mat', {'final_all_te_easiness': final_all_te_easiness})
            sio.savemat('acc_save_different_lambda.mat', {'accuracy': accuracy})
            sio.savemat('final_acc_save_different_lambda.mat', {'final_accuracy': final_accuracy})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
