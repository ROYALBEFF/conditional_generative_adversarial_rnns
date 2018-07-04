import argparse
import matplotlib.gridspec as grd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data


def G(z):
    '''
    Generator

    Recurrent neural network with input z (noise vector). z contains the whole input sequence and will be split into
    step_size (globale variable) parts that are then be processed by the network sequentially.

    :param tf.Tensor z:
        Tensor containing the generator's input sequence.
    :return:
        Generator's output vector representing MNIST-like data.

    '''
    # split input into step_size parts, which form the actual input sequence
    z_ = tf.split(z, step_size, 1)

    # first RNN layer
    with tf.variable_scope("G_hidden"):
        cell_hidden = rnn.BasicRNNCell(noise_size)
        (h, _) = rnn.static_rnn(cell_hidden, z_, dtype=tf.float32)
        h = list(map(lambda t: tf.nn.relu(tf.matmul(t, G_W['hidden']) + G_b['hidden']), h))

    # RNN output layer
    with tf.variable_scope("G_out"):
        cell_out = rnn.BasicRNNCell(392)
        (h, _) = rnn.static_rnn(cell_out, h, dtype=tf.float32)
        logits = tf.matmul(h[-1], G_W['out']) + G_b['out']

    return tf.sigmoid(logits)


def D(x):
    '''
    Discriminator

    Recurrent neural network with input x (MNIST-like data). x contains the whole input sequence and will be split into
    step_size (globale variable) parts that are then be processed by the network sequentially.

    :param tf.Tensor x:
        Tensor containing the discriminator's input sequence. x represents MNIST-like data.
    :return:
        Tuple containing the discriminator's output before applying the output layer's activation function (logits) and
        after applying the activation function.
        The logits are needed for the loss function.
        The actual output scalar is a value between 0 and 1 classifying the input sequence as real (1) or fake (0).

    '''
    # split input into step_size parts, which form the actual input sequence
    x_ = tf.split(x, step_size, 1)

    # first RNN layer
    with tf.variable_scope("D_hidden"):
        cell_hidden = rnn.BasicRNNCell(784)
        (h, _) = rnn.static_rnn(cell_hidden, x_, dtype=tf.float32)
        h = list(map(lambda t: tf.nn.relu(tf.matmul(t, D_W['hidden']) + D_b['hidden']), h))

    # RNN output layer
    with tf.variable_scope("D_out"):
        cell_out = rnn.BasicRNNCell(392)
        (h, _) = rnn.static_rnn(cell_out, h, dtype=tf.float32)
        logits = tf.matmul(h[-1], D_W['out']) + D_b['out']

    return (logits, tf.sigmoid(logits))


def noise(n):
    '''
    Generates **n** random noise samples of shape **noise_size** (global variable).
    The random values are drawn uniformly from the interval [-1, 1].

    :param int n:
        Size of each noise sample.
    :return:
        Array containing **n** random noise samples.
    '''
    # uniform means all the values got the same probability to be drawn
    return np.random.uniform(-1, 1, (n, noise_size))


def plot(samples):
    '''
    Plot generated samples in a 4x4 grid.

    :param np.array samples:
        Array containing the generated samples.
        Should contain 16 samples, because the plot figure is a 4x4 grid.
    :return:
        Figure object containing the plot.
    '''
    # create figure object with figure size 4x4 inches
    fig = plt.figure(figsize=(4, 4))
    # create 4x4 grid in which the subplots are arranged
    grid = grd.GridSpec(4, 4, wspace=0.05, hspace=0.5)

    # create 4x4 grid in which the subplots are arranged with some horizontal and vertical space
    for (i, sample) in enumerate(samples):
        # get ith subplot axis
        ax = plt.subplot(grid[i])
        # turn off the axis lines and labels
        plt.axis('off')
        # set x- and y-axis labels to [] (no labels)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # same scaling for x and y
        ax.set_aspect('equal')
        # scale ith sample to 28x28 px and plot
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the GARNN model on the MNIST data set.")
    parser.add_argument("--batchsize", metavar="INTEGER", default=64, type=int,
                        help="size of mini-batch (default: 64)")
    parser.add_argument("--epochs", metavar="INTEGER", default=50000, type=int,
                        help="number of training iterations (default: 50000)")
    parser.add_argument("--noisesize", metavar="INTEGER", default=100, type=int,
                        help="size of the random noise vector that is used as the generator's input (default: 100)")
    parser.add_argument("--galpha", metavar="FLOAT", default=0.001, type=float,
                        help="generator's learning rate (default: 0.001)")
    parser.add_argument("--dalpha", metavar="FLOAT", default=0.001, type=float,
                        help="discriminator's learning rate (default: 0.001)")
    parser.add_argument("--stepsize", metavar="INTEGER", default=4, type=int,
                        help="number of steps in which the data will be processed (default: 4)")
    parser.add_argument("-k", metavar="INTEGER", default=1, type=int,
                        help="number of discriminator's training steps per training step of the generator")

    args = parser.parse_args()

    print("Building network graph... ", end='')
    # network parameters
    batch_size = args.batchsize
    epochs = args.epochs
    noise_size = args.noisesize
    # length of input sequence, image (28x28) will be separated into step_size chunks
    step_size = args.stepsize
    # learning rate  for generator
    G_alpha = args.galpha
    # learning rate for discriminator
    D_alpha = args.dalpha
    # k is the number of learning steps of the discriminator, while the discriminator takes k steps the generator only
    # takes one
    k = args.k

    # Xavier variable initializer and random seed
    seed = 611357224
    xavier = tf.glorot_normal_initializer(seed=seed)

    # generator's input (noise samples), shape of [None, noise_size] means that there can be an arbitrary number of inputs of
    # shape [noise_size], which allows batches for input
    z = tf.placeholder(tf.float32, [None, noise_size], "z")
    # generator's weights for RNN output layer
    G_W = {
        'hidden': tf.get_variable("G_W_h", [noise_size, 392], initializer=xavier),
        'out': tf.get_variable("G_W_out", [392, 784], initializer=xavier)
    }
    # generator's bias for RNN output layer
    G_b = {
        'hidden': tf.Variable(tf.zeros([392]), name="G_b_h"),
        'out': tf.Variable(tf.zeros([784]), name="G_b_out")
    }

    # discriminator's input (real data), shape of [None, 784] means that there can be an arbitrary number of inputs of shape
    # [784], which allows batches for input
    x = tf.placeholder(tf.float32, [None, 784], "x")
    # discriminators weights for RNN output layer
    D_W = {
        'hidden': tf.get_variable("D_W_h", [784, 392], initializer=xavier),
        'out': tf.get_variable("D_W_out", [392, 1], initializer=xavier)
    }
    # discriminator's bias for RNN output layer
    D_b = {
        'hidden': tf.Variable(tf.zeros([392]), name="D_b_h"),
        'out': tf.Variable(tf.zeros([1]), name="D_b_out")
    }

    # generator's output
    G_ = G(z)
    # D_real and D_fake must share its RNN variables because they represent the same network with different inputs, this is
    # why we use a variable scope D here
    with tf.variable_scope("D") as scope:
        # discrimintors rating of real data
        (D_real_logits, D_real) = D(x)
        scope.reuse_variables()
        # discriminators rating of "fake" (generated) data
        (D_fake_logits, D_fake) = D(G_)

    # generator's loss function (cross-entropy with logits), discriminator should rate fake data as 0, if fake data is
    # rated ~1, the generated data looks likes real data to the discriminator, so the generators goal is to maximize the
    # ~1-rated data
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    # generator's optimizer function minimizes the difference between discriminator's rating of the generated data and 1
    G_optimizer = tf.train.AdamOptimizer(learning_rate=G_alpha).minimize(G_loss,
                                                                         var_list=[G_W['hidden'], G_W['out'], G_b['hidden'],
                                                                                   G_b['out']])

    # discriminator's loss function for real data, real data should be rated as 1, loss describes difference between actual
    # rating and target rating
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
    # discriminator's loss function for fake data, fake data should be rated as 0, loss describes difference between actual
    # rating and target rating
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
    # discriminators loss function is a combination of both the loss for fake data and the loss for real data
    D_loss = D_loss_fake + D_loss_real

    # discriminator's optimizer function minimizes the difference between discriminators rating of real and fake data and the
    # target rating
    D_optimizer = tf.train.AdamOptimizer(learning_rate=D_alpha).minimize(D_loss,
                                                                         var_list=[D_W['hidden'], D_W['out'], D_b['hidden'],
                                                                                   D_b['out']])

    print("Done!")
    print("Initialize variables... ", end='')

    # start session and initialise variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print("Done!")
    print("Loading data set... ", end='')

    # get mnist data and labels
    mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)

    print("Done!")

    # create directory if it doesn't exist yet
    if not os.path.exists('GARNN_out/'):
        os.makedirs('GARNN_out/')

    # create log file for loss progression
    file = open("GARNN_out/loss.csv", "w+")
    # write csv header
    file.write("epoch, g_loss, d_loss\n")
    # plot counter
    n = 0

    print("Training network... ")
    # train network
    for e in range(epochs):
        # train discriminator for k steps
        for _ in range(k):
            (x_, _) = mnist.train.next_batch(batch_size)
            sess.run(D_optimizer, {x: x_, z: noise(batch_size)})
            # train discriminator for one step
        z_ = noise(batch_size)
        sess.run(G_optimizer, {z: z_})

        # plot samples and print loss values every 1000 epoch
        if e % 1000 == 0:
            # plot samples
            samples = sess.run(G_, {z: noise(16)})
            fig = plot(samples)
            # save plot in out/ directory
            plt.savefig('GARNN_out/{}.png'.format(str(n).zfill(3)))
            n += 1
            plt.close(fig)

            # print generator and discriminator loss
            print("Epoch: %d" % e)
            print("G_LOSS: ")
            print(sess.run(G_loss, {x: x_, z: z_}))
            print("D_LOSS: ")
            print(sess.run(D_loss, {x: x_, z: z_}))
            print()

            # write current loss values to csv file
            file.write("%d, %f, %f\n" % (e, sess.run(G_loss, {z: z_}), sess.run(D_loss, {x: x_, z: z_})))

    print("Done!")
    file.close()
    exit(0)
