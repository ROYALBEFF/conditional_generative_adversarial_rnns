import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.examples.tutorials.mnist import input_data


def G(z, y):
    '''
    Generator

    Recurrent neural network with input **z** (noise vector) and context **y**.
    **z** contains the whole input sequence and will be split into **step_size** (globale variable) parts that are
    then be processed by the network sequentially.

    :param tf.Tensor z:
        Tensor containing the generator's input sequence.
    :param tf.Tensor y:
        Context tensor. One hot representation of the label.
    :return:
        Generator's output vector representing MNIST-like data regarding the context vector.
    '''
    # split input into step_size parts, which form the actual input sequence
    z_ = tf.split(z, step_size, 1)
    # concatenate the context vector to each part of the input sequence
    z_ = list(map(lambda t: tf.concat([t, y], 1), z_))

    # first RNN layer
    with tf.variable_scope("G_hidden"):
        cell_h1 = rnn.BasicRNNCell(noise_size + step_size * context_size)
        (h1, _) = rnn.static_rnn(cell_h1, z_, dtype=tf.float32)
        h1 = list(map(lambda t: tf.nn.relu(tf.matmul(t, G_W['h1']) + G_b['h1']), h1))

    # RNN output layer
    with tf.variable_scope("G_out"):
        cell_out = rnn.BasicRNNCell(392)
        (h, _) = rnn.static_rnn(cell_out, h1, dtype=tf.float32)
        logits = tf.matmul(h[-1], G_W['out']) + G_b['out']

    return tf.sigmoid(logits)


def D(x, y):
    '''
    Discriminator

    Recurrent neural network with input **x** (MNIST-like data) and context **y**.
    **x** contains the whole input sequence and will be split into **step_size** (globale variable) parts that are
    then be processed by the network sequentially.

    :param tf.Tensor x:
        Tensor containing the discriminator's input sequence. x represents MNIST-like data regarding the context vector.
    :param tf.Tensor y:
        Context tensor. One hot representation of the label.
    :return:
        Tuple containing the discriminator's output before applying the output layer's activation function (logits) and
        after applying the activation function.
        The logits are needed for the loss function.
        The actual output scalar is a value between 0 and 1 classifying the input sequence as real (1) or fake (0).

    '''
    # split input into step_size parts, which form the actual input sequence
    x_ = tf.split(x, step_size, 1)
    # concatenate the context vector to each part of the input sequence
    x_ = list(map(lambda t: tf.concat([t, y], 1), x_))

    # first RNN layer
    with tf.variable_scope("G_hidden"):
        cell_h1 = rnn.BasicRNNCell(784 + step_size * context_size)
        (h1, _) = rnn.static_rnn(cell_h1, x_, dtype=tf.float32)
        h1 = list(map(lambda t: tf.nn.relu(tf.matmul(t, D_W['h1']) + D_b['h1']), h1))

    # RNN output layer
    with tf.variable_scope("G_out"):
        cell_out = rnn.BasicRNNCell(392)
        (h, _) = rnn.static_rnn(cell_out, h1, dtype=tf.float32)
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


def random_context(n):
    '''
    Generates **n** one hot vectors representing a random digit each.

    :param int n:
        Number of context vectors to generate.
    :return:
        List containing **n** random context vectors.
    '''
    # draw n random digits and make a list of one hot vectors
    context = list(map(one_hot_vector, [random.randint(0, 9) for _ in range(n)]))
    return context


def one_hot_vector(index):
    '''
    Creates a one hot vector with 1 at position **index**.

    :param int index:
        Position of 1 in the one hot vector.
    :return:
        List representing a one hot vector with 1 at position **index**.
    '''
    zeros = [0] * (context_size - 1)
    zeros.insert(index, 1)
    return zeros


def plot(samples, labels):
    '''
    Plot generated samples and their labels in a 4x4 grid.

    :param np.array samples:
        Array containing the generated samples.
        Should contain 16 samples, because the plot figure is a 4x4 grid.
    :param [[int]] labels:
        List containing the corresponding labels represented by one hot vectors.
    :return:
        Figure object containing the plot.
    '''
    # create figure object with figure size 4x4 inches
    fig = plt.figure(figsize=(4, 4))
    # create 4x4 grid in which the subplots are arranged with some horizontal and vertical space
    grid = grd.GridSpec(4, 4, wspace=0.05, hspace=0.5)

    # plot all samples
    for (i, sample) in enumerate(samples):
        # get ith subplot axis
        ax = plt.subplot(grid[i])
        ax.set_title(str(labels[i].index(1)))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CGARNN model on the MNIST data set.")
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
    # one hot vector for each digit => size of 10
    context_size = 10
    # length of input sequence, image (28x28) will be separated into step_size lines
    step_size = args.stepsize
    # learning rate and decay for generator
    G_alpha = args.galpha
    # learning rate and decay for discriminator
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
    # generators weights for RNN output layer
    G_W = {
        'h1' : tf.get_variable("G_W_h", [noise_size + step_size * context_size, 392], initializer=xavier),
        'out': tf.get_variable("G_W_out", [392, 784], initializer=xavier)
    }
    # generator's bias for RNN output layer
    G_b = {
        'h1' : tf.Variable(tf.zeros([392]), name="G_b_h"),
        'out': tf.Variable(tf.zeros([784]), name="G_b_out")
    }

    # discriminator's input (real data), shape of [None, 784] means that there can be an arbitrary number of inputs of shape
    # [784], which allows batches for input
    x = tf.placeholder(tf.float32, [None, 784], "x")
    # discriminator's weights for RNN output layer
    D_W = {
        'h1' : tf.get_variable("D_W_h", [784 + step_size * context_size, 392], initializer=xavier),
        'out': tf.get_variable("D_W_out", [392, 1], initializer=xavier)
    }
    # discriminator's bias for RNN output layer
    D_b = {
        'h1' : tf.Variable(tf.zeros([392]), name="D_b_h"),
        'out': tf.Variable(tf.zeros([1]), name="D_b_out")
    }

    # context variable (label) for both generator and discriminator, shape of [None, context_size] means that there can be
    # an arbitrary number of contexts of shape [context_size], which allows batches for context
    y = tf.placeholder(tf.float32, [None, context_size], "y")

    # generators output
    G_ = G(z, y)
    # D_real and D_fake must share its RNN variables because they represent the same network with different inputs, this is
    # why we use a variable scope D here
    with tf.variable_scope("D") as scope:
        # discrimintors rating of real data
        (D_real_logits, D_real) = D(x, y)
        scope.reuse_variables()
        # discriminators rating of "fake" (generated) data
        (D_fake_logits, D_fake) = D(G_, y)

    # generator's loss function (cross-entropy with logits), discriminator should rate fake data as 0, if fake data is
    # rated ~1, the generated data looks likes real data to the discriminator, so the generators goal is to maximize the
    # ~1-rated data
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    # generator's optimizer function minimizes the difference between discriminators rating of the generated data and 1
    G_optimizer = tf.train.AdamOptimizer(learning_rate=G_alpha).minimize(G_loss, var_list=[G_W['h1'], G_W['out'], G_b['out'], G_b['out']])

    # discriminator's loss function for real data, real data should be rated as 1, loss describes difference between actual
    # rating and target rating
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))

    # discriminator's loss function for fake data, fake data should be rated as 0, loss describes difference between actual
    # rating and target rating
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
    # discriminator's loss function is a combination of both the loss for fake data and the loss for real data
    D_loss = D_loss_fake + D_loss_real

    # discriminator's optimizer function minimizes the difference between discriminators rating of real and fake data and the
    # target rating
    D_optimizer = tf.train.AdamOptimizer(learning_rate=D_alpha).minimize(D_loss, var_list=[D_W['h1'], D_W['out'], D_b['out'], D_b['out']])

    print("Done!")
    print("Initialize variables... ", end='')

    # start session and initialise variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print("Done!")
    print("Loading data set... ", end='')

    # get MNIST data set with labels as one hot vector
    mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)

    print("Done!")

    # create directory if it doesn;t exist yet
    if not os.path.exists('cgarnn_out/'):
        os.makedirs('cgarnn_out/')

    # plot counter
    n = 0
    # file for saving loss values
    file = open("cgarnn_out/loss.csv", "w")
    file.write("epoch, g_loss, d_loss\n")

    print("Training network... ")
    # train network
    for e in range(epochs):
        # train discriminator for k steps
        for _ in range(k):
            (x_, y_) = mnist.train.next_batch(batch_size)
            sess.run(D_optimizer, {x: x_, y: y_, z: noise(batch_size)})
        # train discriminator for one step
        z_ = noise(batch_size)
        y_ = random_context(batch_size)
        sess.run(G_optimizer, {y: y_, z: z_})

        if e % 1000 == 0:
            labels = random_context(16)
            # plot samples
            samples = sess.run(G_, {y: labels, z: noise(16)})
            fig = plot(samples, labels)
            # save plot in out/ directory
            plt.savefig('cgarnn_out/{}.png'.format(str(n).zfill(3)))
            n += 1
            plt.close(fig)

            # print generator and discriminator loss
            print("Epoch: %d" % e)
            print("G_LOSS: ")
            print(sess.run(G_loss, {y: y_, z: z_}))
            print("D_LOSS: ")
            print(sess.run(D_loss, {x: x_, y: y_, z: z_}))
            print()

            # write current loss values to csv file
            file.write("%d, %f, %f\n" % (e, sess.run(G_loss, {y: y_, z: z_}), sess.run(D_loss, {x: x_, y: y_, z: z_})))

    print("Done!")
    file.close()
    exit(0)