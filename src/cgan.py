import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def G(z, y):
    '''
    Generator

    Two layer neural network that generates MNIST-like data.
    Concatenates the noise vector and the context vector and then uses it as the network's input.

    :param tf.Tensor z:
        Input tensor. Random noise vector.
    :param tf.Tensor y:
        Context tensor. One hot representation of the label.
    :return:
        Generator's output vector representing MNIST-like data regarding the context vector.
    '''
    z_ = tf.concat([z, y], 1, "z_")
    G_h1 = tf.nn.relu(tf.matmul(z_, G_W1) + G_b1, "G_h1")
    G_h2_logits = tf.matmul(G_h1, G_W2) + G_b2
    return tf.sigmoid(G_h2_logits)


def D(x, y):
    '''
    Discriminator

    Two layer neural network that classifies the input vector as real or fake.
    Concatenates the noise vector and the context vector and then uses it as the network's input.

    :param tf.Tensor x:
        Input tensor representing MNIST-like data.
    :param tf.Tensor y:
        Context tensor. One hot representation of the label.
    :return:
        Tuple containing the discriminator's output before applying the output layer's activation function (logits) and
        after applying the activation function.
        The logits are needed for the loss function.
        The actual output scalar is a value between 0 and 1 classifying the input sequence as real (1) or fake (0).
    '''
    x_ = tf.concat([x, y], 1, "x_")
    D_h1 = tf.nn.relu(tf.matmul(x_, D_W1) + D_b1, "D_h1")
    D_h2_logits = tf.matmul(D_h1, D_W2) + D_b2

    return (D_h2_logits, tf.sigmoid(D_h2_logits))


def noise(n):
    '''
    Generates **n** random noise samples of shape noise_size (global variable).
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
    context = list(map(one_hot_vector, [random.randint(0, context_size - 1) for _ in range(n)]))
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
    parser = argparse.ArgumentParser(description="Run the CGAN model on the MNIST data set.")
    parser.add_argument("--batchsize", metavar="INTEGER", default=128, type=int,
                        help="size of mini-batch (default: 128)")
    parser.add_argument("--epochs", metavar="INTEGER", default=50000, type=int,
                        help="number of training iterations (default: 50000)")
    parser.add_argument("--noisesize", metavar="INTEGER", default=100, type=int,
                        help="size of the random noise vector that is used as the generator's input (default: 100)")
    parser.add_argument("--alpha", metavar="FLOAT", default=0.001, type=float,
                        help="learning rate (default: 0.001)")
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
    # learning rate for both generator and discriminator
    alpha = args.alpha
    # k is the number of learning steps of the discriminator, while the discriminator takes k steps the generator only
    # takes one
    k = args.k

    # Xavier variable initializer and random seed
    seed = 611357224
    xavier = tf.glorot_normal_initializer(seed=seed)

    # generator's input (noise samples), shape of [None, noise_size] means that there can be an arbitrary number of inputs of
    # shape [noise_size], which allows batches for input
    z = tf.placeholder(tf.float32, [None, noise_size], "z")
    # generator's weights, tensors first dimension is increased by context_size to include the context variable
    G_W1 = tf.get_variable("G_W1", [noise_size + context_size, 392], initializer=xavier)
    G_W2 = tf.get_variable("G_W2", [392, 784], initializer=xavier)
    # generator's biases
    G_b1 = tf.Variable(tf.zeros([392]), name="G_b1")
    G_b2 = tf.Variable(tf.zeros([784]), name="G_b2")

    # discriminator's input (real data), shape of [None, 784] means that there can be an arbitrary number of inputs of shape
    # [784], which allows batches for input
    x = tf.placeholder(tf.float32, [None, 784], "x")
    # discriminator's weights, tensors first dimension is increased by context_size to include the context variable
    D_W1 = tf.get_variable("D_W1", [784 + context_size, 392], initializer=xavier)
    D_W2 = tf.get_variable("D_W2", [392, 1], initializer=xavier)
    # discriminator's biases
    D_b1 = tf.Variable(tf.zeros([392]), name="D_b1")
    D_b2 = tf.Variable(tf.zeros([1]), name="D_b2")

    # context variable (label) for both generator and discriminator, shape of [None, context_size] means that there can be
    # an arbitrary number of contexts of shape [context_size], which allows batches for context
    y = tf.placeholder(tf.float32, [None, context_size], "y")

    # generator's output
    G_ = G(z, y)
    # discrimintors rating of real data
    (D_real_logits, D_real) = D(x, y)
    # discriminators rating of "fake" (generated) data
    (D_fake_logits, D_fake) = D(G_, y)

    # generator's loss function (cross-entropy with logits), discriminator should rate fake data as 0, if fake data is
    # rated ~1, the generated data looks likes real data to the discriminator, so the generators goal is to maximize the
    # ~1-rated data
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    # generator's optimizer function minimizes the difference between discriminators rating of the generated data and 1
    G_optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(G_loss, var_list=[G_W1, G_W2, G_b1, G_b2])

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
    D_optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(D_loss, var_list=[D_W1, D_W2, D_b1, D_b2])

    print("Done!")
    print("Initialize variables... ", end='')

    # start session and initialize variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    print("Done!")
    print("Loading data set... ", end='')

    # get MNIST data set with labels as one hot vector
    mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)

    print("Done!")

    # create output directory if it doesn't exist yet
    if not os.path.exists('cgan_out/'):
        os.makedirs('cgan_out/')

    # plot counter
    n = 0
    # file for saving loss values
    file = open("cgan_out/loss.csv", "w")
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

        # plot samples and print loss values every 1000 epoch
        if e % 1000 == 0:
            labels = random_context(16)
            # plot samples
            samples = sess.run(G_, {y: labels, z: noise(16)})
            fig = plot(samples, labels)
            # save plot in out/ directory
            plt.savefig('cgan_out/{}.png'.format(str(n).zfill(3)))
            n += 1
            plt.close(fig)

            # print generator and discriminator loss
            print("Epoch: %d" % e)
            print("G_LOSS: ")
            print(sess.run(G_loss, {x: x_, y: y_, z: z_}))
            print("D_LOSS: ")
            print(sess.run(D_loss, {x: x_, y: y_, z: z_}))
            print()

            # write current loss values to csv file
            file.write("%d, %f, %f\n" % (e, sess.run(G_loss, {x: x_, y: y_, z: z_}), sess.run(D_loss, {x: x_, y: y_, z: z_})))

    print("Done!")
    file.close()
    exit(0)