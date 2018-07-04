import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def G(z):
    '''
    Generator

    Two layer neural network that generates MNIST-like data.

    :param tf.Tensor z:
        Input tensor. Random noise vector.
    :return:
        Generator's output vector representing MNIST-like data.
    '''
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1, "G_h1")
    G_h2_logits = tf.matmul(G_h1, G_W2) + G_b2
    return tf.sigmoid(G_h2_logits)


def D(x):
    '''
    Discriminator

    Two layer neural network that classifies the input vector as real or fake.

    :param tf.Tensor x:
        Input tensor representing MNIST-like data.
    :return:
        Tuple containing the discriminator's output before applying the output layer's activation function (logits) and
        after applying the activation function.
        The logits are needed for the loss function.
        The actual output scalar is a value between 0 and 1 classifying the input sequence as real (1) or fake (0).
    '''
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1, "D_h1")
    D_h2_logits = tf.matmul(D_h1, D_W2) + D_b2
    return (D_h2_logits, tf.sigmoid(D_h2_logits))


def noise(n):
    '''
    Generates n random noise samples of shape noise_size (global variable).
    The random values are drawn uniformly from the interval [-1, 1].

    :param int n:
        Size of each noise sample.
    :return:
        Array containing n random noise samples.
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
    # create 4x4 grid in which the subplots are arranged with some horizontal and vertical space
    grid = grd.GridSpec(4, 4, wspace=0.05, hspace=0.5)

    # plot all samples
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GAN model on the MNIST data set.")
    parser.add_argument("--batchsize", metavar="INTEGER", default=64, type=int,
                        help="size of mini-batch (default: 64)")
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
    # generator's weights
    G_W1 = tf.get_variable("G_W1", [noise_size, 392], initializer=xavier)
    G_W2 = tf.get_variable("G_W2", [392, 784], initializer=xavier)
    # generator's biases
    G_b1 = tf.Variable(tf.zeros([392]), name="G_b1")
    G_b2 = tf.Variable(tf.zeros([784]), name="G_b2")

    # discriminator's input (real data), shape of [None, 784] means that there can be an arbitrary number of inputs of shape
    # [784], which allows batches for input
    x = tf.placeholder(tf.float32, [None, 784], "x")
    # discriminator's weights
    D_W1 = tf.get_variable("D_W1", [784, 392], initializer=xavier)
    D_W2 = tf.get_variable("D_W2", [392, 1], initializer=xavier)
    # discriminator's biases
    D_b1 = tf.Variable(tf.zeros([392]), name="D_b1")
    D_b2 = tf.Variable(tf.zeros([1]), name="D_b2")


    # generator's output
    G_ = G(z)
    # discrimintor's rating of real data
    (D_real_logits, D_real) = D(x)
    # discriminator's rating of "fake" (generated) data
    (D_fake_logits, D_fake) = D(G_)

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
    # discriminator's loss function is a combination of both the loss for fake data and the loss for real data
    D_loss = D_loss_fake + D_loss_real

    # discriminator's optimizer function minimizes the difference between discriminators rating of real and fake data and the
    # target rating
    D_optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(D_loss, var_list=[D_W1, D_W2, D_b1, D_b2])

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

    # create output directory if it doesn't exist yet
    if not os.path.exists('GAN_out/'):
        os.makedirs('GAN_out/')

    # plot counter
    n = 0
    # open/create file for saving loss values
    file = open("GAN_out/loss.csv", "w")
    # write csv header
    file.write("epoch, g_loss, d_loss\n")

    print("Training network... ")
    # train network
    for e in range(epochs):
        # train discriminator for k steps
        for _ in range(k):
            (x_, _) = mnist.train.next_batch(batch_size)
            sess.run(D_optimizer, {x: x_, z: noise(batch_size)})
        # train generator for one step
        z_ = noise(batch_size)
        sess.run(G_optimizer, {z: z_})

        # plot samples and print loss values every 1000 epoch
        if e % 1000 == 0:
            # plot samples
            samples = sess.run(G_, {z: noise(16)})
            fig = plot(samples)
            # save plot in out/ directory
            plt.savefig('GAN_out/{}.png'.format(str(n).zfill(3)))
            n += 1
            plt.close(fig)

            # print generator's and discriminator's loss
            print("Epoch: %d" % e)
            print("G_LOSS: ")
            print(sess.run(G_loss, {x: x_, z: z_}))
            print("D_LOSS: ")
            print(sess.run(D_loss, {x: x_, z: z_}))

            # write current loss values to csv file
            file.write("%d, %f, %f\n" % (e, sess.run(G_loss, {x: x_, z: z_}), sess.run(D_loss, {x: x_, z: z_})))

    print("Done!")
    file.close()
    exit(0)