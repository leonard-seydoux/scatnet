#!/anaconda3/envs/scatnet/bin/python
# -*- coding: utf-8 -*-
"""Learnable scattering network with GMM loss minimization.

The wavelets and scales of the scattering network are learned with respect to
the clustering loss from GMM.
Authors: Randall Balestriero, Leonard Seydoux.
Email: leonard.seydoux@gmail.com
"""
import numpy as np
import scatnet as sn
import scipy as sp
import sys
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Initilization
# -------------

# Parse arguements.
args = sn.io.parse_arguments(sys.argv[1])

# Create a summary directory, includes all parameters and intialiaze all
# output files. Extract the tag of the run.
summary = sn.io.Summary(args['summary'])
summary.save_args()

# Load time series.
# The data is formatted as (n_segments, channels, patch_shape) and
# the times are given with the step size of the scattering coefficients.
stream = sn.data.read_data(**args['data']['load'])
stream.filter(**args['data']['filter'])
stream.taper(**args['data']['taper'])
data, times = stream.batch(layers_kw=args['layers'], **args['data']['batch'])

# Initialize graph.
# First get the explicit dimensions of the input data, and depth of graph.
batch_size = args['data']['batch']['batch_size']
patch_shape = data.shape[-1]
channels = data.shape[1]
depth = len(args['layers']['j'])

# Some configuration for the graph compilation.
g = tf.Graph()
config = tf.ConfigProto()

with tf.Session(graph=g, config=config) as sess:

    # Graph definition
    # ----------------

    with g.as_default():

        # Input data.
        x_shape = (batch_size, channels, patch_shape)
        x = tf.placeholder(tf.float32, shape=x_shape)

        # Scattering network.
        layers = [sn.Scattering(x, index=0, **args['layers'])]
        for i in range(1, depth):
            layer = sn.Scattering(layers[-1].u, index=i, **args['layers'])
            layers.append(layer)

        # Extract parameters.
        net = [layer.parameters for layer in layers]

        # Get reconstruction losses.
        rl = tf.add_n([a.reconstruction_loss for a in layers])

        # Renormalize coefficients.
        r = list()
        for i in range(1, depth):
            r.append(layers[i].renorm(layers[i - 1], args['eps_norm']))

        # Concatenate.
        sx = tf.transpose(tf.concat(r, axis=1), [1, 0, 2])
        sx = tf.reshape(sx, [sx.get_shape().as_list()[0], -1])
        sx = tf.transpose(sx)
        sx = tf.log(sx + args['eps_log'])

        # Save times-frequency properties of the graph
        summary.save_graph(layers, stream[0].stats.sampling_rate)
        summary.save_times(times)

        # Principal components analysis
        n_times, n_features = sx.get_shape().as_list()
        n_pca = args['pca']['n_components']
        sx_w = tf.placeholder(tf.float32, (n_features, n_pca))
        sx_bar = tf.placeholder(tf.float32, shape=(1, n_features))
        sx_proj = tf.matmul(sx - sx_bar, sx_w)

        # Gaussian mixture centroids, covariances and probability
        n_clusters = args['gmm_init']['n_components']
        mu = tf.placeholder(tf.float32, shape=(n_clusters, n_pca))
        cov = tf.placeholder(tf.float32, shape=(n_clusters, n_pca, n_pca))
        tau = tf.placeholder(tf.float32, shape=(n_clusters,))

        # Calculate Gaussian mixture loss
        loss, cat = sn.models.gmm(mu, cov, tau, sx_proj, **args['gmm'])
        y = tf.argmax(cat, axis=1)

        # loss
        learn_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learn_rate)
        minimizer = optimizer.minimize(loss)

    # Training
    # --------

    # Create graph
    sess.run(tf.global_variables_initializer())
    with tf.device('/GPU:0'):

        # Initialization
        cost = 0
        cost_r = 0
        pca_op = PCA(n_components=args['pca']['n_components'])
        gmm_op = GaussianMixture(**args['gmm_init'])

        # Run over batches
        epochs = args['learning']['epochs']
        learning_rate = args['learning']['rate']
        for epoch in range(epochs):

            # Waitbar
            summary.watch(epoch, epochs)

            # Gradually decrease learning rate over epochs
            if epoch == epochs // 2:
                learning_rate /= 5
            if epoch == 3 * epochs // 4:
                learning_rate /= 5

            # Calculate scattering coefficients for all batches
            scat_all = list()
            n_batches = data.shape[0] // batch_size
            for b in range(n_batches):
                x_batch = data[b * batch_size: (b + 1) * batch_size]
                s = sess.run(sx, feed_dict={x: x_batch})
                s[np.isnan(s)] = np.log(args['eps_log'])
                s[np.isinf(s)] = np.log(args['eps_log'])
                scat_all.append(s)

            # Recalculate principal axes
            # First, (B, T, F) -> (F, B, T) -> (F, B + T) -> (B + T, F)
            # in order to feed the pca with (n_samples, n_features)
            scat_all = np.transpose(np.array(scat_all), [2, 0, 1])
            scat_all = scat_all.reshape(scat_all.shape[0], -1).T
            scat_all_proj = pca_op.fit_transform(scat_all)
            scat_w = pca_op.components_.T

            # Save scattering coefficients
            summary.save_full(scat_all)

            # Extract clusters
            gmm_op.fit(scat_all_proj)
            means = gmm_op.means_.astype(np.float32)
            weights = gmm_op.weights_.astype(np.float32)
            covariances = gmm_op.covariances_.astype(np.float32)

            # Save cluster-related info
            summary.save_hot(scat_all_proj, gmm_op, pca_op)
            summary.save_scalar('loss_reconstruction', cost_r)
            summary.save_scalar('loss_clustering', cost)
            summary.save_wavelets(sess.run(net), args['layers']['hilbert'])

            # Loop over batches for clustering only (use permutations)
            cost = [0]
            cost_r = [0]
            p = sp.random.permutation(data.shape[0])
            trange = summary.trange(n_batches)
            for b in trange:
                # Feed dictionnary
                feed = {
                    x: data[p[b * batch_size:(b + 1) * batch_size]],
                    learn_rate: learning_rate, sx_w: scat_w,
                    sx_bar: np.reshape(pca_op.mean_, (1, -1)),
                    tau: weights, cov: covariances, mu: means
                }

                # Minimize loss
                sess.run(minimizer, feed_dict=feed)
                c, r = sess.run([loss, rl], feed_dict=feed)
                cost.append(c)
                cost_r.append(r)
                trange.set_postfix(loss=cost[-1])

            # Average loss over batches
            cost = np.mean(cost)
            cost_r = np.mean(cost_r)
