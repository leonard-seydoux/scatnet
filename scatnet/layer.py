#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is the scatnet_learnable.py from Randall modified."""

import logging
import numpy as np
import tensorflow as tf

HERMITE = [[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]]
FORMAT = 'float32'


def adam(loss, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam stochastic gradeint."""
    params = tf.trainable_variables()
    all_grads = tf.gradients(loss, params)
    t_prev = tf.Variable(np.float32(0), trainable=False, name='t_Adam')
    updates = list()

    # Using theano constant to prevent upcasting of float32
    one = np.float32(1)

    t = tf.assign_add(t_prev, 1)
    a_t = learning_rate * \
        tf.sqrt(one - tf.pow(beta2, t)) / (one - tf.pow(beta1, t))

    for param, g_t in zip(params, all_grads):
        m_prev = tf.Variable(np.zeros(param.get_shape().as_list(
        ), dtype='float32'), name='m_prev', trainable=False)
        v_prev = tf.Variable(np.zeros(param.get_shape().as_list(
        ), dtype='float32'), name='m_prev', trainable=False)

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * tf.square(g_t)
        step = a_t * m_t / (tf.sqrt(v_t) + epsilon)
        updates.append(tf.assign(m_prev, m_t))
        updates.append(tf.assign(v_prev, v_t))
        updates.append(tf.assign_sub(param, step))

    updates.append(t)
    return tf.group(updates)


def complex_hermite_interp(xi, x, m, p):
    """Complex interpolation with hermite polynomials.

    Arguments
    ---------
        x: array-like
            The knots onto the function is defined (derivative and
            antiderivative).
        t: array-like
            The points where the interpolation is required.
        m: array-like
            The complex values of amplitude onto knots.
        p: array-like
            The complex values of derivatives onto knots.

    Returns
    -------
        yi: array-like
            The interpolated complex-valued function.
    """
    # Hermite polynomial coefficients
    h = tf.Variable(np.array(HERMITE).astype(FORMAT), trainable=False)

    # Concatenate coefficients onto shifted knots (1, n_knots - 1)
    # The knots are defined at each scales, so xx is (n_scales, n_knots - 1, 2)
    xx = tf.stack([x[:, :-1], x[:, 1:]], axis=2)

    # The concatenated coefficients are of shape (2, n_knots - 1, 2)
    mm = tf.stack([m[:, :-1], m[:, 1:]], axis=2)
    pp = tf.stack([p[:, :-1], p[:, 1:]], axis=2)

    # Define the full function y to interpolate (2, n_knots - 1, 4)
    # on the shifted knots
    y = tf.concat([mm, pp], axis=2)

    # Extract Hermite polynomial coefficients from y (n_knots - 1, 4)
    yh = tf.einsum('iab,bc->iac', y, h)

    # Extract normalized piecewise interpolation vector
    # (n_scales, n_knots - 1, n_interp)
    xi_ = tf.expand_dims(tf.expand_dims(xi, 0), 0)
    x0_ = tf.expand_dims(xx[:, :, 0], 2)
    x1_ = tf.expand_dims(xx[:, :, 1], 2)
    xn = (xi_ - x0_) / (x1_ - x0_)

    # Calculate powers of normalized interpolation vector
    mask = tf.logical_and(tf.greater_equal(xn, 0.), tf.less(xn, 1.))
    mask = tf.cast(mask, tf.float32)
    xp = tf.pow(tf.expand_dims(xn, -1), [0, 1, 2, 3])

    # Interpolate
    return tf.einsum('irf,srtf->ist', yh, xp * tf.expand_dims(mask, -1))


def real_hermite_interp(xi, x, m, p):
    """Real interpolation with hermite polynomials.

    Arguments
    ---------
        x: array-like
            The knots onto the function is defined (derivative and
            antiderivative).
        t: array-like
            The points where the interpolation is required.
        m: array-like
            The real values of amplitude onto knots.
        p: array-like
            The real values of derivatives onto knots.

    Returns
    -------
        yi: array-like
            The interpolated real-valued function.
    """
    # Hermite polynomial coefficients
    h = tf.Variable(np.array(HERMITE).astype(FORMAT), trainable=False)

    # Concatenate coefficients onto shifted knots (1, n_knots - 1)
    # The knots are defined at each scales, so xx is (n_scales, n_knots - 1, 2)
    xx = tf.stack([x[:, :-1], x[:, 1:]], axis=2)

    # The concatenated coefficients are of shape (n_knots - 1, 2)
    mm = tf.stack([m[:-1], m[1:]], axis=1)
    pp = tf.stack([p[:-1], p[1:]], axis=1)

    # Define the full function y to interpolate (n_knots - 1, 4)
    # on the shifted knots
    y = tf.concat([mm, pp], axis=1)

    # Extract Hermite polynomial coefficients from y (n_knots - 1, 4)
    yh = tf.matmul(y, h)

    # Extract normalized piecewise interpolation vector
    # (n_scales, n_knots - 1, n_interp)
    xi_ = tf.expand_dims(tf.expand_dims(xi, 0), 0)
    x0_ = tf.expand_dims(xx[:, :, 0], 2)
    x1_ = tf.expand_dims(xx[:, :, 1], 2)
    xn = (xi_ - x0_) / (x1_ - x0_)

    # Calculate powers of normalized interpolation vector
    mask = tf.logical_and(tf.greater_equal(xn, 0.), tf.less(xn, 1.))
    mask = tf.cast(mask, tf.float32)
    xp = tf.pow(tf.expand_dims(xn, -1), [0, 1, 2, 3])

    # Interpolate
    return tf.einsum('rf,srtf->st', yh, xp * tf.expand_dims(mask, -1))


class Scattering:
    """Learnable scattering network layer."""

    def __init__(self, x, j=None, q=None, k=None, pooling_type='average',
                 decimation=2, pooling=2, index=0, **filters_kw):
        """Scattering network layer.

        Computes the convolution modulus and scattering coefficients of the
        input signal.

        Arguments
        ---------
            x: :class:`~tensorflow.Tensor()`
                Input data of shape ``(batch_size, channels, patch_shape).
        """
        # Filter bank properties
        self.shape_input = x.get_shape().as_list()
        self.j = j = j[index] if type(j) is list else j
        self.q = q = q[index] if type(q) is list else q
        self.k = k = k[index] if type(k) is list else k

        # Initialize filter bank (n_features, n_samples), then concatenate the
        # real and imaginary parts to do a single convolution with stacked
        # filters (2 * n_features, n_samples) and reshape filter bank with the
        # same logical dimensions of input data (n_samples, 1, 2 * n_features)
        filters = self.init_filters(j, q, k, **filters_kw)
        n_filters, kernel_size = filters.get_shape().as_list()
        filters_concat = tf.concat([tf.real(filters), tf.imag(filters)], 0)
        filters_kernel = tf.expand_dims(tf.transpose(filters_concat), 1)

        # Pad input in the time dimension before convolution with half the size
        # of filters temporal dimension (kernel_size).
        shape_fast = [np.prod(self.shape_input[:-1]), 1, self.shape_input[-1]]
        paddings = [0, 0], [0, 0], [kernel_size // 2 - 1, kernel_size // 2 + 1]
        x_reshape = tf.reshape(x, shape_fast)
        x_pad = tf.pad(x_reshape, paddings=paddings, mode='SYMMETRIC')
        logging.debug('input {}'.format(self.shape_input))
        logging.debug('reshape {}'.format(shape_fast))
        logging.debug('padding {}'.format(paddings))

        # Differentiate the case of one input channel or multiple
        # which needs reshaping in order to treat them independently
        x_conv = tf.nn.conv1d(x_pad, filters_kernel, stride=decimation,
                              padding='VALID', data_format='NCW')
        u = tf.sqrt(tf.square(x_conv[:, :n_filters]) +
                    tf.square(x_conv[:, n_filters:]))
        self.u = tf.reshape(u, (*self.shape_input[:-1], n_filters, -1))

        # Scattering pooling setup
        if pooling_type == 'average':
            pool = tf.layers.average_pooling1d
        else:
            pool = tf.layers.max_pooling1d

        # Pooling for the scattering coefficients
        if pooling > 1:
            pooled = pool(
                u, pooling // (decimation ** (index + 1)),
                pooling // (decimation ** (index + 1)),
                'VALID', data_format='channels_first')
        else:
            pooled = u

        self.s = tf.reshape(pooled, self.shape_input[:-1] + [j * q] + [-1])

        self.output = self.s
        inverse = tf.gradients(x_conv, x, x_conv)[0]
        self.reconstruction_loss = tf.nn.l2_loss(
            inverse - tf.stop_gradient(x)) / np.prod(self.shape_input)

    def init_filters(self, j, q, k, learn_scales=False, learn_knots=False,
                     learn_filters=True, hilbert=False):
        """Create the filter bank."""
        # If the scales are learnable, allows to go toward lower frequencies.
        extra_octave = 1 if learn_scales else 0
        self.filter_samples = k * 2 ** (j + extra_octave)

        # Define the time grid onto integers such as
        # [-k * 2 ** (j - 1), ..., -1, 0, 1, ..., k * 2 ** (j - 1)]
        # We change the range depending on if the extra octave was added
        time_max = np.float32(k * 2**(j - 1 + extra_octave))
        time_grid = tf.lin_space(-time_max, time_max, self.filter_samples)

        # Scales
        # ------
        # The first scale is at the Nyquist frequency, the increasing scales
        # go to lower frequencies.
        # Note: the following method might not be computationally optimal but
        # is the most stable for learning and precision.
        scales_base = 2**(tf.range(j * q, dtype=tf.float32) / np.float32(q))
        scales_delta = tf.Variable(
            tf.zeros(j * q), trainable=learn_scales, name='scales')
        scales = scales_base + scales_delta

        # Now ensure that the scales are strictly increasing in case of
        # delta scales are too high amplitude. If not, shift by the right
        # amount get the correcting shifts being the cases with negative
        # increase of the scales and the nyquist offset which ensures that the
        # smallest filter has scale of at least 1.
        nyquist_offset = scales + \
            tf.stop_gradient(tf.one_hot(0, j * q) * tf.nn.relu(1 - scales[0]))
        scales_correction = tf.concat(
            [tf.zeros(1),
             tf.nn.relu(nyquist_offset[:-1] - nyquist_offset[1:])], 0)
        self.scales = nyquist_offset + \
            tf.stop_gradient(tf.cumsum(scales_correction))

        # The knots are defined at each scale. We start from the Nyquist
        # frequency where the knots must be [-k//2, ..., -1, 0, 1, ..., k//2]
        knots_base = tf.Variable(
            tf.ones(k), trainable=learn_knots, name='knots')

        # We compute the scaled differences first in order to clip to ensure
        # that you can not have (at any scale) as difference of less than 1,
        # corresponding to a Nyquist node. We then cumsum along the time axis
        # to get the scaled positions which are not yet starting at the
        # correct index they start for now at 0. This step is crucial to
        # ensure that during learning, there is no degenracy and aliasing
        # if trying to put knots closer together than 1. This is also crucial
        # for the high frequency filters as the nyquist one will always violate
        # the sampling if the knots are contracted.
        knots_sum = tf.cumsum(
            tf.clip_by_value(
                tf.expand_dims(knots_base, 0) * tf.expand_dims(self.scales, 1),
                1, self.filter_samples - k), exclusive=True, axis=1)
        self.knots = knots_sum - (k // 2) * tf.expand_dims(self.scales, 1)

        # Interpolation init, add the boundary condition mask and remove the
        # mean filters of even indices are the real parts and odd indices are
        # imaginary part
        if hilbert is True:

            # Create the (real) parameters
            m = (np.cos(np.arange(k) * np.pi) * np.hamming(k)).astype(FORMAT)
            p = (np.zeros(k)).astype(FORMAT)
            self.m = tf.Variable(m, name='m', trainable=learn_filters)
            self.p = tf.Variable(p, name='p', trainable=learn_filters)

            # Boundary Conditions and centering
            mask = np.ones(k, dtype=np.float32)
            mask[0], mask[-1] = 0, 0
            m_null = self.m - tf.reduce_mean(self.m[1:-1])
            filters = real_hermite_interp(
                time_grid, self.knots, m_null * mask, self.p * mask)

            # Renorm and set filter-bank
            filters_renorm = filters / tf.reduce_max(filters, 1, keepdims=True)
            filters_fft = tf.spectral.rfft(filters_renorm)
            filters = tf.ifft(
                tf.concat([filters_fft, tf.zeros_like(filters_fft)], 1))

        else:

            # Create the (complex) parameters
            m = np.stack([np.cos(np.arange(k) * np.pi) * np.hamming(k),
                          np.zeros(k) * np.hamming(k)]).astype(FORMAT)
            p = np.stack([np.zeros(k),
                          np.cos(np.arange(k) * np.pi) * np.hamming(k)]
                         ).astype(FORMAT)
            self.m = tf.Variable(m, name='m', trainable=learn_filters)
            self.p = tf.Variable(p, name='p', trainable=learn_filters)

            # Boundary Conditions and centering
            mask = np.ones((1, k), dtype=np.float32)
            mask[0, 0], mask[0, -1] = 0, 0
            m_null = self.m - \
                tf.reduce_mean(self.m[:, 1:-1], axis=1, keepdims=True)
            filters = complex_hermite_interp(
                time_grid, self.knots, m_null * mask, self.p * mask)

            # Renorm and set filter-bank
            filters_renorm = filters / tf.reduce_max(filters, 2, keepdims=True)
            filters = tf.complex(filters_renorm[0], filters_renorm[1])

        # Define the parameters for saving
        self.parameters = self.m, self.p, self.scales, self.knots
        return filters

    def renorm(self, parent, epsilon=1e-3):
        """Parent renormalization of the scattering coefficients.

        Renormalize scattering coefficients of layer m + 1 with the one
        obtained at layer m. Eeach scale of the Sx at layer m + 1 are divided
        by the amplitude of the parent scale at Sx at layer m.

        Arguments
        ---------
            parent: :class:Scattering:
                The parent scattering layer.
            epsilon: float
                Regularizer for avoiding division by zero.

        Returns
        -------
            s_renorm: :class:np.array
                The renormalized scattering coefficients.
        """
        # Extract all shapes.
        if epsilon > 0:
            s = self.s / (tf.expand_dims(parent.s, -2) + epsilon)
            batch_size, *_, samples = s.get_shape().as_list()
            return tf.reshape(s, [batch_size, -1, samples])
        else:
            return tf.reshape(self.s, [batch_size, -1, samples])
        # batch_size, j_parent, n_parent = parent.s.get_shape().as_list()
        # _, j_child, n_child = self.s.get_shape().as_list()

        # # Check what is the downsampling factor between parent and self.
        # repeat_j = j_child // j_parent

        # # Renormalize accordingly.
        # if repeat_j > 1:
        #     downsample = n_parent // n_child
        #     s_expand = tf.expand_dims(parent.s, axis=2)
        #     s_tile = tf.tile(s_expand, [1, 1, repeat_j, 1])
        #     s_repeat = tf.reshape(s_tile, [batch_size, j_child, n_parent])
        #     return self.s / (s_repeat[:, :, ::downsample] + epsilon)
        # else:
        #     return self.s / (parent.s + epsilon)
