import tensorflow as tf
import numpy as np

def fit(X, y):

    neg_one = tf.constant(-1.0, dtype=tf.float64)
    distances = tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
