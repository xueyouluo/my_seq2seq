"""Utils used in s2s"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_optimizer(opt):
    """
    A function to get optimizer.

    :param opt: optimizer function name
    :returns: the optimizer function
    :raises assert error: raises an assert error
    """
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert False
    return optfn


def single_rnn_cell(cell_name, dim_size, train_phase=True, keep_prob=0.75):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(dim_size)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(dim_size)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(dim_size)
    if train_phase and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob)
    return cell


def multi_rnn_cell(cell_name, dim_size, num_layers=1, train_phase=True, keep_prob=0.75):
    """
    Get multi layer rnn cell
    """
    cells = []
    for _ in range(num_layers):
        cell = single_rnn_cell(cell_name, dim_size, train_phase, keep_prob)
        cells.append(cell)
    if len(cells) > 1:
        final_cell = tf.contrib.rnn.MultiRNNCell(cells=cells)
    else:
        final_cell = cells[0]
    return final_cell
