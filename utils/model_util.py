"""Utils used in s2s"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# copied from tensorflow/nmt project
def create_attention_mechanism(attention_option, num_units, memory,
                               memory_sequence_length):
  """Create attention mechanism based on the attention_option."""
  # Mechanism
  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=memory_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=memory_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=memory_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=memory_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism



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
