"""Utils used in s2s"""

from __future__ import absolute_import, division, print_function

import codecs

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util import nest

from utils.data_util import read_vocab

def load_pretrained_embedding(vocab_file, embed_file, embedding_name, trainable=True):
    print("load pretrained embedding file {0}".format(embed_file))
    emb_dict, emb_size = load_embed_txt(embed_file)
    w2i,i2w = read_vocab(vocab_file)
    for word in w2i:
        if word not in emb_dict:
            emb_dict[word] = [0.0] * emb_size
    
    emb_mat = np.array([emb_dict[word] for word in w2i], dtype=np.float32)
    emb_mat_var = tf.get_variable(embedding_name, shape=emb_mat.shape, initializer=tf.constant_initializer(emb_mat), trainable=trainable)
    return emb_mat_var

def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(
                    vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size

# this is copied from tf 1.4
class ResidualWrapper(tf.contrib.rnn.RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, residual_fn=None):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._residual_fn = residual_fn

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run the cell and then apply the residual_fn on its inputs to its outputs.
    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.
    Returns:
      Tuple of cell outputs and new state.
    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())
    def default_residual_fn(inputs, outputs):
      nest.assert_same_structure(inputs, outputs)
      nest.map_structure(assert_shape_match, inputs, outputs)
      return nest.map_structure(lambda inp, out: inp + out, inputs, outputs)
    res_outputs = (self._residual_fn or default_residual_fn)(inputs, outputs)
    return (res_outputs, new_state)

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
    elif opt == 'adagrad':
        optfn = tf.train.AdagradOptimizer
    else:
        assert False
    return optfn


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       src_vocab_file = None,
                                       tgt_vocab_file = None,
                                       src_pretrained_embedding = None,
                                       tgt_pretrained_embedding = None,
                                       dtype=tf.float32,
                                       scope=None):
    """Create embedding matrix for both encoder and decoder.
    Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
        encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
        embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
        embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    scope: VariableScope for the created subgraph. Default to "embedding".
    Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.
    Raises:
    ValueError: if use share_vocab but source and target have different vocab
        size.
    """
    with tf.variable_scope(scope or "embeddings", dtype=dtype) as scope:
        # Share embedding
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab sizes"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            if src_pretrained_embedding:
                embedding = load_pretrained_embedding(src_vocab_file, src_pretrained_embedding,"embedding_share")
            else:
                embedding = tf.get_variable(
                    "embedding_share", [src_vocab_size, src_embed_size], dtype, initializer=tf.random_uniform_initializer(-1, 1))
            embedding_encoder = embedding
            embedding_decoder = embedding
        else:
            with tf.variable_scope("encoder"):
                if src_pretrained_embedding:
                    embedding_encoder = load_pretrained_embedding(src_vocab_file, src_pretrained_embedding, "embedding_encoder")
                else:
                    embedding_encoder = tf.get_variable(
                        "embedding_encoder", [src_vocab_size, src_embed_size], dtype, initializer=tf.random_uniform_initializer(-1, 1))

            with tf.variable_scope("decoder"):
                if tgt_pretrained_embedding:
                    embedding_decoder = load_pretrained_embedding(tgt_vocab_file, tgt_pretrained_embedding, 'embedding_decoder')
                else:
                    embedding_decoder = tf.get_variable(
                        "embedding_decoder", [tgt_vocab_size, tgt_embed_size], dtype, initializer=tf.random_uniform_initializer(-1, 1))

    return embedding_encoder, embedding_decoder


def single_rnn_cell(cell_name, num_units, train_phase=True, keep_prob=0.75, device_str=None, residual_connection=False, residual_fn=None):
    """
    Get a single rnn cell
    """
    cell_name = cell_name.upper()
    if cell_name == "GRU":
        cell = tf.contrib.rnn.GRUCell(num_units)
    elif cell_name == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(num_units,state_is_tuple=False)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)

    # dropout wrapper
    if train_phase:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=keep_prob,
            output_keep_prob=keep_prob)

    # Residual
    if residual_connection:
        cell = ResidualWrapper(cell, residual_fn = residual_fn)

    # device wrapper
    if device_str:
        print("RNN cell on device:", device_str)
        cell = tf.contrib.rnn.DeviceWrapper(cell, device_str)
    return cell


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def get_cell_list(cell_name, num_units, num_layers, num_residual_layers=0,
                  train_phase=True, num_gpus=1, base_gpu=0, keep_prob=0.8, residual_fn=None):
    """Create a list of RNN cells."""
    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        single_cell = single_rnn_cell(
            cell_name=cell_name,
            num_units=num_units,
            keep_prob=keep_prob,
            train_phase=train_phase,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
            residual_fn=residual_fn
        )
        cell_list.append(single_cell)

    return cell_list

def multi_rnn_cell(cell_name, dim_size, num_layers=1, train_phase=True, keep_prob=0.80, num_residual_layers=0, num_gpus=1, base_gpu=0, residual_fn=None):
    """
    Get multi layer rnn cell
    """
    cells = get_cell_list(cell_name, dim_size, num_layers,
                          num_residual_layers, train_phase, 
                          num_gpus, base_gpu, keep_prob, residual_fn)
    if len(cells) > 1:
        final_cell = tf.contrib.rnn.MultiRNNCell(cells=cells)
    else:
        final_cell = cells[0]
    return final_cell

    
def bidirection_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, num_gpus, sequence_length, inputs):
    print("forward cell")
    fw_cell = multi_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, 0, num_gpus)
    print("backward cell")
    bw_cell = multi_rnn_cell(cell_name, num_units, num_bi_layers, train_phase, keep_prob, 0, num_gpus, num_bi_layers)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        dtype=tf.float32,
        sequence_length=sequence_length,
        inputs=inputs,
        swap_memory = True
    )
    outputs_concat = tf.concat(outputs, -1)
    return outputs_concat, states


def get_config_proto(log_device_placement=True, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0,per_process_gpu_memory_fraction=0.95):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto
