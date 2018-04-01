from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from model.config import BasicConfig
import tensorflow as tf
from utils.model_util import bidirection_rnn_cell, get_cell_list, create_attention_mechanism, multi_rnn_cell
from tensorflow.python.util import nest
from tensorflow.python.layers.core import Dense


def gnmt_residual_fn(inputs, outputs):
    """Residual function that handles different inputs and outputs inner dims.
    Args:
    inputs: cell inputs, this is actual inputs concatenated with the attention
        vector. refer to GNMT RNN cell
    outputs: cell outputs
    Returns:
    outputs + actual inputs
    """
    def split_input(inp, out):
        out_dim = out.get_shape().as_list()[-1]
        inp_dim = inp.get_shape().as_list()[-1]
        return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
    actual_inputs, _ = nest.map_structure(split_input, inputs, outputs)

    def assert_shape_match(inp, out):
        inp.get_shape().assert_is_compatible_with(out.get_shape())
    nest.assert_same_structure(actual_inputs, outputs)
    nest.map_structure(assert_shape_match, actual_inputs, outputs)
    return nest.map_structure(lambda inp, out: inp + out, actual_inputs, outputs)


class GNMTModel(S2SModelWithPipeline):
    def __init__(self, sess, data_iterator, config=BasicConfig()):
        super(GNMTModel, self).__init__(sess, data_iterator, config)

    def setup_bidirection_encoder(self):
        print("set up GNMT encoder")
        num_layer = self.config.encode_layer_num
        num_residual_layer = num_layer - 2
        num_bi_layer = 1
        num_ui_layer = num_layer - num_bi_layer

        with tf.variable_scope("Encoder"):
            bi_encoder_outputs, bi_encoder_state = bidirection_rnn_cell(self.config.encode_cell_type, self.config.num_units, num_bi_layer, self.train_phase,
                                                                        self.config.keep_prob, self.config.num_gpus, self.source_length, self.encode_inputs)
            uni_cell = multi_rnn_cell(self.config.encode_cell_type, self.config.num_units,
                                      num_ui_layer,  self.train_phase, self.config.keep_prob, num_residual_layer, self.config.num_gpus, 1)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                uni_cell,
                bi_encoder_outputs,
                dtype=tf.float32,
                sequence_length=self.source_length)

            encoder_state = (bi_encoder_state[1],) + (
                (encoder_state,) if num_ui_layer == 1 else encoder_state)

            self.encode_output = encoder_outputs
            self.encode_state = encoder_state

    def setup_attention_decoder(self):
        print("set up GNMT attetion decoder")
        with tf.variable_scope("Decoder"):
            num_residual_layers = self.config.decode_layer_num - 2
            # multi-layer decoder
            decode_cell = get_cell_list(self.config.decode_cell_type, self.config.num_units,
                                        self.config.decode_layer_num, num_residual_layers,
                                        self.train_phase, self.config.num_gpus, 0, self.config.keep_prob,
                                        gnmt_residual_fn)

            memory = self.encode_output
            memory_length = self.source_length
            batch_size = self.batch_size

            if self.config.mode == "inference":
                memory = tf.contrib.seq2seq.tile_batch(
                    memory, self.config.beam_size)
                memory_length = tf.contrib.seq2seq.tile_batch(
                    memory_length, self.config.beam_size)
                self.encode_state =  tf.contrib.seq2seq.tile_batch(
                    self.encode_state, self.config.beam_size)
                batch_size = self.batch_size * self.config.beam_size

            atten_mech = create_attention_mechanism(
                self.config.attention_option, self.config.num_units, memory, memory_length)

            # Only wrap the bottom layer with the attention mechanism.
            attention_cell = decode_cell.pop(0)
            # Only generate alignment in greedy INFER mode.
            alignment_history = (self.config.mode=='inference' and self.config.beam_size == 0)

            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                attention_cell,
                atten_mech,
                attention_layer_size=None,  # don't use attention layer.
                output_attention=False,
                alignment_history = alignment_history,
                name="attention")

            cell = GNMTAttentionMultiCell(attention_cell, decode_cell)

            # setup initial state of decoder
            # pass encoder state to decoder
            self.initial_state = tuple(
                zs.clone(cell_state=es)
                if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
                for zs, es in zip(
                    cell.zero_state(batch_size, dtype=tf.float32), self.encode_state))

            self.decode_cell = cell

class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with GNMT attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=True):
        """Creates a GNMTAttentionMultiCell.
        Args:
        attention_cell: An instance of AttentionWrapper.
        cells: A list of RNNCell wrapped with AttentionInputWrapper.
        use_new_attention: Whether to use the attention generated from current
            step bottom layer's output. Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(GNMTAttentionMultiCell, self).__init__(
            cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):
                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)
