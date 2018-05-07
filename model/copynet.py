from model.basic_s2s_model import BasicS2SModel
from model.config import CopyNetConfig
import tensorflow as tf
from utils.model_util import create_attention_mechanism,  get_cell_list
from utils.data_util import UNK_ID
from utils.copynet_helper import CopyNetWrapper

class CopyNet(BasicS2SModel):
    def __init__(self,sess,config=CopyNetConfig()):
        super(CopyNet,self).__init__(sess,config)

    def setup_input_placeholders(self):
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.source_length = tf.placeholder(tf.int32, shape=[None, ])

        # input tokens using source oov words and vocab 
        self.source_extend_tokens = tf.placeholder(tf.int32, shape=[None, None], name='source_extend_tokens')

        # using dynamic batch size
        self.batch_size = tf.shape(self.source_tokens)[0]

        if self.train_phase:
            self.target_tokens = tf.placeholder(tf.int32, shape=[None, None], name='target_tokens')
            self.target_length = tf.placeholder(tf.int32, shape=[None,], name='target_length')

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.config.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.config.end_token

            # decoder_inputs: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            decoder_inputs = tf.concat([decoder_start_token,
                                             self.target_tokens], axis=1)
            
            condition = tf.less(decoder_inputs, self.config.tgt_vocab_size)
            self.decoder_inputs = tf.where(condition, decoder_inputs, tf.ones_like(decoder_inputs) * UNK_ID, name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = self.target_length + 1

            # decoder_targets: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets = tf.concat([self.target_tokens,
                                              decoder_end_token], axis=1, name='decoder_targets')

            self.predict_count = tf.reduce_sum(self.decoder_inputs_length)

    def setup_attention_decoder(self):
        print("set up attention decoder")
        with tf.variable_scope("Decoder"):
            # multi-layer decoder
            decode_cell = get_cell_list(self.config.decode_cell_type, self.config.num_units,
                                        self.config.decode_layer_num, 0, self.train_phase, self.config.num_gpus, 0, self.config.keep_prob)

            memory = self.encode_output
            memory_length = self.source_length

            if self.config.mode == "inference":
                memory = tf.contrib.seq2seq.tile_batch(
                    memory, self.config.beam_size)
                memory_length = tf.contrib.seq2seq.tile_batch(
                    memory_length, self.config.beam_size)

            atten_mech = create_attention_mechanism(
                self.config.attention_option, self.config.num_units, memory, memory_length)

            decode_cell = tf.contrib.rnn.MultiRNNCell(decode_cell)
            decode_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decode_cell,
                attention_mechanism=atten_mech,
                attention_layer_size=self.config.num_units
            )

            batch_size = self.batch_size
            # setup initial state of decoder
            if self.config.mode == "inference":
                batch_size = self.batch_size * self.config.beam_size
                self.decode_initial_state = tf.contrib.seq2seq.tile_batch(
                    self.decode_initial_state, self.config.beam_size)
            
            initial_state = [self.decode_initial_state for i in range(
                self.config.decode_layer_num)]
            initial_state = decode_cell.zero_state(batch_size, tf.float32).clone(
                cell_state=tuple(initial_state))

            # we only need to wrapper the decode cell
            self.decode_cell = CopyNetWrapper(decode_cell, self.source_extend_tokens, self.config.max_oovs, self.encode_output, self.output_layer, self.config.tgt_vocab_size)

            self.initial_state = self.decode_cell.zero_state(self.batch_size,
                tf.float32).clone(cell_state=initial_state)

            self.output_layer = None

    def setup_beam_search(self):
        print("setup beam search")
        # using greedying decoder right now
        # Because we use argmax to get last_ids in copynet wrapper, this may be not compatible with beam search decoder
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.decode_embedding,
            start_tokens=tf.tile([self.config.start_token], [self.batch_size]),
            end_token=self.config.end_token
        )

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decode_cell,
            helper=helper,
            initial_state=self.initial_state,
            output_layer=self.output_layer
        )

        dec_outputs, dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            output_time_major=False,
            maximum_iterations=self.config.max_inference_length,
            swap_memory=True)

        # batch * max_time_step * 1
        # make it compatible with beam search result
        beam_predictions = tf.expand_dims(dec_outputs.sample_id, -1)
        self.beam_predictions = tf.transpose(beam_predictions, perm=[0, 2, 1])


    def train_one_batch(self, source_tokens, source_length, source_extend_tokens, target_tokens, target_length, run_info=False):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length

        if run_info:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            losses, summary, global_step, _ = self.sess.run(
                [self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        else:
            losses, summary, global_step, _ = self.sess.run(
                [self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict)
        if run_info:
            self.summary_writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
            print("adding run meta for",global_step)
        self.summary_writer.add_summary(summary, global_step)
        return losses, global_step

    def eval_one_batch(self, source_tokens, source_length, source_extend_tokens, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        predicted_ids, losses, logits = self.sess.run(
            [self.valid_predictions, self.losses, self.logits], feed_dict=feed_dict)
        return predicted_ids, losses, logits

    def inference(self, source_tokens, source_length, source_extend_tokens,):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        predictions = self.sess.run(self.beam_predictions, feed_dict=feed_dict)
        return predictions