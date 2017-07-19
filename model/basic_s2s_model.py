"""A basic s2s model"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import os
from utils.model_util import get_optimizer, multi_rnn_cell, single_rnn_cell, create_attention_mechanism, create_emb_for_encoder_and_decoder, get_cell_list
from model.config import BasicConfig


class BasicS2SModel(object):
    def __init__(self, sess, config=BasicConfig()):
        assert config.mode in ["train", "eval", "inference"]
        self.train_phase = config.mode == "train"
        self.sess = sess
        self.config = config
        self.build()

    def build(self):
        self.global_step = tf.Variable(0, trainable=False)

        self.setup_input_placeholders()
        self.setup_embedding()
        if self.config.use_bidirection:
            self.setup_bidirection_encoder()
        else:
            self.setup_multilayer_encoder()
        self.setup_attention_decoder()
        if self.train_phase:
            self.setup_training_decode_layer()
            self.setup_train()
            self.setup_summary()
        else:
            self.setup_beam_search()
        self.setup_saver()

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def setup_train(self):
        if self.config.exponential_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.config.learning_rate, self.global_step, self.config.decay_steps, self.config.learning_rate_decay, staircase=True)
        else:
            self.learning_rate = tf.Variable(
                self.config.learning_rate, trainable=False)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * self.config.learning_rate_decay)

        opt = get_optimizer(self.config.optimizer)(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.losses, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.config.checkpoint_dir, self.sess.graph)
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.summary_op = tf.summary.merge_all()

    def setup_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def restore_model(self, epoch=None):
        if epoch is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                self.config.checkpoint_dir))
        else:
            self.saver.restore(
                self.sess, self.config.checkpoint_dir + "model.ckpt" + ("-%d" % epoch))
        print("restored model")

    def save_model(self, epoch=None):
        if epoch is None:
            self.saver.save(self.sess, self.config.checkpoint_dir +
                            "model.ckpt", global_step=self.global_step)
        else:
            self.saver.save(self.sess, self.config.checkpoint_dir +
                            "model.ckpt", global_step=epoch)

    def setup_input_placeholders(self):
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.source_length = tf.placeholder(tf.int32, shape=[None, ])
        # using dynamic batch size
        self.batch_size = tf.shape(self.source_tokens)[0]

        if self.train_phase:
            # the train data should pad with eos
            self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
            self.target_length = tf.placeholder(tf.int32, shape=[None, ])

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.config.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * self.config.end_token

            # decoder_inputs: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs = tf.concat([decoder_start_token,
                                             self.target_tokens], axis=1)

            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = self.target_length + 1

            # decoder_targets: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets = tf.concat([self.target_tokens,
                                              decoder_end_token], axis=1)

    def setup_embedding(self):
        with tf.variable_scope("Embedding"):
            with tf.device('/cpu:0'):
                self.encode_embedding, self.decode_embedding = create_emb_for_encoder_and_decoder(
                    self.config.share_vocab, self.config.src_vocab_size, self.config.tgt_vocab_size, self.config.embedding_size, self.config.embedding_size)
            self.encode_inputs = tf.nn.embedding_lookup(
                self.encode_embedding, self.source_tokens)
            if self.train_phase:
                self.decoder_inputs = tf.nn.embedding_lookup(
                    self.decode_embedding, self.decoder_inputs)

    def setup_multilayer_encoder(self):
        with tf.variable_scope("Encoder"):
            encode_cell = multi_rnn_cell(self.config.encode_cell_type, self.config.num_units,
                                         self.config.encode_layer_num, self.train_phase,
                                         self.config.keep_prob, 0, self.config.num_gpus)
            outputs, states = tf.nn.dynamic_rnn(
                encode_cell, inputs=self.encode_inputs, sequence_length=self.source_length, dtype=tf.float32)
        self.encode_output = outputs
        self.encode_state = states
        self.decode_initial_state = states[-1]

    def setup_bidirection_encoder(self):
        num_bi_layers = int(self.config.encode_layer_num / 2)

        with tf.variable_scope("Encoder"):
            fw_cell = multi_rnn_cell(self.config.encode_cell_type, self.config.num_units,
                                     num_bi_layers, self.train_phase, self.config.keep_prob, 0, self.config.num_gpus)
            bw_cell = multi_rnn_cell(self.config.encode_cell_type, self.config.num_units, num_bi_layers,
                                     self.train_phase, self.config.keep_prob, 0, self.config.num_gpus, num_bi_layers)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                dtype=tf.float32,
                sequence_length=self.source_length,
                inputs=self.encode_inputs
            )
            outputs_concat = tf.concat(outputs, 2)
        self.encode_output = outputs_concat
        self.encode_state = states

        # use Dense layer to convert bi-direction state to decoder inital state
        with tf.variable_scope("Bi_Encode_State_Convert"):
            convert_layer = Dense(
                self.config.num_units, dtype=tf.float32, name="bi_convert")
            self.decode_initial_state = convert_layer(
                tf.concat(self.encode_state, axis=1))

    def setup_attention_decoder(self):
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

            decode_cell[0] = tf.contrib.seq2seq.AttentionWrapper(
                cell=decode_cell[0],
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
            # initial state for attention cell
            attention_cell_state = decode_cell[0].zero_state(
                dtype=tf.float32, batch_size=batch_size)
            initial_state[0] = attention_cell_state.clone(
                cell_state=initial_state[0])
            self.initial_state = tuple(initial_state)
            self.decode_cell = tf.contrib.rnn.MultiRNNCell(decode_cell)
            self.output_layer = Dense(
                self.config.tgt_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    def setup_beam_search(self):
        start_tokens = tf.fill([self.batch_size], self.config.start_token)
        bsd = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decode_cell,
            embedding=self.decode_embedding,
            start_tokens=start_tokens,
            end_token=self.config.end_token,
            initial_state=self.initial_state,
            beam_width=self.config.beam_size,
            output_layer=self.output_layer,
            length_penalty_weight=0.0)
        # final_outputs are instances of FinalBeamSearchDecoderOutput
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            bsd,
            output_time_major=False,
            # impute_finished=True,
            maximum_iterations=self.config.max_inference_length
        )
        beam_predictions = final_outputs.predicted_ids
        # TODO: found will got -1, beam search will add -1 when meet end token? and tf will delete start tokens?
        # I will spend some time to read the source code of tf.contrib.seq2seq and tf.contrib.rnn
        self.beam_predictions = tf.transpose(beam_predictions, perm=[0, 2, 1])

    def setup_training_decode_layer(self):
        max_dec_len = tf.reduce_max(
            self.decoder_inputs_length, name="max_dec_len")
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_inputs,
            self.decoder_inputs_length
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decode_cell,
            helper=training_helper,
            initial_state=self.initial_state,
            output_layer=self.output_layer
        )

        train_dec_outputs, train_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_dec_len)

        # logits: [batch_size x max_dec_len x vocab_size]
        logits = tf.identity(train_dec_outputs.rnn_output, name='logits')

        masks = tf.sequence_mask(
            self.decoder_inputs_length, max_dec_len, dtype=tf.float32, name="mask")

        # targets: [batch_size x max_dec_len]
        # this is important, because we may have padded endings
        targets = tf.slice(self.decoder_targets, [
                           0, 0], [-1, max_dec_len], 'targets')

        # self.losses = tf.contrib.seq2seq.sequence_loss(
        #    logits=logits, targets=targets, weights=masks, name="losses", average_across_timesteps=True, average_across_batch=True,)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)
        self.losses = tf.reduce_sum(
            crossent * masks) / tf.to_float(self.batch_size)
        # prediction sample for validation
        self.valid_predictions = tf.identity(
            train_dec_outputs.sample_id, name='valid_preds')

    def train_one_batch(self, source_tokens, source_length, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        losses, _ = self.sess.run(
            [self.losses, self.updates], feed_dict=feed_dict)
        return losses

    def eval_one_batch(self, source_tokens, source_length, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        predicted_ids, losses = self.sess.run(
            [self.valid_predictions, self.losses], feed_dict=feed_dict)
        return predicted_ids, losses

    def inference(self, source_tokens, source_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        predictions = self.sess.run(self.beam_predictions, feed_dict=feed_dict)
        return predictions
