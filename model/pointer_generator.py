import collections
import os

import tensorflow as tf

from model.basic_s2s_model import BasicS2SModel
from model.config import BasicConfig
from utils.data_util import UNK_ID
from utils.model_util import create_attention_mechanism, get_cell_list
from utils.pointer_generator_helper import PointerGeneratorDecoder, PointerGeneratorGreedyEmbeddingHelper, PointerGeneratorBahdanauAttention,PointerGeneratorAttentionWrapper

class PointerGeneratorModel(BasicS2SModel):
    def __init__(self,sess,config=BasicConfig()):
        super(PointerGeneratorModel,self).__init__(sess,config)

    def setup_input_placeholders(self):
        # the original input tokens, using the vocab
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None], name="source_tokens")
        self.source_length = tf.placeholder(tf.int32, shape=[None, ], name='source_length')
        # max number of oov words in this batch
        self.source_oov_words = tf.placeholder(tf.int32, shape=[], name='batch_oov_words')
        # input tokens using source oov words and vocab 
        self.source_extend_tokens = tf.placeholder(tf.int32, shape=[None,None], name='source_extend_tokens')
        
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

    def convert_to_coverage_model(self):
        print("restore old model")
        saver = tf.train.Saver([v for v in tf.global_variables() if 'coverage' not in v.name and "Adagrad" not in v.name])
        saver.restore(self.sess, tf.train.latest_checkpoint(
                self.config.checkpoint_dir))
        
        new_fname = self.config.checkpoint_dir + '_coverage'
        new_saver = tf.train.Saver() # this one will save all variables that now exist
        print("save to new model")
        new_saver.save(self.sess, os.path.join(new_fname,"model.ckpt"), global_step=self.global_step.eval(self.sess))
        self.saver = new_saver
        self.config.checkpoint_dir = new_fname

    def setup_attention_decoder(self):
        print("setup attention decoder")
        with tf.variable_scope("Decoder"):
            # multi-layer decoder
            decode_cell = get_cell_list(self.config.decode_cell_type, self.config.num_units,
                                        self.config.decode_layer_num, 0, self.train_phase, self.config.num_gpus, 0, self.config.keep_prob)

            memory = self.encode_output
            memory_length = self.source_length
            batch_size = self.batch_size

            # we don't support other attention right now
            # you need implement other attention mechanism by yourself
            atten_mech = PointerGeneratorBahdanauAttention(
                self.config.num_units, memory, memory_sequence_length=memory_length, coverage = self.config.coverage)

            decode_cell[0] = PointerGeneratorAttentionWrapper(
                cell=decode_cell[0],
                attention_mechanism=atten_mech,
                attention_layer_size=self.config.num_units,
                alignment_history = True,
                coverage = self.config.coverage
            )

            # setup initial state of decoder
            initial_state = [self.decode_initial_state for i in range(
                 self.config.decode_layer_num)]
            # initial state for attention cell
            initial_state[0] = decode_cell[0].zero_state(dtype=tf.float32, batch_size=batch_size).clone(
                cell_state=self.decode_initial_state)
            self.initial_state = tuple(initial_state)
            self.decode_cell = tf.contrib.rnn.MultiRNNCell(decode_cell)

    def setup_beam_search(self):
        print("setup beam search")
        # using greedying decoder right now
        helper = PointerGeneratorGreedyEmbeddingHelper(
            embedding=self.decode_embedding,
            start_tokens=tf.tile([self.config.start_token], [self.batch_size]),
            end_token=self.config.end_token
        )

        inference_decoder = PointerGeneratorDecoder(
            source_extend_tokens = self.source_extend_tokens,
            source_oov_words = self.source_oov_words,
            coverage = self.config.coverage,
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

    def setup_training_decode_layer(self):
        print("setup training decoder")
        max_dec_len = tf.reduce_max(
            self.decoder_inputs_length, name="max_dec_len")

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_inputs,
            self.decoder_inputs_length
        )

        training_decoder = PointerGeneratorDecoder(
            source_extend_tokens = self.source_extend_tokens,
            source_oov_words = self.source_oov_words,
            coverage = self.config.coverage,
            cell=self.decode_cell,
            helper=training_helper,
            initial_state=self.initial_state,
            output_layer=self.output_layer
        )

        train_dec_outputs, train_dec_last_state, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_dec_len,
            swap_memory=True)

        # output of decoder
        # batch size * decoder length * extend vocab size
        logits = train_dec_outputs.rnn_output
        self.logits = logits
        # setup loss
        '''
            Note: the author didn't use sparse_softmax_cross_entropy_with_logits with final dists. I will try to use it later.
            final dist has already been softmax, so we need pick the correct prob and calculate the cross entroy by ourselves
            this is almost the same as final dist but we use tf.gather_nd instead
            suppose we have finnal dist as:
            array([[[ 0.3451128 ,  0.89498031,  0.97154093,  0.14212787,  0.75319886,
                    0.04996049,  0.69889939,  0.4412986 ,  0.75493288,  0.27171338,
                    0.18648636,  0.4576633 ],
                    [ 0.21023917,  0.90042853,  0.91051352,  0.05150759,  0.2882216 ,
                    0.92774153,  0.90885901,  0.28111482,  0.73560727,  0.35180628,
                    0.23423457,  0.65652728],
                    [ 0.88603485,  0.73089767,  0.74546456,  0.547315  ,  0.03917027,
                    0.66219461,  0.28154099,  0.07492244,  0.69947207,  0.09645844,
                    0.89412236,  0.92925715],
                    [ 0.37891686,  0.55921805,  0.14969325,  0.25432551,  0.49430585,
                    0.87305665,  0.32583809,  0.9329983 ,  0.96388829,  0.10060763,
                    0.11731851,  0.16561043],
                    [ 0.69298077,  0.73208559,  0.48923552,  0.22197175,  0.62429607,
                    0.03505456,  0.7998606 ,  0.67086065,  0.35519254,  0.09402156,
                    0.64395237,  0.55462575],
                    [ 0.9918412 ,  0.19353426,  0.56901765,  0.24497354,  0.64668715,
                    0.30509686,  0.86582935,  0.80181813,  0.57659829,  0.88711154,
                    0.98655462,  0.78871655],
                    [ 0.78579485,  0.21403694,  0.65902781,  0.87479532,  0.11452317,
                    0.7675612 ,  0.02025175,  0.69162011,  0.30777824,  0.17244208,
                    0.70707357,  0.43202615],
                    [ 0.4054718 ,  0.43004405,  0.4875772 ,  0.30384171,  0.11415899,
                    0.53144836,  0.30152166,  0.61763275,  0.29379785,  0.27471113,
                    0.31000984,  0.31739724]]]])
            then for each decoder step, we need get the truth prob of the target word, which means our indices should like belowing:
            array([[[ 0,  0,  9],
                    [ 0,  1,  8],
                    [ 0,  2,  5],
                    [ 0,  3, 11],
                    [ 0,  4,  1],
                    [ 0,  5,  6],
                    [ 0,  6,  4],
                    [ 0,  7, 11]]]])

            This is more convinent if we use tf.meshgrid:
            i1, i2 = tf.meshgrid(tf.range(2),
                     tf.range(8), indexing="ij")
            indices = tf.stack((i,i2,decoder_targets),axis=2)

            Just check the code
        '''
        masks = tf.sequence_mask(
            self.decoder_inputs_length, max_dec_len, dtype=tf.float32, name="mask")

        # targets: [batch_size x max_dec_len]
        # this is important, because we may have padded endings
        targets = tf.slice(self.decoder_targets, [
                           0, 0], [-1, max_dec_len], 'targets')

        i1, i2 = tf.meshgrid(tf.range(self.batch_size),
                     tf.range(max_dec_len), indexing="ij")
        indices = tf.stack((i1,i2,targets),axis=2)
        probs = tf.gather_nd(logits, indices)

        # To prevent padding tokens got 0 prob, and get inf when calculating log(p), we set the lower bound of prob
        # I spent a lot of time here to debug the nan losses, inf * 0 = nan
        probs = tf.where(tf.less_equal(probs,0),tf.ones_like(probs)*1e-10,probs)
        
        #is_nan = tf.reduce_sum(tf.cast(tf.is_nan(probs),tf.int32))
        #is_inf = tf.reduce_sum(tf.cast(tf.is_inf(probs),tf.int32))
        #x = tf.reduce_sum(tf.cast(tf.equal(probs[0], 0),tf.int32))
        #probs = tf.Print(probs,[x],message="prob is nan or inf: ")

        crossent = -tf.log(probs)

        #is_nan = tf.cast(tf.is_nan(crossent),tf.int32)
        #is_inf = tf.cast(tf.is_inf(crossent),tf.int32)
        #crossent = tf.Print(crossent, [tf.reduce_sum(is_inf),tf.reduce_sum(is_nan), crossent],message="crossent is nan or inf: ")
        #alignment_history = train_dec_last_state[0].alignment_history.stack()
        #alignments = train_dec_last_state[0].alignments
        #alignment_history = tf.transpose(alignment_history,[1,2,0])
        
        #crossent = tf.Print(crossent,[tf.shape(alignments),tf.shape(alignment_history)],message='loss')
        self.losses = tf.reduce_sum(
            crossent * masks) / tf.to_float(self.batch_size)

        if self.config.coverage:
            # we got all the alignments from last state
            # shape is: batch * atten_len * max_len
            alignment_history = train_dec_last_state[0].alignment_history.stack()
            alignment_history = tf.transpose(alignment_history,[1,2,0])
            coverage_loss = tf.minimum(alignment_history,tf.cumsum(alignment_history, axis=2, exclusive=True))
            # debug
            #coverage_loss = tf.Print(coverage_loss,[coverage_loss,tf.shape(coverage_loss)],message='loss')
            coverage_loss = self.config.coverage_loss_ratio * tf.reduce_sum(coverage_loss / tf.to_float(self.batch_size))
            self.losses = self.losses + coverage_loss
            self.coverage_loss = coverage_loss

        # prediction sample for validation
        self.valid_predictions = tf.cast(tf.argmax(logits,axis=-1), tf.int32, name='valid_predict')

    def train_one_batch(self, source_tokens, source_length, source_oov_words, source_extend_tokens, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_oov_words] = source_oov_words
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        losses, summary, global_step, _ = self.sess.run(
            [self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        return losses, global_step

    def train_coverage_one_batch(self, source_tokens, source_length, source_oov_words, source_extend_tokens, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_oov_words] = source_oov_words
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        cov_loss, losses, summary, global_step, _ = self.sess.run(
            [self.coverage_loss, self.losses, self.summary_op, self.global_step, self.updates], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        return cov_loss, losses, global_step

    def eval_one_batch(self, source_tokens, source_length,  source_oov_words, source_extend_tokens, target_tokens, target_length):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_oov_words] = source_oov_words
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        feed_dict[self.target_tokens] = target_tokens
        feed_dict[self.target_length] = target_length
        predicted_ids, losses, logits = self.sess.run(
            [self.valid_predictions, self.losses, self.logits], feed_dict=feed_dict)
        return predicted_ids, losses, logits

    def inference(self, source_tokens, source_length, source_oov_words, source_extend_tokens,):
        feed_dict = {}
        feed_dict[self.source_tokens] = source_tokens
        feed_dict[self.source_length] = source_length
        feed_dict[self.source_oov_words] = source_oov_words
        feed_dict[self.source_extend_tokens] = source_extend_tokens
        predictions = self.sess.run(self.beam_predictions, feed_dict=feed_dict)
        return predictions
