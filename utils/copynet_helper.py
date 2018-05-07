import tensorflow as tf
import collections
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.layers.core import Dense

class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("cell_state", "last_ids", "prob_c"))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))

class CopyNetWrapper(tf.contrib.rnn.RNNCell):
    '''
    A copynet RNN cell wrapper
    '''
    def __init__(self, cell, source_extend_tokens, max_oovs, encode_output, output_layer, vocab_size, name=None):
        '''
        Args:
            - cell: the decoder cell
            - source_extend_tokens: input tokens with oov word ids
            - max_oovs: max number of oov words in each batch
            - encode_output: the output of encoder cell
            - output_layer: the layer used to map decoder output to vocab distribution
            - vocab_size: this is target vocab size
        '''
        super(CopyNetWrapper, self).__init__(name=name)
        self.cell = cell
        self.source_extend_tokens = source_extend_tokens
        self.encode_output = encode_output
        self.max_oovs = max_oovs
        self.output_layer = output_layer
        self._output_size = vocab_size + max_oovs
        self.copy_layer = Dense(self.cell.output_size,activation=tf.tanh,use_bias=False,name="Copy_Weigth")

    def __call__(self, inputs, state):
        with tf.variable_scope("CopyNetWrapper"):
            last_ids = state.last_ids
            prob_c = state.prob_c
            cell_state = state.cell_state
            
            '''
                - Selective read
                    At first, my implement is based on the paper, which looks like belowing:
                        mask = tf.cast(tf.equal(tf.expand_dims(last_ids,1),self.source_extend_tokens), tf.float32)
                        pt = mask * prob_c
                        pt_sum = tf.reduce_sum(pt, axis=1)
                        pt = tf.where(tf.less(pt_sum, 1e-7), pt, pt / tf.expand_dims(pt_sum, axis=1))
                        selective_read = tf.einsum("ijk,ij->ik",self.encode_output, pt)
                    It looks OK for me, but I got NAN loss after one training step. I tried tf.Print to debug, but cannot find 
                    the problem. Then I tried tfdbg to detect the source of NAN and found it refer to the tf.where code piece, 
                    which is 'pt / tf.expand_dims(pt_sum, axis=1)'. 
                    I am not sure why this code will got NANs. Maybe pt and pt_sum are not of the same shape, due to the broadcasting,
                    we will got NANs, and these NANs will be used to calculate the gradients, and cause NANs values.

                    Take a close look at the paper, we will find that p(y_t,c|.) is same for all the same input tokens. And selective_read
                    is sum of p_t * h_t, in which p_t is 1/K *(p(x_t,c|.)), K is sum of p(x_t,c|.). So p_t is equal to:
                    p_t = 1 / (number of times x_t shows in input tokens)
            '''
            with tf.variable_scope("selective_read"):
                # get selective read
                # batch * input length
                mask = tf.cast(tf.equal(tf.expand_dims(last_ids,1),self.source_extend_tokens), tf.float32)
                mask_sum = tf.reduce_sum(mask, axis=1)
                mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, axis=1))
                pt = mask * prob_c
                selective_read = tf.einsum("ijk,ij->ik",self.encode_output, pt)

            inputs = tf.concat([inputs, selective_read], axis=-1)
            outputs, cell_state = self.cell(inputs, cell_state)

            with tf.variable_scope("generate_mode"):
                # this is generate mode
                vocab_dist = self.output_layer(outputs)

                vocab_size = tf.shape(vocab_dist)[-1]
                extended_vsize = vocab_size + self.max_oovs 

                batch_size = tf.shape(vocab_dist)[0]
                extra_zeros = tf.zeros((batch_size, self.max_oovs))
                # batch * extend vocab size
                vocab_dists_extended = tf.concat(axis=-1, values=[vocab_dist, extra_zeros])

            with tf.variable_scope("copy_mode"):
                # this is copy mode
                # batch * length * output size
                copy_score = self.copy_layer(self.encode_output)
                # batch * length
                copy_score = tf.einsum("ijk,ik->ij",copy_score,outputs)
                
                # this part is same as that of point generator, but using einsum is much simpler.
                # although einsum is easy to understand, but using one_hot will comsume lots of 
                # memory, I choose to use the old trick again.

                #source_mask = tf.one_hot(self.source_extend_tokens,extended_vsize)
                #attn_dists_projected = tf.einsum("ijn,ij->in", source_mask, copy_score)
                
                batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
                batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
                attn_len = tf.shape(self.source_extend_tokens)[1] # number of states we attend over
                batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
                indices = tf.stack((batch_nums, self.source_extend_tokens), axis=2) # shape (batch_size, enc_t, 2)
                shape = [batch_size, extended_vsize]
                attn_dists_projected = tf.scatter_nd(indices, copy_score, shape)
                
                final_dist = vocab_dists_extended + attn_dists_projected

            # This is greeding search, need to test with beam search
            last_ids = tf.argmax(final_dist, axis=-1, output_type=tf.int32)

            with tf.variable_scope("p_c"):
                # this is used to calculate p(y_t,c|.)
                # safe softmax
                final_dist_max = tf.expand_dims(tf.reduce_max(final_dist,axis=1), axis=1)
                final_dist_exp = tf.reduce_sum(tf.exp(final_dist - final_dist_max),axis=1)
                p_c = tf.exp(attn_dists_projected - final_dist_max) / tf.expand_dims(final_dist_exp, axis=1)
                #p_c = tf.einsum("ijn,in->ij",source_mask,p_c)
                p_c = tf.gather_nd(p_c, indices)

        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=p_c)
        return final_dist, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self.cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = self.source_extend_tokens.shape[-1].value)

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._output_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self.cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.zeros([batch_size, tf.shape(self.encode_output)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)