from model.basic_s2s_model import BasicS2SModel
from model.config import BasicConfig
import tensorflow as tf

class S2SModelWithPipeline(BasicS2SModel):
    def __init__(self,sess,data_iterator,config=BasicConfig()):
        self.iterator = data_iterator        
        super(S2SModelWithPipeline,self).__init__(sess,config)

    def setup_input_placeholders(self):
        self.batch_size = tf.size(self.iterator.source_sequence_length)
        self.source_tokens = self.iterator.source
        self.source_length = self.iterator.source_sequence_length

        if self.train_phase:
            self.decoder_inputs = self.iterator.target_input
            self.decoder_inputs_length = self.iterator.target_sequence_length
            self.decoder_targets = self.iterator.target_output

            # To calculate ppl
            self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

    def init(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

    def train_one_batch(self):
        _,loss,predict_count,global_step,batch_size,summary = self.sess.run([self.updates,self.losses,self.predict_count,self.global_step,self.batch_size,self.summary_op])
        self.summary_writer.add_summary(summary, global_step)
        return loss,predict_count,global_step,batch_size
