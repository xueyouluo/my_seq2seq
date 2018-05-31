import math
import os
import pickle
import time

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from model.config import BasicConfig
from utils.model_util import get_config_proto
from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from utils.data_util import (EOS, EOS_ID, SOS, SOS_ID, UNK, UNK_ID,
                             create_vocab, get_train_iterator, read_vocab)

if __name__ == "__main__":
    data_dir = "/data/xueyou/textsum/lcsts_0507"
    src_vocab_file = os.path.join(data_dir,"vocab.source")
    tgt_vocab_file = os.path.join(data_dir,"vocab.target")
    train_src_file = os.path.join(data_dir,"train.source")
    train_tgt_file = os.path.join(data_dir,"train.target")

    config = BasicConfig()
    src_w2i,i2w = read_vocab(src_vocab_file)
    tgt_w2i,_ = read_vocab(tgt_vocab_file)

    config.src_vocab_size = len(src_w2i)
    config.tgt_vocab_size = len(tgt_w2i)
    config.start_token = SOS_ID
    config.end_token = EOS_ID
    config.use_bidirection = True
    config.decode_layer_num = 4
    config.num_units = 512
    config.embedding_size = 256
    config.encode_cell_type = 'lstm'
    config.decode_cell_type = 'lstm'
    config.batch_size = 128
    config.attention_option = "bahdanau"
    config.checkpoint_dir = data_dir + "/baseline"
    config.reverse_source = False
    config.optimizer = 'adagrad'
    config.learning_rate = 0.15
    config.max_inference_length = 25
    config.beam_size = 1
    config.src_vocab_file = os.path.join(data_dir,"vocab.txt")
    config.src_pretrained_embedding = os.path.join(data_dir,"pretrained_w2v_50000_glove.txt")
    # test with 2 gpus, set to 1 if you only have 1 gpu
    config.num_gpus = 1

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    pickle.dump(config,open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

    with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
        # got error if we use tf.contrib.lookup.index_table_from_file
        src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)

        train_src_dataset = tf.contrib.data.TextLineDataset(train_src_file)
        train_tgt_dataset = tf.contrib.data.TextLineDataset(train_tgt_file)

        train_skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        train_iterator = get_train_iterator(
            train_src_dataset,
            train_tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=config.batch_size,
            sos=SOS,
            eos=EOS,
            source_reverse=config.reverse_source,
            random_seed=201,
            num_buckets=5,
            src_max_len=85,
            tgt_max_len=25,
            skip_count=train_skip_count_placeholder)
            
        model = S2SModelWithPipeline(sess,train_iterator,config)
        try:
            model.restore_model()
        except Exception as e:
            print("fail to load model with error: {0}".format(e))
            model.init()

        step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
        train_ppl = 0.0
        start_train_time = time.time()

        print("# Start to train with learning rate {0}, {1}".format(config.learning_rate,time.ctime()))

        # Initialize all of the iterators
        sess.run(
            train_iterator.initializer,
            feed_dict={
                train_skip_count_placeholder: 0
            })

        global_step = model.global_step.eval(session=sess)
        steps_per_stats = 100
        steps_per_save = 10000
        num_train_steps = 100000
        while global_step < num_train_steps:
            start_time = time.time()
            try:
                step_result = model.train_one_batch()
                (step_loss, step_predict_count, global_step, batch_size) = step_result
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset.  Go to next epoch.
                print("# Finished an epoch, step %d. Perform external evaluation" % global_step)
                sess.run(
                    train_iterator.initializer,
                    feed_dict={
                        train_skip_count_placeholder: 0
                    })
                model.save_model()
                continue

            step_time += (time.time() - start_time)
            checkpoint_loss += (step_loss * batch_size)
            checkpoint_predict_count += step_predict_count

            def safe_exp(value):
                """Exponentiation with catching of overflow error."""
                try:
                    ans = math.exp(value)
                except OverflowError:
                    ans = float("inf")
                return ans

            # Once in a while, we print statistics.
            if global_step % steps_per_stats == 0:
                # Print statistics for the previous epoch.
                avg_step_time = step_time / steps_per_stats
                train_ppl = safe_exp(checkpoint_loss / checkpoint_predict_count)
                print(
                    "#  global step %d lr %g "
                    "step-time %.2fs ppl %.2f" %
                    (global_step, model.learning_rate.eval(session=sess),
                    avg_step_time, train_ppl))
                if math.isnan(train_ppl): break

                source,target,predictions = model.eval_one_batch()
                
                print(''.join([i2w[i] for i in source[0]]))
                print(''.join([i2w[i] for i in target[0]]))
                print(''.join([i2w[i] for i in predictions[0]]))
                
                # Reset timer and loss.
                step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0

            if global_step % steps_per_save == 0:
                model.save_model()
            # TODO: add eval data, add bleu metric ...

        # save model after train
        model.save_model()
