import os
import random
import time
import pickle

from collections import Counter
import tensorflow as tf

from txtsum.pg_data_utils import get_batch, convert_ids_to_sentences
from model.pointer_generator import PointerGeneratorModel
from model.config import PointGeneratorConfig
from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from utils.model_util import get_config_proto

DATA_DIR = "/data/xueyou/textsum/lcsts_0507"
w2i,i2w = read_vocab(os.path.join(DATA_DIR,'vocab.txt'))
train_source_file = os.path.join(DATA_DIR,"train.source")
train_target_file = os.path.join(DATA_DIR,"train.target")

config = PointGeneratorConfig()
config.src_vocab_size = len(w2i)
config.tgt_vocab_size = len(w2i)
config.start_token = SOS_ID
config.end_token = EOS_ID
config.use_bidirection = True
config.encode_layer_num = 2
config.decode_layer_num = 3
config.num_units = 512
config.embedding_size = 256
config.encode_cell_type = 'lstm'
config.decode_cell_type = 'lstm'
config.batch_size = 128
config.checkpoint_dir = os.path.join(DATA_DIR,"pointer_generator_lstm_0528")
config.num_gpus = 1
config.num_train_steps = 300000
config.optimizer = 'adagrad'
config.learning_rate = 0.15
config.decay_scheme = 'luong10'
config.coverage = False
config.src_vocab_file = os.path.join(DATA_DIR,"vocab.txt")

pickle.dump(config, open(os.path.join(DATA_DIR,"config.pkl"),'wb'))

with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
    model = PointerGeneratorModel(sess, config)
    sess.run(tf.global_variables_initializer())

    if config.coverage:
        model.convert_to_coverage_model()

    try:
        model.restore_model()
        print("restore model successfully")
    except Exception as e:
        print(e)
        print("fail to load model")

    epoch = 0
    step = 0
    losses = 0.
    cov_losses = 0.
    cov_loss = 0.
    step_time = 0.0
    step_per_show = 100
    step_per_predict = 1000
    step_per_save = 10000
    global_step = model.global_step.eval(session=sess)
    while global_step < config.num_train_steps:         
        for batch in get_batch(w2i, train_source_file, train_target_file, config.batch_size):
            step += 1
            source_tokens, source_lengths, source_extend_tokens, source_oovs, target_tokens, target_length, max_oovs= batch
            start = time.time()
            if config.coverage:
                cov_loss, batch_loss, global_step = model.train_coverage_one_batch(source_tokens, source_lengths, max_oovs, source_extend_tokens, target_tokens, target_length)
                cov_losses += cov_loss
            else:
                batch_loss, global_step = model.train_one_batch(source_tokens, source_lengths, max_oovs, source_extend_tokens, target_tokens, target_length)
            end = time.time()
            losses += batch_loss
            step_time += (end-start)
            if step % step_per_show == 0:
                print("Epoch {0}, step {1}, loss {2}, cov_loss {4}, step-time {3}".format(epoch + 1, global_step, losses/step_per_show, step_time/step_per_show, cov_losses/step_per_show))
                losses = 0.0
                step_time = 0.0
                cov_losses = 0.0

            if step % step_per_predict == 0:
                predictions,_,_ = model.eval_one_batch(source_tokens, source_lengths, max_oovs, source_extend_tokens, target_tokens, target_length)
                idx = random.sample(range(len(source_tokens)),1)[0]
                oovs = source_oovs[idx]
                print("Input:", convert_ids_to_sentences(source_tokens[idx],i2w,oovs))
                print("Input Extend:", convert_ids_to_sentences(source_extend_tokens[idx],i2w,oovs))
                print("Prediction:",convert_ids_to_sentences(predictions[idx],i2w,oovs))
                print("Truth:", convert_ids_to_sentences(target_tokens[idx],i2w,oovs))

            if step % step_per_save == 0:
                model.save_model()

        model.save_model()
        epoch += 1


