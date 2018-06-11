import os
import random
import time

from collections import Counter
import tensorflow as tf

from model.copynet import CopyNet
from model.config import CopyNetConfig
from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from utils.model_util import get_config_proto
from txtsum.copy_data_utils import get_batch, convert_ids_to_sentences
from utils.bleu import compute_bleu
from utils.rouge import rouge
import pickle

DATA_DIR = "/data/xueyou/textsum/lcsts_0507"
w2i,i2w = read_vocab(os.path.join(DATA_DIR,'vocab.txt'))
train_source_file = os.path.join(DATA_DIR,"train.source")
train_target_file = os.path.join(DATA_DIR,"train.target")

config = CopyNetConfig()
config.src_vocab_size = len(w2i)
config.tgt_vocab_size = len(w2i)
config.start_token = SOS_ID
config.end_token = EOS_ID
config.use_bidirection = True
config.encode_layer_num = 2
config.decode_layer_num = 4
config.num_units = 512
config.embedding_size = 256
config.encode_cell_type = 'lstm'
config.decode_cell_type = 'lstm'
config.batch_size = 256
config.checkpoint_dir = os.path.join(DATA_DIR,"copynet_0604")
if not os.path.isdir(config.checkpoint_dir):
    os.mkdir(config.checkpoint_dir)
config.max_oovs = 200
config.num_gpus = 1
config.num_train_steps = 300000
# using Adam, not decay schema
config.decay_scheme = None 
config.optimizer = 'adagrad'
config.learning_rate = 0.15
config.max_inference_length = 25
config.src_vocab_file = os.path.join(DATA_DIR,"vocab.txt")
config.src_pretrained_embedding = os.path.join(DATA_DIR,"pretrained_w2v_50000_glove.txt")

pickle.dump(config, open(os.path.join(config.checkpoint_dir,"config.pkl"),'wb'))

with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
    model = CopyNet(sess, config)
    sess.run(tf.global_variables_initializer())

    try:
        model.restore_model()
        print("restore model successfully")
    except:
        print("fail to load model")

    epoch = 0
    step = 0
    losses = 0.
    step_time = 0.0
    step_pre_show = 10
    step_pre_predict = 500
    step_pre_save = 5000
    global_step = model.global_step.eval(session=sess)
    while global_step < config.num_train_steps:         
        for batch in get_batch(config.max_oovs, w2i, train_source_file, train_target_file, config.batch_size):
            step += 1
            source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length, batch_oovs = batch
            start = time.time()
            batch_loss, global_step = model.train_one_batch(source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length, True if step%step_pre_save==0 else False)
            end = time.time()
            losses += batch_loss
            step_time += (end-start)
            if step % step_pre_show == 0:
                print("Epoch {0}, step {1}, loss {2}, step-time {3}".format(epoch + 1, global_step, losses/step_pre_show, step_time/step_pre_show))
                losses = 0.0
                step_time = 0.0

            if step % step_pre_predict == 0:
                predictions,_,_ = model.eval_one_batch(source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length)
                idx = random.sample(range(len(source_tokens)),1)[0]
                oovs = {v:k for k,v in batch_oovs.items()}
                print("Input:", convert_ids_to_sentences(source_tokens[idx],i2w,oovs))
                print("Input Extend:", convert_ids_to_sentences(source_extend_tokens[idx],i2w,oovs))
                print("Prediction:",convert_ids_to_sentences(predictions[idx],i2w,oovs))
                print("Truth:", convert_ids_to_sentences(target_tokens[idx],i2w,oovs))

            if step % step_pre_save == 0:
                model.save_model()
        epoch += 1
        model.save_model()


