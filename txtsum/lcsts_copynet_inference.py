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

DATA_DIR = "/data/xueyou/textsum/lcsts_0507"
w2i,i2w = read_vocab(os.path.join(DATA_DIR,'vocab.txt'))
train_source_file = os.path.join(DATA_DIR,"test.source")
train_target_file = os.path.join(DATA_DIR,"test.target")

config = CopyNetConfig()
config.mode = 'inference'
config.src_vocab_size = len(w2i)
config.tgt_vocab_size = len(w2i)
config.start_token = SOS_ID
config.end_token = EOS_ID
config.use_bidirection = True
config.encode_layer_num = 2
config.decode_layer_num = 4
config.num_units = 512
config.embedding_size = 256
config.encode_cell_type = 'gru'
config.decode_cell_type = 'gru'
config.batch_size = 24
config.beam_size = 1
config.checkpoint_dir = os.path.join(DATA_DIR,"copynet_new")
config.max_oovs = 200

with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
    model = CopyNet(sess, config)

    try:
        model.restore_model()
        print("restore model successfully")
    except Exception as e:
        print(e)
        print("fail to load model")

    for batch in get_batch(config.max_oovs, w2i, train_source_file, train_target_file, config.batch_size):
        source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length, batch_oovs = batch
        predictions = model.inference(source_tokens, source_lengths, source_extend_tokens)
        idx = random.sample(range(len(source_tokens)),1)[0]
        oovs = {v:k for k,v in batch_oovs.items()}
        print("Input:", convert_ids_to_sentences(source_tokens[idx],i2w,oovs))
        print("Input Extend:", convert_ids_to_sentences(source_extend_tokens[idx],i2w,oovs))
        print("Prediction:",convert_ids_to_sentences(predictions[idx][0],i2w,oovs))
        print("Truth:", convert_ids_to_sentences(target_tokens[idx],i2w,oovs))



