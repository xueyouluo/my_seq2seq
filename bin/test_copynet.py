# we need set the cuda_visible_devices firstly to avoid create session errors
import os
# depends on your GPU resources
os.environ['CUDA_VISIBLE_DEVICES']='0'

from utils.data_util import create_vocab,UNK_ID,SOS_ID,EOS_ID
from tensorflow.python import debug as tf_debug
DATA_DIR = "toy_data/"
train_source_file = os.path.join(DATA_DIR,"train","sources.txt")
w2i,i2w,_ = create_vocab(train_source_file,max_size=17)

def sent2idx(sent, vocab=w2i, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

    pad_length = max_sentence_length - current_length
    oovs = []
    extend_tokens = []
    tokenized = []
    for token in tokens:
        if token not in vocab:
            tokenized.append(UNK_ID)
            if token not in oovs:
                oovs.append(token)
            extend_tokens.append(len(vocab) + oovs.index(token))
        else:
            extend_tokens.append(vocab[token])
            tokenized.append(vocab[token])
    return tokenized + [2] * pad_length, extend_tokens + [2] * pad_length, oovs, current_length

def target2idx(sent, oovs, vocab=w2i, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

    pad_length = max_sentence_length - current_length
    tokenized = []
    for token in tokens:
        if token not in vocab:
            if token not in oovs:
                tokenized.append(UNK_ID)
            else:
                tokenized.append(len(vocab) + oovs.index(token))
        else:
            tokenized.append(vocab[token])
    return tokenized + [2] * pad_length, current_length



train_source_file = os.path.join(DATA_DIR,"train","sources.txt")
source_lines = open(train_source_file).readlines()
train_target_file = os.path.join(DATA_DIR,"train","targets.txt")
target_lines = open(train_target_file).readlines()


from model.copynet import CopyNet
from model.config import CopyNetConfig
from utils.data_util import SOS_ID, EOS_ID


config = CopyNetConfig()
config.src_vocab_size = len(w2i)
config.tgt_vocab_size = len(w2i)
config.start_token = SOS_ID
config.end_token = EOS_ID
config.use_bidirection = True
config.encode_layer_num = 2
config.decode_layer_num = 1
config.num_units = 64
config.embedding_size = 64
config.encode_cell_type = 'gru'
config.decode_cell_type = 'gru'
config.batch_size = 64
config.checkpoint_dir = "/tmp/test_copynet/"
config.max_oovs = 50

import tensorflow as tf

sess = tf.InteractiveSession()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
model = CopyNet(sess,config)

def get_batch(batch_size):
    input_batch_tokens = []
    input_batch_extend_tokens = []
    input_oovs = []
    target_batch_tokens = []
    enc_sentence_lengths = []
    dec_sentence_lengths = []
        
    for sline,tline in zip(source_lines,target_lines):
        s,es,oovs,sl = sent2idx(sline.strip())
        t,tl = target2idx(tline.strip(),oovs)
        input_batch_tokens.append(s)
        enc_sentence_lengths.append(sl)
        input_batch_extend_tokens.append(es)
        target_batch_tokens.append(t)
        input_oovs.append(oovs)
        dec_sentence_lengths.append(tl)
        if len(input_batch_tokens) == batch_size:
            oovs = max(len(x) for x in input_oovs)
            yield input_batch_tokens,enc_sentence_lengths,input_batch_extend_tokens,oovs,target_batch_tokens,dec_sentence_lengths
            input_batch_tokens = []
            input_batch_extend_tokens = []
            input_oovs = []
            target_batch_tokens = []
            enc_sentence_lengths = []
            dec_sentence_lengths = []


sess.run(tf.global_variables_initializer())
import random
print("start to run")
loss_history = []
for epoch in range(100):
    epoch_loss = 0
    i = 0
    for batch in get_batch(config.batch_size):
        input_batch_tokens,enc_sentence_lengths,input_batch_extend_tokens,oovs,target_batch_tokens,dec_sentence_lengths = batch
        #print(target_batch_tokens[0])
        batch_loss = model.train_one_batch(input_batch_tokens,enc_sentence_lengths,
                                                input_batch_extend_tokens, target_batch_tokens,
                                                dec_sentence_lengths)
        epoch_loss += batch_loss
        #print("Epoch-{0} batch-{1} batch loss-{2}".format(epoch,i+1,batch_loss))
        i += 1
    loss_history.append(epoch_loss)
    predictions,_,logits = model.eval_one_batch(input_batch_tokens,enc_sentence_lengths,
                                                       input_batch_extend_tokens, target_batch_tokens,
                                                       dec_sentence_lengths)
    print('Epoch', epoch)
    print('epoch loss: ', epoch_loss )
    idx = random.sample(range(len(input_batch_tokens)),1)[0]
    print("Input:", input_batch_tokens[idx])
    print("Input Extend:",input_batch_extend_tokens[idx])
    print("Prediction:",predictions[idx])
    print("Truth:",target_batch_tokens[idx])

