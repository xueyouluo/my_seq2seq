import os
import random
import time

from collections import Counter
import tensorflow as tf

from model.copynet import CopyNet
from model.config import CopyNetConfig
from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from utils.model_util import get_config_proto

def get_oov_tokens(tokens, vocab, oovs):
    extend_tokens = []
    for t in tokens:
        if t in vocab:
            extend_tokens.append(vocab[t])
        elif t in oovs:
            extend_tokens.append(oovs[t])
        else:
            extend_tokens.append(UNK_ID)
    return extend_tokens

def padding(batch, max_len):
    padded_tokens = []
    for tokens in batch:
        padded_tokens.append(tokens + [EOS_ID] * (max_len - len(tokens)))
    return padded_tokens

def batch_preprocess(max_oovs, vocab,  sources, targets = None, max_source_len = 75, max_target_len = 15):
    source_tokens = []
    source_extend_tokens = []
    source_lengths = []
    target_tokens = []
    target_length = []

    source_oovs = Counter()
    batch_max_source_len = -1
    
    batch = []
    for sent in sources:
        tokens = sent.split()
        tokens = tokens[:max_source_len]
        batch.append(tokens)
        source_tokens.append([vocab.get(t,UNK_ID)  for t in tokens])
        s_len = len(tokens)
        source_lengths.append(s_len)
        batch_max_source_len = max(batch_max_source_len,s_len)
        source_oovs.update([t for t in tokens if t not in vocab])
    
    vocab_size = len(vocab)
    oovs = {}
    i = 0
    for w,_ in source_oovs.most_common()[:max_oovs]:
        oovs[w] = vocab_size + i
        i += 1

    source_extend_tokens = [get_oov_tokens(tokens,vocab,oovs) for tokens in batch]
    source_extend_tokens = padding(source_extend_tokens,batch_max_source_len)
    source_tokens = padding(source_tokens,batch_max_source_len)

    if targets[0] is not None:
        batch_max_target_len = -1
        for sent in targets:
            tokens = sent.split()
            tokens = tokens[:max_target_len]
            t_len = len(tokens)
            target_length.append(t_len)
            batch_max_target_len = max(t_len,batch_max_target_len)
            target_tokens.append(get_oov_tokens(tokens,vocab,oovs))

        target_tokens = padding(target_tokens,batch_max_target_len)
    return source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length, oovs

def get_batch(max_oovs, vocab, source_file, target_file=None, batch_size=64):
    sf = open(source_file)
    tf = open(target_file) if target_file else None
    lines = []
    buffer = []
    buffer_size = 1000
    while True:
        sline = sf.readline().strip()
        if not sline:
            break
        tline = tf.readline().strip() if tf else None
        lines.append((sline,tline))
        if len(lines) == batch_size * buffer_size:
            lines.sort(key = lambda s:s[0])
            for i in range(0,len(lines),batch_size):
                sources,targets = zip(*lines[i:i+batch_size])
                ret = batch_preprocess(max_oovs, vocab, sources, targets)
                buffer.append(ret)
            lines = []

        if len(buffer) == buffer_size:
            print("prefetched")
            for i in range(len(buffer)):
                yield buffer.pop()

    if len(lines) != 0:
        lines.sort(key = lambda s:s[0])
        for i in range(0,len(lines),batch_size):
            sources,targets = zip(*lines[i:i+batch_size])
            ret = batch_preprocess(max_oovs, vocab, sources, targets)
            buffer.append(ret)

    for i in range(len(buffer)):
        yield buffer.pop()
        

def convert_ids_to_sentences(ids, vocab, oovs):
    tokens = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in vocab:
            tokens.append(vocab[i])
        elif i in oovs:
            tokens.append(oovs[i])
        else:
            tokens.append(vocab[UNK_ID])
    return "".join(tokens)


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
config.encode_cell_type = 'gru'
config.decode_cell_type = 'gru'
config.batch_size = 128
config.checkpoint_dir = os.path.join(DATA_DIR,"copynet_new")
config.max_oovs = 200
config.num_gpus = 2
config.num_train_steps = 500000
# using Adam, not decay schema
config.decay_scheme = None 
config.src_vocab_file = os.path.join(DATA_DIR,"vocab.txt")

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
    step_pre_predict = 1000
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


