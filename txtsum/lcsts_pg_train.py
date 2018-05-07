import os
import random
import time

from collections import Counter
import tensorflow as tf

from model.pointer_generator import PointerGeneratorModel
from model.config import PointGeneratorConfig
from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from utils.model_util import get_config_proto

def padding(batch, max_len):
    padded_tokens = []
    for tokens in batch:
        padded_tokens.append(tokens + [EOS_ID] * (max_len - len(tokens)))
    return padded_tokens

def sent2idx(sent, vocab, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

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
    return tokenized, extend_tokens, oovs, current_length

def target2idx(sent, oovs, vocab, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

    tokenized = []
    for token in tokens:
        if token not in vocab:
            if token not in oovs:
                tokenized.append(UNK_ID)
            else:
                tokenized.append(len(vocab) + oovs.index(token))
        else:
            tokenized.append(vocab[token])
    return tokenized, current_length

def batch_preprocess(vocab,  sources, targets = None, max_source_len = 75, max_target_len = 15):
    source_tokens = []
    source_extend_tokens = []
    source_lengths = []
    target_tokens = []
    target_length = []
    source_oovs = []
    max_oovs = -1

    for i,sent in enumerate(sources):
        tokenized, extend_tokens, oovs, current_length = sent2idx(sent, vocab, max_source_len)
        source_tokens.append(tokenized)
        source_extend_tokens.append(extend_tokens)
        source_lengths.append(current_length)
        source_oovs.append(oovs)
        max_oovs = max(max_oovs, len(oovs))    

        if targets[0] is not None:
            tokenized, current_length = target2idx(targets[i],oovs,vocab,max_target_len)
            target_tokens.append(tokenized)
            target_length.append(current_length)

    source_tokens = padding(source_tokens, max(source_lengths))
    source_extend_tokens = padding(source_extend_tokens, max(source_lengths))
    target_tokens = padding(target_tokens,max(target_length))
    return source_tokens, source_lengths, source_extend_tokens, source_oovs, target_tokens, target_length, max_oovs

def get_batch(vocab, source_file, target_file=None, batch_size=64):
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
                ret = batch_preprocess(vocab, sources, targets)
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
            ret = batch_preprocess(vocab, sources, targets)
            buffer.append(ret)

    for i in range(len(buffer)):
        yield buffer.pop()

def convert_ids_to_sentences(ids, vocab, oovs):
    tokens = []
    oovs_vocab = {len(vocab)+i:w for i,w in enumerate(oovs)}
    for i in ids:
        if i == EOS_ID:
            break
        if i in vocab:
            tokens.append(vocab[i])
        elif i in oovs_vocab:
            tokens.append(oovs_vocab[i])
        else:
            tokens.append(vocab[UNK_ID])
    return "".join(tokens)


DATA_DIR = "/data/xueyou/textsum/lcsts"
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
config.decode_layer_num = 4
config.num_units = 512
config.embedding_size = 256
config.encode_cell_type = 'gru'
config.decode_cell_type = 'gru'
config.batch_size = 128
config.checkpoint_dir = os.path.join(DATA_DIR,"pointer_generator")
config.num_gpus = 2
config.num_train_steps = 500000
config.optimizer = 'sgd'
config.learning_rate = 1.0
config.decay_scheme = 'luong10'
config.src_vocab_file = os.path.join(DATA_DIR,"vocab.txt")

with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
    model = PointerGeneratorModel(sess, config)
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
    step_per_show = 100
    step_per_predict = 1000
    step_per_save = 10000
    global_step = model.global_step.eval(session=sess)
    while global_step < config.num_train_steps:         
        for batch in get_batch(w2i, train_source_file, train_target_file, config.batch_size):
            step += 1
            source_tokens, source_lengths, source_extend_tokens, source_oovs, target_tokens, target_length, max_oovs= batch
            start = time.time()
            batch_loss, global_step = model.train_one_batch(source_tokens, source_lengths, max_oovs, source_extend_tokens, target_tokens, target_length)
            end = time.time()
            losses += batch_loss
            step_time += (end-start)
            if step % step_per_show == 0:
                print("Epoch {0}, step {1}, loss {2}, step-time {3}".format(epoch + 1, global_step, losses/step_per_show, step_time/step_per_show))
                losses = 0.0
                step_time = 0.0

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


