# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from utils.data_util import (EOS, EOS_ID, SOS, SOS_ID, UNK, UNK_ID,
                             read_vocab, get_infer_iterator)

def tokeninze_sentence(sentence, src_w2i, src_i2w, src_max_len = None, reverse_source = False):
    sentence_ids = list(map(lambda x:src_w2i.get(x,UNK_ID), sentence.split(" ")))
    if src_max_len:
        sentence_ids = sentence_ids[:src_max_len]
    if reverse_source:
        sentence_ids = list(reversed(sentence_ids))
    sentence_ids = np.reshape(sentence_ids, [-1,len(sentence_ids)])
    sentence_length = np.reshape(sentence_ids.shape[1],[-1])
    return sentence_ids, sentence_length
    
def ids_to_sentences(sentence_ids, tgt_i2w):
    sentence = []
    for idx in sentence_ids:
        if idx == EOS_ID:
            break
        sentence.append(tgt_i2w.get(idx,UNK))
    return " ".join(sentence)

if __name__ == "__main__":

    checkpoint_dir = "/data/xueyou/fashion/fashion_s2s_0909/"
    data_dir = "/data/xueyou/fashion/0909/"

    vocab_file = os.path.join(data_dir, "vocab.0909.txt")
    
    config = pickle.load(
        open(os.path.join(checkpoint_dir, "config.pkl"), 'rb'))
    config.mode = "inference"
    config.max_inference_length = 60
    config.beam_size = 10

    with tf.Session() as sess:

        src_w2i, src_i2w = read_vocab(vocab_file)
        tgt_w2i, tgt_i2w = read_vocab(vocab_file)

        model = S2SModelWithPipeline(sess, None, config)
        model.init()
        model.restore_model()

        while True:
            raw = input("输入单词，空格分割(-1 to exit):")
            try:
                if int(raw) == -1:
                    break
            except:
                pass
            
            sentence_ids, sentence_length = tokeninze_sentence(raw,src_w2i,src_i2w,reverse_source = config.reverse_source)
            predictions = model.inference(sentence_ids,sentence_length)
            print("Predictions:")
            for p in predictions[0]:
                print(ids_to_sentences(p,tgt_i2w))

