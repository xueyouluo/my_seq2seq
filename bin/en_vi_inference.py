# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from utils.data_util import (EOS, EOS_ID, SOS, SOS_ID, UNK, UNK_ID,
                             read_vocab, get_infer_iterator)

def tokeninze_sentence(sentence, src_w2i, src_max_len = None, reverse_source = False):
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

    checkpoint_dir = "/tmp/envi_nmt/"
    data_dir = "/tmp/nmt_data"

    src_vocab_file = os.path.join(data_dir, "vocab.en")
    tgt_vocab_file = os.path.join(data_dir, "vocab.vi")

    config = pickle.load(
        open(os.path.join(checkpoint_dir, "config.pkl"), 'rb'))
    config.mode = "inference"
    config.max_inference_length = 20
    with tf.Session() as sess:

        src_w2i, src_i2w = read_vocab(src_vocab_file)
        tgt_w2i, tgt_i2w = read_vocab(tgt_vocab_file)

        model = S2SModelWithPipeline(sess, None, config)
        model.init()
        model.restore_model()

        while True:
            raw = raw_input("Enter sentence(-1 to exit):")
            try:
                if int(raw) == -1:
                    break
            except:
                pass
            
            sentence_ids, sentence_length = tokeninze_sentence(raw,src_w2i,reverse_source = config.reverse_source)
            predictions = model.inference(sentence_ids,sentence_length)
            print("Predictions:")
            # showing wierd charactors on my screen, but showing correctly when I copied them here.
            for p in predictions[0]:
                print(ids_to_sentences(p,tgt_i2w))

            # examples:
            # some translations are ok. The repeat problem is a common problem of s2s.
            # 1. this is a nice place
            #   nơi xinh đẹp đó là một nơi rất hay nơi đây là một nơi tuyệt vời .
            #   nơi xinh đẹp đó là một nơi rất hay nơi đây là một nơi rất hay nơi đây là một
            # 2. My grandmother never let me forget his life .
            #   Bà tôi không bao giờ để tôi quên mất cuộc đời mình .
            #   Bà tôi không bao giờ để tôi quên cuộc đời mình . 
            # 3. I didn &apos;t know how to talk about anything .
            #   Tôi không biết làm thế nào để nói về điều gì đó .
            #   Tôi không biết làm thế nào để nói về bất cứ thứ gì . 
