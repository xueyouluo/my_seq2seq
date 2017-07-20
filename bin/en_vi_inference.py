# -*- coding: utf-8 -*-
import os
import pickle

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from utils.data_util import (EOS, EOS_ID, SOS, SOS_ID, UNK, UNK_ID,
                             create_vocab, get_infer_iterator)

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
        src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=UNK_ID)
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=UNK_ID)

        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=UNK)

        infer_src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        infer_batch_size_placeholder = tf.constant(1, dtype=tf.int64)

        infer_src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            infer_src_placeholder)
        infer_iterator = get_infer_iterator(
            infer_src_dataset,
            src_vocab_table,
            batch_size=infer_batch_size_placeholder,
            eos=EOS,
            source_reverse=config.reverse_source,
            src_max_len=config.source_max_len)

        model = S2SModelWithPipeline(sess, infer_iterator, config)
        prediction_tokens = reverse_tgt_vocab_table.lookup(
            tf.to_int64(model.beam_predictions))
        model.init()
        model.restore_model()

        while True:
            raw = raw_input("Enter sentence(-1 to exit):")
            try:
                if int(raw) == -1:
                    break
            except:
                pass

            feed_dict = {infer_src_placeholder: [raw]}
            # this is hard for exporting for serving, so the better choice is use placeholders instead of infer iterator
            sess.run(infer_iterator.initializer, feed_dict=feed_dict)

            predictions = sess.run(prediction_tokens)
            print("Predictions:")
            # showing wierd charactors on my screen, but showing correctly when I copied them here.
            for p in predictions[0]:
                print(" ".join(p))

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
