# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

import tensorflow as tf

from model.s2s_model_with_data_pipeline import S2SModelWithPipeline
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.session_bundle import exporter

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("export", None, "export path")
tf.flags.DEFINE_integer("inference_length", 100, "max inference length")
FLAGS = tf.flags.FLAGS


def main(_argv):
    """Program entry point.
    """
    if not FLAGS.model_dir:
        raise Exception("you must provide model directory")

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)) as sess:
        # read config from pickle
        config = pickle.load(
            open(os.path.join(FLAGS.model_dir, "config.pkl"), 'rb'))
        config.mode = "inference"
        config.max_inference_length = FLAGS.inference_length
        config.beam_size = 10

        model = S2SModelWithPipeline(sess, None, config)
        model.init()
        model.restore_model()

        export_path = FLAGS.export
        builder = saved_model_builder.SavedModelBuilder(export_path)
        prediction_input_tokens = utils.build_tensor_info(model.source_tokens)
        prediction_input_length = utils.build_tensor_info(model.source_length)
        prediction_preditons = utils.build_tensor_info(model.beam_predictions)
        prediction_signature = signature_def_utils.build_signature_def(
            inputs={'query': prediction_input_tokens, 'query_length': prediction_input_length},
            outputs={"predictions": prediction_preditons},
            method_name=signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                'predict':
                prediction_signature,
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
            },
            legacy_init_op=legacy_init_op,
            clear_devices=True)

        builder.save()

        print('Done exporting!')


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
