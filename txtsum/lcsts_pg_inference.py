import os
import random
import tensorflow as tf
from model.pointer_generator import PointerGeneratorModel
from model.config import PointGeneratorConfig
from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from utils.model_util import get_config_proto
from txtsum.pg_data_utils import get_batch,convert_ids_to_sentences

DATA_DIR = "/data/xueyou/textsum/lcsts_0507"
w2i,i2w = read_vocab(os.path.join(DATA_DIR,'vocab.txt'))
train_source_file = os.path.join(DATA_DIR,"test.source")
train_target_file = os.path.join(DATA_DIR,"test.target")
output_file = os.path.join(DATA_DIR,"test.predict")

config = PointGeneratorConfig()
config.mode = 'inference'
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
config.batch_size = 1
config.checkpoint_dir = os.path.join(DATA_DIR,"pointer_generator_lstm_pretrain_embed_0529",'best_rouge')
config.coverage = False

with tf.Session(config=get_config_proto(log_device_placement=False)) as sess:
    model = PointerGeneratorModel(sess, config)

    #if config.coverage:
    #    model.convert_to_coverage_model()

    try:
        model.restore_model()
        print("restore model successfully")
    except Exception as e:
        print(e)
        print("fail to load model")

    with open(output_file, 'w') as f:
        for batch in get_batch(w2i, train_source_file, train_target_file, config.batch_size):
            source_tokens, source_lengths, source_extend_tokens, source_oovs, target_tokens, target_length, max_oovs= batch
            predictions = model.inference(source_tokens, source_lengths, max_oovs, source_extend_tokens)[0]
            idx = random.sample(range(len(source_tokens)),1)[0]
            oovs = source_oovs[idx]
            print("Input:", convert_ids_to_sentences(source_tokens[idx],i2w,oovs))
            print("Input Extend:", convert_ids_to_sentences(source_extend_tokens[idx],i2w,oovs))
            print("Prediction:",convert_ids_to_sentences(predictions[idx],i2w,oovs))
            print("Truth:", convert_ids_to_sentences(target_tokens[idx],i2w,oovs))

            for i in range(len(source_tokens)):
                predict = convert_ids_to_sentences(predictions[idx],i2w,oovs)
                f.write(predict + '\n')




