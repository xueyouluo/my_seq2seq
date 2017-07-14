from model.basic_s2s_model import BasicS2SModel
from model.config import BasicConfig
from utils.data_util import create_vocab,UNK_ID,SOS_ID,EOS_ID
import os
import tensorflow as tf
import time 
import numpy as np
import pickle

if __name__ == "__main__":
    # use toy data to train a model
    DATA_DIR = "toy_data/"

    # 1. get vocab
    print("create vocab")
    train_source_file = os.path.join(DATA_DIR,"train","sources.txt")
    w2i,i2w,_ = create_vocab(train_source_file)

    # 2. tokenize 
    # TODO - according to nmt, we can use dataset iterators to process data, will add it next time
    data = {}
    lengths = {}
    length = 0
    cnt = 0
    print("tokenize data")
    for d in ['train','test']:
        for f in ['sources','targets']:
            filename_base = os.path.join(DATA_DIR,d,f)
            raw_filename = filename_base + ".txt"
            key = d + f
            data[key] = []
            lengths[key] = []
            for line in open(raw_filename):
                tokenize_list = map(lambda x:w2i.get(x,UNK_ID),line.strip().split(" "))
                length += len(tokenize_list)
                cnt += 1
                data[key].append(tokenize_list)
                lengths[key].append(len(tokenize_list))
    
    avg_length = length / cnt
    max_len = 2 * int(avg_length)

    # pad data
    print("pad data")
    for key in data:
        unpad = data[key]
        for i,tokenize_list in enumerate(unpad):
            if len(tokenize_list) > max_len:
                unpad[i] = tokenize_list[:max_len]
            else:
                unpad[i] = tokenize_list + [EOS_ID] * (max_len - len(tokenize_list))

    def get_batch_data(i,batch_size,mode="train"):
        assert mode in ['train','test']
        batch_sources = np.asarray(data[mode+"sources"][i*batch_size:(i+1)*batch_size])
        batch_sources_len = np.asarray(lengths[mode+"sources"][i*batch_size:(i+1)*batch_size])
        batch_targets = np.asarray(data[mode+'targets'][i*batch_size:(i+1)*batch_size])
        batch_targets_len = np.asarray(lengths[mode+"targets"][i*batch_size:(i+1)*batch_size])
        return batch_sources,batch_sources_len,batch_targets,batch_targets_len

    # 3. build model & train
    with tf.Session() as sess:
        config = BasicConfig()
        config.start_token = SOS_ID
        config.end_token = EOS_ID
        config.vocab_size = len(w2i)
        config.exponential_decay = True
        config.embedding_size = 100
        pickle.dump(config,open(config.checkpoint_dir + "config.pkl",'wb'))
        
        print("build model")
        model = BasicS2SModel(sess,config)
        model.init()

        def eval():
            # eval
            eval_loss = 0.0
            test_size = int(len(data['test' + 'sources']) / config.batch_size)
            for i in range(test_size):
                batch_x,batch_x_len,batch_y,batch_y_len = get_batch_data(i,config.batch_size,'test')
                prediction, loss= model.eval_one_batch(batch_x,batch_x_len, batch_y,batch_y_len)
                eval_loss += loss
            eval_loss /= test_size
            return eval_loss,batch_x,batch_y,prediction

        batches = int(len(data['train' + 'sources']) / config.batch_size)
        avg_loss = 0.0
        print_per_step = 20
        print("ready to train")
        for i in range(10):
            for j in range(batches):
                batch_sources,batch_sources_len,batch_targets,batch_targets_len = get_batch_data(j,config.batch_size)
                start_time = time.time()
                loss = model.train_one_batch(batch_sources,batch_sources_len,batch_targets,batch_targets_len)
                end_time = time.time()
                avg_loss += loss
                if (j+1)%print_per_step == 0:
                    print("epoch {0}, step {1} of {2}, avg loss = {3}, {4} batches/sec".format(i+1,j+1,batches,avg_loss/print_per_step,1.0/(end_time-start_time)))
                    avg_loss = 0.0
            
            eval_loss,x,y,prediction = eval()
            print("Epoch {0} eval loss {1}".format(i+1,eval_loss))
            print("sample:")
            for i in range(5):
                print("Input: {0}".format(x[i]))
                print("Predict: {0}".format(prediction[i]))
                print("Truth: {0}".format(y[i]))

            model.save_model()
