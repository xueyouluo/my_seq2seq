from model.basic_s2s_model import BasicS2SModel
from model.config import BasicConfig
import tensorflow as tf
import numpy as np
import pickle

if __name__ == "__main__":
    with tf.Session() as sess:
        config = pickle.load(open("/tmp/basic_s2s/config.pkl",'rb'))
        config.mode = "inference"
        config.batch_size = 1
        
        model = BasicS2SModel(sess,config)
        model.restore_model()

        while True:
            # since this is a revese sequence, I leave out the tokeninze step, use the word id instead
            raw = raw_input("Enter list of word ids (id should be less than 23), -1 to exit:")
            try:
                if int(raw) == -1:
                    break
            except:
                pass
            
            try:
                ids = map(int,raw.split(" "))
            except:
                continue

            ids = np.reshape(ids,[-1,len(ids)])
            length = np.reshape(ids.shape[1],[-1])
            predict = model.inference(ids,length)
            print("Prediction:")
            print(predict)