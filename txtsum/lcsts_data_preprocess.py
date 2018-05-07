import os
import re
import jieba
import statistics
from collections import Counter
from tqdm import tqdm
from utils.data_util import UNK, UNK_ID, SOS, SOS_ID, EOS, EOS_ID, read_vocab

def stats(lens):
    print("min:",min(lens))
    print("max:",max(lens))
    mean = statistics.mean(lens)
    print("mean:",mean)
    print("median:",statistics.median(lens))
    stddev = statistics.stdev(lens)
    print("stddev:",stddev)
    return mean,stddev

# You should download the LCSTS data by yourself. I have preprocessed the PART_I ,II, III data, and write to 
# files with format "summary \t text \t human label". For PART_I, all the human label is 0.
DATA_DIR = "/data/share/txtsum_cn/LCSTS/DATA"
OUT_DIR = "/data/xueyou/textsum/lcsts_0507"

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

def tokenize_and_vocab():
    # preprocess training data
    train_file = os.path.join(DATA_DIR,"part1.txt")
    train_source_file = os.path.join(OUT_DIR,"train.source")
    train_target_file = os.path.join(OUT_DIR,"train.target")

    word_cnt = Counter()

    source_lengths = []
    target_lengths = []

    print("process training data")
    with open(train_source_file,'w') as sf:
        with open(train_target_file,'w') as tf:
            for line in tqdm(open(train_file)):
                line = re.sub(" +"," ",line)
                tokens = line.strip().split("\t")
                if len(tokens) == 3:
                    target,source,_ = tokens
                    target = list(jieba.cut(target,HMM=False))
                    source = list(jieba.cut(source,HMM=False))

                    source_lengths.append(len(source))
                    target_lengths.append(len(target))

                    word_cnt.update(target + source)
                    sf.write(" ".join(source) + '\n')
                    tf.write(" ".join(target) + '\n') 

    print("info of source")
    stats(source_lengths)
    print("info of target")
    stats(target_lengths)

    '''
    info of source
    min: 21
    max: 125
    mean: 62.93010093352051
    median: 63.0
    stddev: 8.418641158758847
    info of target
    min: 1
    max: 29
    mean: 10.471895242419572
    median: 10.0
    stddev: 3.1334953897981994
    '''

    print("unique words:",len(word_cnt))
    print("Top 50K words as our vocab")
    MAX_CNT = 50000
    vocab_file = os.path.join(OUT_DIR,"vocab.txt")
    word2id = {SOS: SOS_ID, EOS: EOS_ID, UNK: UNK_ID}
    for w,_ in word_cnt.most_common():
        if len(word2id) >= MAX_CNT:
            break
        if w.strip():
            word2id[w] = len(word2id)

    id2word = {v:k for k,v in word2id.items()}

    with open(vocab_file,'w') as f:
        for i in range(len(id2word)):
            f.write(id2word[i] + "\n")

    def create_dev_or_test(fpath, name):
        source_file = os.path.join(OUT_DIR,"{0}.source".format(name))
        target_file = os.path.join(OUT_DIR,"{0}.target".format(name))

        with open(source_file,'w') as sf:
            with open(target_file,'w') as tf:
                for line in tqdm(open(fpath)):
                    tokens = line.strip().split("\t")
                    if len(tokens) == 3:
                        target,source,score = tokens
                        # using score >= 3 as test data
                        if int(score) >= 3:
                            target = list(jieba.cut(target))
                            source = list(jieba.cut(source))

                            sf.write(" ".join(source) + '\n')
                            tf.write(" ".join(target) + '\n') 

    create_dev_or_test(os.path.join(DATA_DIR,"part2.txt"),"dev")
    create_dev_or_test(os.path.join(DATA_DIR,"part3.txt"),"test")

def analysis_oovs(batch_size=64):
    '''
    batch size depends on your GPU memory
    '''
    train_source_file = os.path.join(OUT_DIR,"train.source")
    w2i,i2w = read_vocab(os.path.join(OUT_DIR,'vocab.txt'))
    batch = []
    oovs = Counter()
    batch_oovs = []
    for line in tqdm(open(train_source_file)):
        tokens = line.strip().split()
        batch.append(tokens)

        if len(batch) == batch_size:
            for sent in batch:
                oovs.update([t for t in sent if t not in w2i])

            batch_oovs.append(len(oovs))
            oovs = Counter()
            batch = []

    # ignore last batch, make no differences
    print("batch oov statistics")
    stats(batch_oovs)
    '''
    batch oov statistics
    min: 29
    max: 232
    mean: 87.14372550587859
    median: 86
    stddev: 18.613032434998438
    '''
    
if __name__ == "__main__":
    tokenize_and_vocab()
    analysis_oovs(64)
