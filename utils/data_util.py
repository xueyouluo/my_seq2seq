import codecs
from collections import Counter

SOS = '<sos>'
SOS_ID = 0
EOS = '<eos>'
EOS_ID = 1
UNK = '<unk>'
UNK_ID = 2

def create_vocab(filename, delimiter=" ", min_count=0):
    """
    words should be split by delimiter, default is space
    """
    counter = Counter()
    with codecs.open(filename, encoding='utf8') as f:
        line = f.readline().strip()
        while line:
            tokens = line.split(delimiter)
            counter.update(tokens)
            line = f.readline().strip()
    words = counter.most_common()
    words = [word for word in words if word[1] >= min_count]
    word2id = {SOS:SOS_ID,EOS:EOS_ID,UNK:UNK_ID}
    for word, _ in words:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word, words
    


