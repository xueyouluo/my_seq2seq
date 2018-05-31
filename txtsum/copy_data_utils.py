from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID
from collections import Counter


def get_oov_tokens(tokens, vocab, oovs):
    extend_tokens = []
    for t in tokens:
        if t in vocab:
            extend_tokens.append(vocab[t])
        elif t in oovs:
            extend_tokens.append(oovs[t])
        else:
            extend_tokens.append(UNK_ID)
    return extend_tokens

def padding(batch, max_len):
    padded_tokens = []
    for tokens in batch:
        padded_tokens.append(tokens + [EOS_ID] * (max_len - len(tokens)))
    return padded_tokens

def batch_preprocess(max_oovs, vocab,  sources, targets = None, max_source_len = 75, max_target_len = 15):
    source_tokens = []
    source_extend_tokens = []
    source_lengths = []
    target_tokens = []
    target_length = []

    source_oovs = Counter()
    batch_max_source_len = -1
    
    batch = []
    for sent in sources:
        tokens = sent.split()
        tokens = tokens[:max_source_len]
        batch.append(tokens)
        source_tokens.append([vocab.get(t,UNK_ID)  for t in tokens])
        s_len = len(tokens)
        source_lengths.append(s_len)
        batch_max_source_len = max(batch_max_source_len,s_len)
        source_oovs.update([t for t in tokens if t not in vocab])
    
    vocab_size = len(vocab)
    oovs = {}
    i = 0
    for w,_ in source_oovs.most_common()[:max_oovs]:
        oovs[w] = vocab_size + i
        i += 1

    source_extend_tokens = [get_oov_tokens(tokens,vocab,oovs) for tokens in batch]
    source_extend_tokens = padding(source_extend_tokens,batch_max_source_len)
    source_tokens = padding(source_tokens,batch_max_source_len)

    if targets[0] is not None:
        batch_max_target_len = -1
        for sent in targets:
            tokens = sent.split()
            tokens = tokens[:max_target_len]
            t_len = len(tokens)
            target_length.append(t_len)
            batch_max_target_len = max(t_len,batch_max_target_len)
            target_tokens.append(get_oov_tokens(tokens,vocab,oovs))

        target_tokens = padding(target_tokens,batch_max_target_len)
    return source_tokens, source_lengths, source_extend_tokens, target_tokens, target_length, oovs

def get_batch(max_oovs, vocab, source_file, target_file=None, batch_size=64):
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
                ret = batch_preprocess(max_oovs, vocab, sources, targets)
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
            ret = batch_preprocess(max_oovs, vocab, sources, targets)
            buffer.append(ret)

    for i in range(len(buffer)):
        yield buffer.pop()
        

def convert_ids_to_sentences(ids, vocab, oovs):
    tokens = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in vocab:
            tokens.append(vocab[i])
        elif i in oovs:
            tokens.append(oovs[i])
        else:
            tokens.append(vocab[UNK_ID])
    return "".join(tokens)