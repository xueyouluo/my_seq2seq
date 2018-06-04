from utils.data_util import read_vocab,UNK_ID,EOS_ID,SOS_ID

def padding(batch, max_len):
    padded_tokens = []
    for tokens in batch:
        padded_tokens.append(tokens + [EOS_ID] * (max_len - len(tokens)))
    return padded_tokens

def sent2idx(sent, vocab, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

    oovs = []
    extend_tokens = []
    tokenized = []
    for token in tokens:
        if token not in vocab:
            tokenized.append(UNK_ID)
            if token not in oovs:
                oovs.append(token)
            extend_tokens.append(len(vocab) + oovs.index(token))
        else:
            extend_tokens.append(vocab[token])
            tokenized.append(vocab[token])
    return tokenized, extend_tokens, oovs, current_length

def target2idx(sent, oovs, vocab, max_sentence_length=15):
    tokens = sent.split()
    tokens = tokens[:max_sentence_length]
    current_length = len(tokens)

    tokenized = []
    for token in tokens:
        if token not in vocab:
            if token not in oovs:
                tokenized.append(UNK_ID)
            else:
                tokenized.append(len(vocab) + oovs.index(token))
        else:
            tokenized.append(vocab[token])
    return tokenized, current_length

def batch_preprocess(vocab,  sources, targets = None, max_source_len = 70, max_target_len = 15):
    source_tokens = []
    source_extend_tokens = []
    source_lengths = []
    target_tokens = []
    target_length = []
    source_oovs = []
    max_oovs = -1

    for i,sent in enumerate(sources):
        tokenized, extend_tokens, oovs, current_length = sent2idx(sent, vocab, max_source_len)
        source_tokens.append(tokenized)
        source_extend_tokens.append(extend_tokens)
        source_lengths.append(current_length)
        source_oovs.append(oovs)
        max_oovs = max(max_oovs, len(oovs))    

        if targets[0] is not None:
            tokenized, current_length = target2idx(targets[i],oovs,vocab,max_target_len)
            target_tokens.append(tokenized)
            target_length.append(current_length)

    source_tokens = padding(source_tokens, max(source_lengths))
    source_extend_tokens = padding(source_extend_tokens, max(source_lengths))
    target_tokens = padding(target_tokens,max(target_length))
    return source_tokens, source_lengths, source_extend_tokens, source_oovs, target_tokens, target_length, max_oovs

def get_batch(vocab, source_file, target_file=None, batch_size=64):
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
            lines.sort(key = lambda s:len(s[0].split()))
            for i in range(0,len(lines),batch_size):
                sources,targets = zip(*lines[i:i+batch_size])
                ret = batch_preprocess(vocab, sources, targets)
                buffer.append(ret)
            lines = []

        if len(buffer) == buffer_size:
            print("prefetched")
            for i in range(len(buffer)):
                yield buffer.pop()

    if len(lines) != 0:
        lines.sort(key = lambda s:len(s[0]))
        for i in range(0,len(lines),batch_size):
            sources,targets = zip(*lines[i:i+batch_size])
            ret = batch_preprocess(vocab, sources, targets)
            buffer.append(ret)

    for i in range(len(buffer)):
        yield buffer.pop()

def convert_ids_to_sentences(ids, vocab, oovs, join=True):
    tokens = []
    oovs_vocab = {len(vocab)+i:w for i,w in enumerate(oovs)}
    for i in ids:
        if i == EOS_ID:
            break
        if i in vocab:
            tokens.append(vocab[i])
        elif i in oovs_vocab:
            tokens.append(oovs_vocab[i])
        else:
            tokens.append(vocab[UNK_ID])
    if join:
        return "".join(tokens)
    else:
        return tokens