import codecs
import collections
import tensorflow as tf

from collections import Counter


UNK = '<unk>'
UNK_ID = 0
SOS = '<s>'
SOS_ID = 1
EOS = '</s>'
EOS_ID = 2


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
    pass


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
    word2id = {SOS: SOS_ID, EOS: EOS_ID, UNK: UNK_ID}
    for word, _ in words:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word, words


def read_vocab(vocab_file):
    """read vocab from file, one word per line
    """
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
      vocab_size = 0
      for word in f:
        vocab_size += 1
        vocab.append(word.strip())
    
    if vocab[0] != UNK or vocab[1] != SOS or vocab[2] != EOS:
        print("The first 3 vocab words [%s, %s, %s]"
                    " are not [%s, %s, %s]" %
                    (vocab[0], vocab[1], vocab[2], UNK, SOS, EOS))
        vocab = [UNK, SOS, EOS] + vocab
        vocab_size += 3
    
    word2id = {}
    for word in vocab:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id,id2word    

def get_infer_iterator(
        src_dataset, src_vocab_table, batch_size,
        source_reverse, eos, src_max_len=None):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
        src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
    if source_reverse:
        src_dataset = src_dataset.map(lambda src: tf.reverse(src, axis=[0]))
    # Add in the word counts.
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),     # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(src_eos_id,  # src
                            0))          # src_len -- unused

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None)


# copied from nmt
def get_train_iterator(
        src_dataset, tgt_dataset, src_vocab_table, tgt_vocab_table,
        batch_size,  sos, eos, source_reverse, random_seed, num_buckets,
        src_max_len=None, tgt_max_len=None, src_min_len=0, tgt_min_len=0,
        num_threads=4, output_buffer_size=None, skip_count=None):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(
        src_vocab_table.lookup(tf.constant(eos)),
        tf.int32)
    tgt_sos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(sos)),
        tf.int32)
    tgt_eos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(eos)),
        tf.int32)

    src_tgt_dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))

    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > src_min_len, tf.size(tgt) > tgt_min_len))
        
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    if source_reverse:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.reverse(src, axis=[0]), tgt),
            num_threads=num_threads,
            output_buffer_size=output_buffer_size)
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    # Add in the word counts.  Subtract one from the target to avoid counting
    # the target_input <eos> tag (resp. target_output <sos> tag).
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_threads=num_threads,
        output_buffer_size=output_buffer_size)
    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),      # src_len
                           tf.TensorShape([])),     # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(src_eos_id,  # src
                            tgt_eos_id,  # tgt_input
                            tgt_eos_id,  # tgt_output
                            0,           # src_len -- unused
                            0))          # tgt_len -- unused
    if num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(
                src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)
        batched_dataset = src_tgt_dataset.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size)
    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
        batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)
