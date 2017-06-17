
class BasicConfig(object):

    mode = "train"

    optimizer = "adam"
    learning_rate = 0.001
    learning_rate_decay = 0.95
    exponential_decay = False
    # only used when exponential_decay is True
    decay_steps = 10000

    max_gradient_norm = 5.0
    keep_prob = 0.75

    vocab_size = 1000
    embedding_size = 256

    encode_cell_type = 'gru'
    encode_cell_size = 256
    use_bidirection = False
    encode_layer_num = 2

    decode_cell_type = 'gru'
    decode_cell_size = 256
    decode_layer_num = 2

    # These settings are used for beam search
    # beam size = 1 will use greedy search
    beam_size = 5
    # only useful for beam search
    batch_size = 32
    max_source_len = 10
    max_inference_lenght = 10

    checkpoint_dir = "/tmp/basic_s2s/"