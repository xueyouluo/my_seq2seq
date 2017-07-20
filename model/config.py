
class BasicConfig(object):
    # Training settings
    mode = "train"
    num_gpus = 1    
    optimizer = "adam"
    learning_rate = 0.001
    learning_rate_decay = 0.95
    exponential_decay = False
    decay_steps = 10000 # only used when exponential_decay is True
    max_gradient_norm = 5.0

    # input data setting
    source_max_len = 50
    target_max_len = 50

    # vocab settings
    start_token = 0
    end_token = 1
    reverse_source = True

    # embedding setting
    share_vocab = False
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embedding_size = 256
    
    # settings for encoder, decoder and attention
    keep_prob = 0.80    
    num_units = 128
    
    # encoder
    encode_cell_type = 'gru'
    use_bidirection = False
    encode_layer_num = 2

    # decoder
    decode_cell_type = 'gru'
    decode_layer_num = 2

    attention_option = "bahdanau" # Bahdanau or Luong

    # inference
    # These settings are used for beam search
    # beam size = 1 will use greedy search
    beam_size = 5
    # input data batch size
    batch_size = 32
    max_inference_length = 10

    checkpoint_dir = "/tmp/basic_s2s/"

    def __repr__(self):
        arttribute = vars(self)
        arttribute = {k:v for k,v in arttribute.items() if not k.startswith("__")}
        return str(arttribute)
