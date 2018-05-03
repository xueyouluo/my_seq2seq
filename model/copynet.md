# CopyNet

A tensorflow implementation of CopyNet based on tf.contrib.seq2seq APIs.

Refer to [Incorporating Copying Mechanism in Sequence-to-Sequence Learning
](https://arxiv.org/abs/1603.06393) for the details of copynet.

Also, this implementation is based on the code of [CopyNet Implementation with Tensorflow and nmt
](https://github.com/lspvic/CopyNet), I borrowed the ideas from here.

## Implement Details

The main idea here is create a new RNN cell wrapper(CopyNetWrapper), which is used to cacluate the copy and generate probability, and update the state.

Since we don't need the attention information, we don't need to change the decoder class. Everything can be done in a RNN cell.

### OOV words

This part is different from point-generator. In PG, we use a placeholder to feed the max number of oov words(refer to max\_oov later) in a batch. But we can set the max\_oov in a batch in advance, say 100. In this case, the source input extend tokens should only have ids in range 0 to (vocab size + 100), other oov words should set to UNK. You can analysis the training data to get a proper max\_oov.

### Copy and Generate

Using the formula in the paper to get the probability of the two modes and get the final vocab distribution. Then we use this to calculate p(y_t,c|.).

### Selective Read

In selective read, we need get $y_{t-1}$