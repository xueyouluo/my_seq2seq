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

This part is different from point-generator. In PG, we use a placeholder to feed the max number of oov words(use max\_oov for short later) in a batch. But we can set the max\_oov in a batch in advance, say 100. In this case, the source input extend tokens should only have ids in range 0 to (vocab size + 100), other oov words should set to UNK. You can analysis the training data to get a proper max\_oov.

### Copy and Generate

Using the formula in the paper to get the probability of the two modes and get the final vocab distribution. Then we use this to calculate $p(y_t,c|.)$.

### Selective Read

In selective read, we need get $y_{t-1}$. I use argmax(final_distribution) to get the last ids, which is a greeding way, not sure if it works with beam search.

Actually,  $S(y_{t-1})$ is the average sum of encoder outputs where $x_t = y_{t-1}$:

$S(y_{t-1}) = \sum{p_{ti}*h_i}$

$p_{ti} = 1/K * p(x_i,c|.) \space if \space x_i==y_{t-1}  \space else \space 0$

Since $p(x_i,c|.)$ is same for the same $x$, so:

$S(y_{t-1}) = \sum_{x_{ti} == y_{t-1}} (1/ K * h_i)$

And K equals to the total number of input tokens which is same as $y_{t-1}$.

### Tricks

I found tf.einsum is a very useful api to manipulate matrixes, you can find how many lines of codes saved in CopyNet to get the attention distribution compared with PG. Everytime you need to manipulate the matrixes, check if einsum works for you.

You can refer to [A basic introduction to NumPy's einsum](http://ajcr.net/Basic-guide-to-einsum/) for a basic introduction of einsum.

## Test

I test the code using the toy data, check the test_copynet.py in the bin folder for details. After about 20 epochs, the results  is almost the same as the truth.


