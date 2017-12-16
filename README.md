# Seq2Seq models
This is a project to learn to implement different s2s model on tensorflow.

> This project is only used for learning, which means it will contain many bugs. I suggest to use nmt project to do experiments. You can find it in the reference part.

## Structure

A typical sequence to sequence(seq2seq) model contains an encoder, an decoder and an attetion structure. Tensorflow provide many useful apis to implement a seq2seq model, usually you will need belowing apis:
- [tf.contrib.rnn](https://tensorflow.google.cn/api_docs/python/tf/contrib/rnn)
    - Different RNNs
- [tf.contrib.seq2seq](https://tensorflow.google.cn/api_docs/python/tf/contrib/seq2seq)
    - Provided different attention mechanism and also a good implementation of beam search
- [tf.data](https://tensorflow.google.cn/api_docs/python/tf/data)
    - data preproces pipeline apis
- Other apis you need to build and train a model

### Encoder

Use either:
- Multi-layer rnn
    - use the last state of the last layer rnn as the initial decode state
- Bi-direction rnn
    - use a Dense layer to convert the fw and bw state to the initial decode state
- GNMT encoder
    - a bidirection rnn + serveral rnn with residual conection

### Decoder

- Use multi-layer rnn, and set the inital state of each layer to initial decode state
- GNMT decoder 
    - only apply attention to the bottom layer of decoder, so we can utilize multi gpus during training

### Attention

- Bahdanau
- Luong

### Metrics
Right now I only have cross entropy loss. Will add following metrics:
- bleu
    - for translation problems
- rouge
    - for summarization problems

### Dependency

- Using tf-1.3
- Python 3

## Run

Run the model on a toy dataset, ie. reverse the sequence

train:
```python
python -m bin.toy_train
```

inference:
```python
python -m bin.toy_inference
```

Also you can run on en-vi dataset, see en\_vietnam\_train.py in bin for more details

## TODO

What I will do next:

- [ ] implement the point-generator model, which shows promising results on summarization tasks
- [ ] read the source code of nmt
- [ ] read the source code of tf.contrib.seq2seq
- [ ] be able to implement different seq2seq structures based on tensorflow

## Reference

Thanks to following resources:

- https://github.com/tensorflow/nmt
    - google's NMT tutorial, very good resource to learn seq2seq
- https://github.com/JayParks/tf-seq2seq
    - A good implementation of seq2seq with beam search based on tf 1.2.rc1
- https://github.com/j-min/tf_tutorial_plus
    - I used the demo data from here
- https://github.com/stanfordmlgroup/nlc
    - This project shows how to implement an attention wrapped rnn cell