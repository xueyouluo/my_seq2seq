# Seq2Seq models
This is a project to learn to implement different s2s model on tensorflow.

> This project is only used for learning, which means it will contain many bugs. I suggest to use nmt project to do experiments and train seq2seq models. You can find it in the reference part.

## Models

The models I have implemented are as following:

- Basic seq2seq model 
    - A model with bi-direction RNN encdoer and attention mechanism
- Seq2seq model 
    - Same as basic model, but using tf.data pipeline to process input data
- GNMT model
    - Residual conection and attention same as GNMT model to speed up training
    - refer to [GNMT](https://arxiv.org/abs/1609.08144) for more details
- Pointer-Generator model
    - A model that support copy mechanism
    - refer to [Pointer-Generator](https://arxiv.org/abs/1704.04368) for more details
- CopyNet model
    - A model also support copy mechanism
    - refer to [CopyNet](https://arxiv.org/abs/1603.06393) for more details.

For the implement details, refer to [ReadMe](./model/readme.md) in the model folder.


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

- Using tf-1.4
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

Also you can run on en-vi dataset, refer to en\_vietnam\_train.py in bin for more details.

You can find more training scripts in bin directory.

## Reference

Thanks to following resources:

- https://github.com/tensorflow/nmt
    - google's NMT tutorial, very good resource to learn seq2seq
- https://github.com/OpenNMT/OpenNMT-tf
    - code from harvardnlp group, also a good resource to learn seq2seq. Good code style and structure. You can find tensor2tensor implementation details here, which is becoming more and more popular nowdays.
- https://github.com/JayParks/tf-seq2seq
    - A good implementation of seq2seq with beam search based on tf 1.2.rc1
- https://github.com/j-min/tf_tutorial_plus
    - I used the demo data from here
- https://github.com/vahidk/EffectiveTensorflow
    - how to use tensorflow effectivly
- https://github.com/abisee/pointer-generator
    - The original pointer-generator network that use old seq2seq apis
- https://github.com/stanfordmlgroup/nlc
    - This project shows how to implement an attention wrapped rnn cell
- https://github.com/lspvic/CopyNet
    - this project using nmt to implement copynet
