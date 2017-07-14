# my_seq2seq
This is a project to learn to implement different s2s model on tensorflow.

## Structure

### Encoder

Use either:
- Multi-layer rnn
    - use the last state of the last layer rnn as the initial decode state
- Bi-direction rnn
    - use a Dense layer to convert the fw and bw state to the initial decode state

### Decoder

- Use multi-layer rnn, and set the inital state of each layer to initial decode state
- only apply attention to first layer of decoder

### Attention

- Bahdanau
- Luong

### Note

- Using tf-1.2.1
- ~~I also tried to run this code on tf1.2.rc1 on windows, but got some strange errors about gather_tree operation.~~

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

## TODO

What I will do next:

- [x] code refactoring, make the code more readable and easy to build new model
- [ ] test on more dataset
- [ ] how to add other features before attention layer
- [ ] how to add more layer after attention layer
- [ ] how to implement our own attention layer
- [ ] read the source code of tf.contrib.seq2seq

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