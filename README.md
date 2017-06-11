# my_seq2seq
this is a project to create s2s of my own.

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

Note:
- Using tf.1.2.rc1, and using the newest attention_wrapper and beam_search_decoder code(2017/6/7) since the beam search decoder is not correct in this version, and they have fixed the bug in the master branch.
- I also tried to run this code on tf1.2.rc1 on windows, but got some strange errors about gather_tree operation.

##TODO

What I will do next:

- [ ] code refactoring, make the code more readable and easy to build new model
- [ ] test on more dataset

##Reference

Thanks to following resources:

- https://github.com/JayParks/tf-seq2seq
    - A good implementation of seq2seq with beam search based on tf 1.2.rc1
- https://github.com/j-min/tf_tutorial_plus
    - I used the demo data from here
- https://github.com/stanfordmlgroup/nlc
    - This project shows how to implement an attention wrapped rnn cell