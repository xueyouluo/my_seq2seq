# Pointer-Generator

This is my implementation of pointer generator network based on tf.contrib.seq2seq.

You can refer to the original paper [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) to find out what is Pointer-Generator networks. For an intuitive overview of the paper, read the [blog post](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html).

The author has shared her implementation based on old seq2seq apis, you can find it [here](https://github.com/abisee/pointer-generator). I used the new seq2seq api, and test on tensorflow with version 1.4. I found some methods are changed in tf-1.7, so it won't work in tf-1.7.


### Copy

The copy implementation is almost the same as the orginal code, and I create a new Decoder class. At each decode step, we calculate the p_gen and the extended final vocab distribution.

When I calculate the loss, I got NANs. After a painful debugging, I found out that it is because we have paddings in the input tokens, we will got 0 probability at those positions. If we use them to calculate log(p), then we will get INF values, and apply maskes to them will get NAN values.


### Coverage

Since I want to use seq2seq apis, I have to read the source code of these apis and rewrite some classes. Thanks to the contributors of seq2seq, it's not difficult to add new features. I thought coverage should be easy to implement, but I was wrong. I need to rewrite the attention mechanism, the attention wrapper. 

Tensorflow is a static graph, so the code doesn't run in the order you write. The while loop makes me want to die. I want to use pytorch now.

I don't have time to test coverage right now. I only use the toy data to test the copy mechanism, which works.

### Beam search

Havn't got time to implement beam search. The beam search decoder is totally different from basic decoder. So I may not want to try to implement beam search in graph.
