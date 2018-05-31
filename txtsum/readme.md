# Text Summarization

Since I have implemented some seq2seq models, I want to test them in real dataset. This folder is used to save all the data preprocessing, training scripts and evaluation results.

## Dataset

- LCSTS


## CopyNet

I have replaced one\_hot and some einsum code with scatter\_nd and gather\_nd, which reduced the memory and speed up the training. Also, 2 GPUs are faster than 1 GPU.

The training step time is reduced from 3~4 seconds to 0.8 seconds.