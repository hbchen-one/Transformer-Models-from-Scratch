# Transformer-Models-from-Scratch
This repository contains various transformer models that I implemented from scratch (using PyTorch) when I started to learn Machine Learning. These models include:

1. encoder-only transformer model for text classification:
    - [Encoder_only_transformer_AG_News_classification.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Encoder_only_transformer_AG_News_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Encoder_only_transformer_AG_News_classification.ipynb)
        - This notebook trained a simple encoder-only transformer model for doing text classification on the AG News dataset. It easily achieves a accuracy of about 91.9%.

2. decoder-only transformer model (GPT-like) trained for doing n-digit addition 
    - [GPT_Addition.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/GPT_Addition.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/GPT_Addition.ipynb) 
        - The same model (with only about 0.28 million parameters) is trained on 2-digit, 5-digit, 10-digit and 18-digit additions separately, and it got all the 2-digit addition right, and only a very small fraction of the higher digit additions wrong (test accuracy for 18-digit is about 96.6%). 
        - The wrong answers that the model gave are mostly off by one or two digits.
     
3. full transformer model (encoder + decoder) for machine translation 
    - [Transformer_Multi30k_German_to_English.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Multi30k_German_to_English.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Multi30k_German_to_English.ipynb)
        - This notebook trained a transformer model of about 26 million parameters on the Multi30k dataset, and achieved a BLEU score of 35.5 on the test set. This BLUE score seems high, which I think one reason is that the sentences in this dataset are relatively simple. 
    - [Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb) 
        -   This notebook trained a transformer with about 90 million parameters on the news-commentary-v16 dataset. The main purpose of this notebook is to study how the performance of the model (test loss and BLEU score) changes as training set size increases. The result is shown in the plots at the end of this notebook.

## Notes
[Transformer_details.pdf](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_details.pdf) ([HTML version](https://htmlpreview.github.io/?https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_details.html)) contains some details of the transformer model that I
found a little bit confusing when I first tried to implement it from
scratch. 
## References
1. the Attention Is All You Need paper [arXiv:1706.03762](https://arxiv.org/pdf/1706.03762.pdf)
2. [The Annotated Transformer by Alexander Rush](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
4. Andrej Karpathy's minGPT Github repository: [karpathy/minGPT](https://github.com/karpathy/minGPT)

