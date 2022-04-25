# Transformer-Models-from-Scratch
This repository contains various transformer models that I implemented from scratch when I started to learn Machine Learning. These models include:

1. encoder-only transformer model for text classification:
    - [Encoder_only_transformer_AG_News.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Encoder_only_transformer_AG_News_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Encoder_only_transformer_AG_News_classification.ipynb) 

2. decoder-only transformer model (GPT-like) trained for doing n-digit addition 
    - [GPT_Addition.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/GPT_Addition.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/GPT_Addition.ipynb)

3. full transformer model (encoder + decoder) for machine translation 
    - [Transformer_Multi30k_German_to_English.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Multi30k_German_to_English.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Multi30k_German_to_English.ipynb) trained a transformer model of about 26 million parameters on the Multi30k dataset, and achieved a BLEU score of 34.9.
    - [Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_Chinese_To_English_Translation_news-commentary-v16.ipynb) trained a transformer with about 90 million parameters on the news-commentary-v16 dataset. The main purpose of this notebook is to study how the performance of the model (test loss and BLEU score) changes as training set size increases. The result is shown in the plots at the end of this notebook.

4. and more to be added...

## Notes
[Transformer_details.pdf](https://github.com/hbchen-one/Transformer-Models-from-Scratch/blob/main/Transformer_details.pdf) contains some details of the transformer model that I
found a little bit confusing when I first tried to implement it from
scratch.

## References
