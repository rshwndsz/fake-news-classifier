# Fake news classifier

## Abstract

A fake news detector based on the LIAR-PLUS dataset.

## Getting Started

Create a conda environment with python 3.6.8

```bash
conda create --name pytorch1.1 python=3.6.8
conda activate pytorch1.1
```
Install all dependencies using `conda env create -f environment.yml`.

```bash
cd fake-news-classifier
mkdir datasets/
```
Download the `LIAR_PLUS` dataset into `datasets/LIAR_PLUS` and the pretrained word vectors 
`wiki-news-300d-1M.vec` from [fasttext](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) 
into `datasets/word_embeddings`.

### Training

Train the model (for binary classification)

```bash
python main.py --phase train --binary yes
```

For hex classification use `--binary no`

### Testing

Similar to training, provide the phase and binary state options through the CLI.

```bash
python main.py --phase test --binary yes
```

## References

* [Where is your Evidence: Improving Fact-checking by Justification Modeling](https://aclweb.org/anthology/W18-5513)
* 2015, Conroy, Niall J., Victoria L. Rubin, and Yimin Chen.   
["Automatic deception detection: Methods for finding fake news."](https://onlinelibrary.wiley.com/doi/epdf/10.1002/pra2.2015.145052010082)  
Proceedings of the Association for Information Science and Technology 52, no. 1 (2015): 1-4.
* [Kaggle - Toxic Comments EDA by jagangupta](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda)
* [Kaggle - Spooky NLP Modelling by arthurtok](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)
* [Kaggle - Quora Insincere questions by ziliwang](https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm)
* [Awesome Fake News - GitHub](https://github.com/sumeetkr/AwesomeFakeNews)
