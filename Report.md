# Report

## Highest Accuracy

* Binary Classification: 56.25
* Six-way Classification: 27.32

## How I achieved the results

I opted for a 300 dimension embedding layer having 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset followed by a 2 layer BiDirectional LSTM with 128 hidden dimensions and then a classifier with ouput dimensions equal to number of classes i.e. 6.
The model also uses batchnormalization and dropout layers to avoid overfitting.

## Different ideas I tried out

The first idea was to use a pretrained BERT network to process the text, but due to lack of computing resources I opted for a more conservative Bidirectional LSTM.
I also tried using GRU inplace of LSTM layers and did not find a huge difference in the accuracy, so I dropped the idea.

## Citations

### Libraries/Frameworks/Tools Used

* [Pytorch](https://pytorch.org/docs/master/)
* [torchtext](https://github.com/pytorch/text)
* [Scikit-learn metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
* Numpy
* Pandas
* Miniconda
* PyCharm IDE
* Jupyter Notebook

### Datasets

* [LIAR-PLUS by Alhindi, Tariq and Petridis, Savvas and Muresan, Smaranda](https://github.com/Tariq60/LIAR-PLUS)

### Other

* Pretrained Word Vectors by fasttext

### Papers Referred

* [Where is your Evidence: Improving Fact-checking by Justification Modeling by Alhindi, Tariq and Petridis, Savvas and Muresan, Smaranda](https://aclweb.org/anthology/W18-5513)
* [Universal Language Model Fine-tuning for Text Classification by Jeremy Howard & Sebastian Ruder](https://arxiv.org/pdf/1801.06146.pdf)
* Other papers in awesome-fake-news github repo
