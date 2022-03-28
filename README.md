# Deep Learning  Models for Chinese Sentiment Analysis


## Introduction
Effective representation of a text is critical for various natural language processing tasks. For the particular task of Chinese sentiment analysis,
it is important to understand and choose an effective representation of a text from different forms of Chinese representations such as word, character and pinyin.
As illustrated in the paper of "Exploiting Effective Representations for Chinese Sentiment Analysis Using a Multi-Channel Convolutional Neural Network",
the character representation is efficient in terms of small vocabulary size and requires the least pre-processing effort without the need for Chinese word segmentation.
This project implements the baseline models, including CNN, LSTM, self-attention and BERT for Chinese sentiment analysis based on the character representation.

## Dataset
This repository releases the dataset crawled from Amazon reviews, where each review has a star (1-5) rated by a customer.
To evaluate Chinese sentiment analysis, each review was labeled as positive (star 4, 5), negative (star 1, 2) and neutral (star 3),
and was tokenized into characters for feeding the models.

|       | #positive | #neutral | #negative |
|-------|-----------|----------|-----------|
| train | 46982     | 4272     | 4854      |
| test  | 11710     | 1084     | 1233      |

## Installation
Please setup a conda environment and install the required packages.
```
pip install -r requirements.txt
```

## Experiments
To evaluate the models, the experiments can be run in batch as follows:
```
bash batch.sh
```
### Experimental results
|               | **Accuracy**   |
| ------------- | -------------- |
| **CNN**       | 87.03%         |
| **CNN+SA**    | 87.20%         |
| **LSTM**      | 87.37%         |
| **LSTM+SA**   | 87.17%         |
| **BERT**      | 88.69%         |

## References:
For details of the implemented models, please refer to the following papers:

<pre>
@article{liu2018exploiting,
  title={Exploiting effective representations for chinese sentiment analysis using a multi-channel convolutional neural network},
  author={Liu, Pengfei and Zhang, Ji and Leung, Cane Wing-Ki and He, Chao and Griffiths, Thomas L},
  journal={arXiv preprint arXiv:1808.02961},
  year={2018}
}

@inproceedings{kim-2014-convolutional,
    title = "Convolutional Neural Networks for Sentence Classification",
    author = "Kim, Yoon",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D14-1181",
    doi = "10.3115/v1/D14-1181",
    pages = "1746--1751",
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
</pre>


## Report
Please feel free to create an issue or send emails to the author at ppfliu@gmail.com.
