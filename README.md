# thai2fit (formerly thai2vec)
ULMFit Language Modeling, Text Feature Extraction and Text Classification in Thai Language.
Created as part of [pyThaiNLP](https://github.com/PyThaiNLP/) with [ULMFit](https://arxiv.org/abs/1801.06146) implementation from [fast.ai](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

Models and word embeddings can also be downloaded via [Dropbox](https://www.dropbox.com/sh/lgd8wf5h0eoehzr/AACD0ZnpOiMKQq1N94WmfV-Va?dl=1).

We pretrained a language model with 60,005 embeddings on [Thai Wikipedia Dump](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) (perplexity of 28.71067) and text classification (micro-averaged F-1 score of 0.60322 on 5-label classification problem. Benchmarked to 0.5109 by [fastText](fasttext.cc) and 0.4976 by LinearSVC on [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction). The language model can also be used to extract text features for other downstream tasks.

![random word vectors](https://github.com/cstorm125/thai2fit/blob/master/images/random.png?raw=true)

# Dependencies
* Python>=3.6
* PyTorch>=1.0
* fastai>=1.0.38

# Version History

## v0.1

* Pretrained language model based on Thai Wikipedia with the perplexity of 46.61
* Pretrained word embeddings (.vec) with 51,556 tokens and 300 dimensions
* Classification benchmark of 94.4% accuracy compared to 65.2% by [fastText](https://fasttext.cc/) for 4-label classification of [BEST](https://thailang.nectec.or.th/best/)

## v0.2

* Refactored to use `fastai.text` instead of `torchtext`
* Pretrained word embeddings (.vec and .bin) with 60,000 tokens and 300 dimensions (`word2vec_examples.ipynb`)
* Classification benchmark of 0.60925 micro-averaged F1 score compared to 0.49366 by [fastText](https://fasttext.cc/) and 0.58139 by competition winner for 5-label classification of [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) (`ulmfit_wongnai.ipynb`)
* Text feature extraction for other downstream tasks such as clustering (`ulmfit_ec.ipynb`)

## v0.3
* Repo name changed to `thai2fit` in order to avoid confusion since this is ULMFit not word2vec implementation
* Migrate to Pytorch 1.0 and fastai 1.0 API
* Add QRNN-based models; inference time drop by 50% on average
* Pretrained language model based on Thai Wikipedia with the perplexity of 46.04264 (20% validation) and 23.32722 (1% validation) (`pretrain_wiki.ipynb`)
* Pretrained word embeddings (.vec and .bin) with 60,000 tokens and 400 dimensions (`word2vec_examples.ipynb`) based on QRNN
* Classification benchmark of 0.60925 micro-averaged F1 score compared to 0.49366 by [fastText](https://fasttext.cc/) and 0.58139 by competition winner for 5-label classification of [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) (`ulmfit_wongnai.ipynb`)
* LSTM weights are copied from v0.2 according to guideline provided in [fastai forum](https://forums.fast.ai/t/migrate-ulmfit-weights-trained-using-fastai-0-7-to-fastai-1-0/35100)
```
I remember someone doing a script but I can’t find it. For both, you just have to map the old names of the weights to the new ones. Note that:

in language models, there is a bias in the decoder in fastai v1 that you probably won’t have
in the classifier, the order you see for the layers is artificial (it’s the pytorch representation that takes the things in the order you put them in __init__ when not using Sequential) but the two models (old and new) apply batchnorm, dropout and linear in the same order
tokenizing is done differently in fastai v1, so you may have to fine-tune your models again (we add an xxmaj token for words beginning with a capital for instance)
for weight dropout, you want the weights you have put both in '0.rnns.0.module.weight_hh_l0' and 0.rnns.0.weight_hh_l0_raw (the second one is copied to the first with dropout applied anyway)
```

## v0.31
* Support fastai>=1.0.38
* Pretrained [Thai Wikipedia Dump](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) with the same training scheme as [ulmfit-multilingual](https://github.com/n-waves/ulmfit-multilingual)
* Remove QRNN models due to inferior performance
* Classification benchmarks now include for [wongnai-corpus](https://github.com/wongnai/wongnai-corpus) (See `wongnai_cls`), [prachathai-67k](https://github.com/PyThaiNLP/prachathai-67k) (See `prachathai_cls`), and [wisesight-sentiment](https://github.com/cstorm125/wisesight-sentiment) (See `wisesight_cls`)

## v0.32
* Better text cleaning rules resulting in [Thai Wikipedia Dump](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) pretrained perplexity of 28.71067.

## v0.4 (In Progress)
* Replace AWD-LSTM/QRNN with tranformers-based models
* Named-entity recognition

# Text Classification

We trained the [ULMFit model](https://arxiv.org/abs/1801.06146) implemented by`thai2fit` for text classification. We use [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) as our benchmark as it is the only sizeable and publicly available text classification dataset at the time of writing (June 21, 2018). It has 39,999 reviews for training and validation, and 6,203 reviews for testing. 

We achieved validation perplexity at 35.75113 and validation micro F1 score at 0.598 for five-label classification. Micro F1 scores for public and private leaderboards are 0.59313 and 0.60322 respectively, which are state-of-the-art as of the time of writing (February 27, 2019). FastText benchmark based on their own [pretrained embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) has the performance of 0.50483 and 0.49366 for public and private leaderboards respectively. See `ulmfit_wongnai.ipynb` for more details.

# Text Feature Extraction

The pretrained language model of `thai2fit` can be used to convert Thai texts into vectors, after which said vectors can be used for various machine learning tasks such as classification, clustering, translation, question answering and so on. The idea is to train a language model that "understands" the texts then extract certain vectors that the model "thinks" represents the texts we want. You can access this functionality easily via [pythainlp](https://github.com/pyThaiNLP/pythainlp/)

```
from pythainlp.ulmfit import *
document_vector('วันนี้วันดีปีใหม่',learn,data)
>> array([ 0.066298,  0.307813,  0.246051,  0.008683, ..., -0.058363,  0.133258, -0.289954, -1.770246], dtype=float32)
```

# Language Modeling


The goal of this notebook is to train a language model using the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182), with data from [Thai Wikipedia Dump](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) last updated February 17, 2019. Using 40M/200k/200k tokens of train-validation-test split, we achieved validation perplexity of **27.81627 with 60,004 embeddings at 400 dimensions**, compared to state-of-the-art as of October 27, 2018 at **42.41 for English WikiText-2 by [Yang et al (2018)](https://arxiv.org/abs/1711.03953)**. To the best of our knowledge, there is no comparable research in Thai language at the point of writing (February 17, 2019). See `thwiki_lm` for more details.

# Word Embeddings

We use the embeddings from `v0.1` since it was trained specifically for word2vec as opposed to latter versions which garner to classification. The `thai2vec.bin` 51,556 word embeddings of 300 dimensions, in descending order by their frequencies (See `thai2vec.vocab`). The files are in word2vec format readable by `gensim`. Most common applications include word vector visualization, word arithmetic, word grouping, cosine similarity and sentence or document vectors. For sample code, see `thwiki_lm/word2vec_examples.ipynb`.

## Word Arithmetic

You can do simple "arithmetic" with words based on the word vectors such as:
* ผู้หญิง (female) + ราชา (king) - ผู้ชาย (male) = ราชินี (queen)
* หุ้น (stock) - พนัน (gambling) = กิจการ (business)
* อเมริกัน (american) + ฟุตบอล (football) = เบสบอล (baseball)

![word arithmetic](https://github.com/cstorm125/thai2fit/blob/master/images/word_arithematic_queen.png?raw=true)

## Word Grouping

It can also be used to do word groupings. For instance:
* อาหารเช้า อาหารสัตว์ อาหารเย็น อาหารกลางวัน (breakfast animal-food dinner lunch) - อาหารสัตว์ (animal-food) is type of food whereas others are meals in the day
* ลูกสาว ลูกสะใภ้ ลูกเขย ป้า (duaghter daughter-in-law son-in-law aunt) - ลูกสาว (daughter) is immediate family whereas others are not
* กด กัด กิน เคี้ยว (press bite eat chew) - กด (press) is not verbs for the eating process
Note that this could be relying on a different "take" than you would expect. For example, you could have answered ลูกเขย in the second example because it  is the one associated with male gender.

![word grouping](https://github.com/cstorm125/thai2fit/blob/master/images/doesnt_match1.png?raw=true)

## Cosine Similarity

Calculate cosine similarity between two word vectors.

* จีน (China) and ปักกิ่ง (Beijing): 0.31359560752667964
* อิตาลี (Italy) and โรม (Rome): 0.42819627065839394
* ปักกิ่ง (Beijing) and โรม (Rome): 0.27347283956785434
* จีน (China) and โรม (Rome): 0.02666692964073511
* อิตาลี (Italy) and ปักกิ่ง (Beijing): 0.17900795797557473

![cosine similarity](https://github.com/cstorm125/thai2fit/blob/master/images/cosin_sim_arrows.png?raw=true)

# Citation

```
@software{charin_polpanumas_2021_4429691,
  author       = {Charin Polpanumas and
                  Wannaphong Phatthiyaphaibun},
  title        = {thai2fit: Thai language Implementation of ULMFit},
  month        = jan,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.3},
  doi          = {10.5281/zenodo.4429691},
  url          = {https://doi.org/10.5281/zenodo.4429691}
}
```

# NLP Workshop at Chiangmai University

- [Getting Started with PyThaiNLP](https://github.com/PyThaiNLP/pythainlp/blob/dev/notebooks/pythainlp-get-started.ipynb)

- [thai2fit slides](https://www.canva.com/design/DADc1jbD1Hk/Iz4eFFQlbEMqjn8r99M85w/view)

- [Text Generation with Wiki Language Model](https://github.com/PyThaiNLP/pythainlp/blob/dev/notebooks/text_generation.ipynb)

- [Word Vectors](https://github.com/cstorm125/thai2fit/blob/master/thwiki_lm/word2vec_examples.ipynb)

- [Sentiment Analysis](https://github.com/PyThaiNLP/pythainlp/blob/dev/notebooks/sentiment_analysis.ipynb)

- [PyThaiNLP tutorial](https://www.thainlp.org/pythainlp/tutorials/)

- [pyThaiNLP documentation](https://www.thainlp.org/pythainlp/docs/2.0/)
