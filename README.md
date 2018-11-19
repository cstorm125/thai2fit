# thai2fit (formerly thai2vec)
State-of-the-Art Language Modeling, Text Feature Extraction and Text Classification in Thai Language.
Created as part of [pyThaiNLP](https://github.com/PyThaiNLP/) with [ULMFit](https://arxiv.org/abs/1801.06146) implementation from [fast.ai](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

Models and word embeddings can also be downloaded via [Dropbox](https://www.dropbox.com/sh/t9qfj2ethst8g20/AAAgud8rZ_Wuv6fkXq0HEj4da?dl=1).

We provide state-of-the-art language modeling (perplexity of 46.04264 at 20% validation and **randnum** at 1% validation on Thai wikipedia) and text classification (micro-averaged F-1 score of **randnum** on 5-label classification problem. Benchmarked to 0.49366 by [fastText](fasttext.cc) on [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction). The language model can also be used to extract text features for other downstream tasks.

![random word vectors](https://github.com/cstorm125/thai2fit/blob/dev/images/random.png?raw=true)

# Dependencies
* Python 3.6
* PyTorch 1.0
* fastai 1.0

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
* Replace LSTM layers with QRNN; inference time drop by 50% on average
* Pretrained language model based on Thai Wikipedia with the perplexity of 46.04264 (20% validation) and 23.32722 (1% validation) (`pretrain_wiki.ipynb`)
* Pretrained word embeddings (.vec and .bin) with 60,000 tokens and 400 dimensions (`word2vec_examples.ipynb`)
* Classification benchmark of **randnum** micro-averaged F1 score compared to 0.49366 by [fastText](https://fasttext.cc/) and 0.58139 by competition winner for 5-label classification of [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) (`ulmfit_wongnai.ipynb`)

## v0.4 (In Progress)
* Replace AWD-QRNN with tranformers
* Named-entity recognition

# Text Classification

We trained the [ULMFit model](https://arxiv.org/abs/1801.06146) implemented by`thai2fit` for text classification. We use [Wongnai Challenge: Review Rating Prediction](https://www.kaggle.com/c/wongnai-challenge-review-rating-prediction) as our benchmark as it is the only sizeable and publicly available text classification dataset at the time of writing (June 21, 2018). It has 39,999 reviews for training and validation, and 6,203 reviews for testing. 

We achieved validation perplexity at **randnum** and validation micro F1 score at **randnum** for five-label classification. Micro F1 scores for public and private leaderboards are **randnum** and **randnum** respectively (supposedly we could train further with the 15% validation set we did not use), which are state-of-the-art as of the time of writing (June 21, 2018). FastText benchmark based on their own [pretrained embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) has the performance of 0.50483 and 0.49366 for public and private leaderboards respectively. See `ulmfit_wongnai.ipynb` for more details.

# Text Feature Extraction

The pretrained language model of `thai2fit` can be used to convert Thai texts into vectors (roll credits!), after which said vectors can be used for various machine learning tasks such as classification, clustering, translation, question answering and so on. The idea is to train a language model that "understands" the texts then extract certain vectors that the model "thinks" represents the texts we want. We use 113,962 product reviews scraped from an ecommerce website as our sample dataset. See `ulmfit_ec.ipynb` for more details.


# Language Modeling

Thai word embeddings and language model are trained using the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182)--QRNN with dropouts--with data from [Wikipedia](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2). Using 80/20 train-validation split, we achieved perplexity of **46.04264 with 60,002 embeddings at 400 dimensions**, compared to state-of-the-art as of June 12, 2018 at **40.68 for English WikiText-2 by [Yang et al (2017)](https://arxiv.org/abs/1711.03953)**. To the best of our knowledge, there is no comparable research in Thai language at the point of writing (November 11, 2018). See `pretrain_wiki.ipynb` for more details.

# Word Embeddings

The `thai2vec.vec` contains 60,000 word embeddings (plus padding and unknown tokens) of 400 dimensions, in descending order by their frequencies (See `thai2vec.vocab`). The files are in word2vec format readable by `gensim`. Most common applications include word vector visualization, word arithmetic, word grouping, cosine similarity and sentence or document vectors. For sample code, see `word2vec_examples.ipynb`.

## Word Arithmetic

You can do simple "arithmetic" with words based on the word vectors such as:
* ผู้หญิง (female) + ราชา (king) - ผู้ชาย (male) = ราชินี (queen)
* หุ้น (stock) - พนัน (gambling) = กิจการ (business)
* อเมริกัน (american) + ฟุตบอล (football) = เบสบอล (baseball)

![word arithmetic](https://github.com/cstorm125/thai2fit/blob/dev/images/word_arithematic_queen.png)

## Word Grouping

It can also be used to do word groupings. For instance:
* อาหารเช้า อาหารสัตว์ อาหารเย็น อาหารกลางวัน (breakfast animal-food dinner lunch) - อาหารสัตว์ (animal-food) is type of food whereas others are meals in the day
* ลูกสาว ลูกสะใภ้ ลูกเขย ป้า (duaghter daughter-in-law son-in-law aunt) - ลูกสาว (daughter) is immediate family whereas others are not
* กด กัด กิน เคี้ยว (press bite eat chew) - กด (press) is not verbs for the eating process
Note that this could be relying on a different "take" than you would expect. For example, you could have answered ลูกเขย in the second example because it  is the one associated with male gender.

![word grouping](https://github.com/cstorm125/thai2fit/blob/dev/images/doesnt_match1.png)

## Cosine Similarity

Calculate cosine similarity between two word vectors.

* จีน (China) and ปักกิ่ง (Beijing): 0.31359560752667964
* อิตาลี (Italy) and โรม (Rome): 0.42819627065839394
* ปักกิ่ง (Beijing) and โรม (Rome): 0.27347283956785434
* จีน (China) and โรม (Rome): 0.02666692964073511
* อิตาลี (Italy) and ปักกิ่ง (Beijing): 0.17900795797557473

![cosine similarity](https://github.com/cstorm125/thai2fit/blob/dev/images/cosin_sim_arrows.png)



