# thai2vec
Language Modeling, Word2Vec and Text Classification in Thai Language.
Created as part of [pyThaiNLP](https://github.com/PyThaiNLP/).

Models and word embeddings can also be downloaded via [Google Drive](https://drive.google.com/drive/folders/1_vZr_iR_LqIX4rEi7i5spN_6QDgj5r81?usp=sharing) or [Dropbox](https://www.dropbox.com/sh/t9qfj2ethst8g20/AAAgud8rZ_Wuv6fkXq0HEj4da?dl=0).

We provide state-of-the-art language modeling (perplexity of 46.61 on Thai wikipedia) and text classification (94.4% accuracy on four-label classification problem. Benchmarked to 65.2% by [fastText](fasttext.cc) on NECTEC's [BEST dataset](https://thailang.nectec.or.th/best/)). Credits to [fast.ai](http://www.fast.ai/).

![random word vectors](https://raw.githubusercontent.com/cstorm125/thai2vec/master/data/thaiwiki/png/random.png)
*Random word vectors* 

# Word Embeddings

The `thai2vec.vec` contains 51556 word embeddings of 300 dimensions, in descending order by their frequencies (See `thai2vec.vocab`). The files are in word2vec format readable by `gensim`. Most common applications include word vector visualization, word arithmetic, word grouping, cosine similarity and sentence or document vectors. For sample code, see `examples.ipynb`.

## Word Arithmetic

You can do simple "arithmetic" with words based on the word vectors such as:
* ผู้หญิง (female) + ราชา (king) - ผู้ชาย (male) = ราชินี (queen)
* หุ้น (stock) - พนัน (gambling) = กิจการ (business)
* อเมริกัน (american) + ฟุตบอล (football) = เบสบอล (baseball)

![word arithmetic](https://raw.githubusercontent.com/cstorm125/thai2vec/master/data/thaiwiki/png/word_arithematic_queen.png)

## Word Grouping

It can also be used to do word groupings. For instance:
* อาหารเช้า อาหารสัตว์ อาหารเย็น อาหารกลางวัน (breakfast animal-food dinner lunch) - อาหารสัตว์ (animal-food) is type of food whereas others are meals in the day
* ลูกสาว ลูกสะใภ้ ลูกเขย ป้า (duaghter daughter-in-law son-in-law aunt) - ลูกสาว (daughter) is immediate family whereas others are not
* กด กัด กิน เคี้ยว (press bite eat chew) - กด (press) is not verbs for the eating process
Note that this could be relying on a different "take" than you would expect. For example, you could have answered ลูกเขย in the second example because it  is the one associated with male gender.

![word grouping](https://raw.githubusercontent.com/cstorm125/thai2vec/master/data/thaiwiki/png/doesnt_match1.png)

## Cosine Similarity

Calculate cosine similarity between two word vectors.

* จีน (China) and ปักกิ่ง (Beijing): 0.31359560752667964
* อิตาลี (Italy) and โรม (Rome): 0.42819627065839394
* ปักกิ่ง (Beijing) and โรม (Rome): 0.27347283956785434
* จีน (China) and โรม (Rome): 0.02666692964073511
* อิตาลี (Italy) and ปักกิ่ง (Beijing): 0.17900795797557473

![cosine similarity](https://raw.githubusercontent.com/cstorm125/thai2vec/master/data/thaiwiki/png/cosin_sim_arrows.png)

## Sentence/Document Vectors

One of the most immediate use cases for thai2vec is using it to estimate a sentence vector for text classification.


# Language Modeling

Thai word embeddings and language model are trained using the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182)--basically LSTM with droupouts--with data from [Wikipedia](https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2) (pulled on January 16, 2018). We achieved perplexity of **46.61 with 51556 embeddings** (80/20 validation split; cut by pyThaiNLP), compared to [state-of-the-art on November 17, 2017](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems) at **40.68** for English language. To the best of our knowledge, there is no comparable research in Thai language at the point of writing (January 25, 2018). Details can be found in the notebook `language_modeling.ipynb`.

# Text Classification

We follow [Howard and Ruder (2018)](https://arxiv.org/abs/1801.06146) approach on finetuning language models for text classification. The language model used is the one previously trained--the [fast.ai](http://www.fast.ai/) version of [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182). The dataset is NECTEC's [BEST](https://thailang.nectec.or.th/best/), which is labeled as article, encyclopedia, news and novel. We preprocessed to remove the segmentation token and used an 80/20 split for training and validation. This resulted in 119241 sentences in the training and 29250 sentences in the validation set. We achieved **94.4%** accuracy of four-label classification using the finetuning model as compared to **65.2%** by [fastText](fasttext.cc) using their own [pretrained embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). 

# To-do

v0.1
* [x] Language modeling based on wikipedia dump
* [x] Extract embeddings and save as gensim format
* [x] Fine-tuning model for text classification on BEST
* [x] Benchmark text classification with FastText

v0.2 - the refactoring and ulmfit
* [] Language modeling based on wikipedia dump
* [] Extract embeddings and save as gensim format
* [] Fine-tuning model for text classification on e-commerce review data
* [] Benchmark text classification with FastText

