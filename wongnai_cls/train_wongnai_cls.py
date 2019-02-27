from fastai.text import *
from fastai.callbacks import CSVLogger

from pythainlp.ulmfit import *

model_path = 'wongnai_data/'

#process data
train_df = pd.read_csv('w_review_train.csv',sep=';',header=None).drop_duplicates()
train_df.columns = ['review','rating']
test_df = pd.read_csv('test_file.csv',sep=';')
test_df['rating'] = 0
all_df = pd.concat([pd.DataFrame(test_df['review']),\
                   pd.DataFrame(train_df['review'])]).reset_index(drop=True)
two_df = pd.concat([train_df[train_df.rating==2].copy() for i in range(2)]).reset_index(drop=True)
one_df = pd.concat([train_df[train_df.rating==1].copy() for i in range(10)]).reset_index(drop=True)
train_bal = pd.concat([train_df,one_df,two_df]).reset_index(drop=True)

#databunch
data_lm = load_data(model_path,'wongnai_lm.pkl')
tt = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', pre_rules = pre_rules_th, post_rules=post_rules_th)
processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=3)]

data_cls = (TextList.from_df(train_bal, model_path, cols=['review'], processor=processor)
    .random_split_by_pct(valid_pct = 0.01, seed = 1412)
    .label_from_df('rating')
    .add_test(TextList.from_df(test_df, model_path, cols=['review'], processor=processor))
    .databunch(bs=32)
    )
data_cls.sanity_check()
data_cls.save('wongnai_cls.pkl')

#make sure we got the right number of vocab
print(f'Vocab size matched: {len(data_cls.vocab.itos)==len(data_lm.vocab.itos)}')

#create learner
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
trn_args = dict(bptt=70, drop_mult=0.5, alpha=2, beta=1,max_len=1400)
learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
learn.opt_func = partial(optim.Adam, betas=(0.7, 0.99))
learn.callback_fns += [partial(CSVLogger, filename="logs_cls")]

#load pretrained finetuned model
learn.load_encoder('wongnai_enc')

#train unfrozen
learn.freeze_to(-1)
learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))

# #gradual unfreezing
learn.freeze_to(-2)
learn.fit_one_cycle(5, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
#learn.freeze_to(-3)
#learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
# learn.unfreeze()
# learn.fit_one_cycle(1, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))

learn.save('wongnai_cls')
print('done')
