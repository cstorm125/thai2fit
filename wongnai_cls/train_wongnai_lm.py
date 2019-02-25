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
tt = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', pre_rules = pre_rules_th, post_rules=post_rules_th)
processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=3)]

data_lm = (TextList.from_df(all_df, model_path, cols=['review'], processor=processor)
    .random_split_by_pct(valid_pct = 0.01, seed = 1412)
    .label_for_lm()
    .databunch(bs=64))
data_lm.sanity_check()
data_lm.save('wongnai_lm.pkl')

#learner
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
trn_args = dict(drop_mult=0.9, clip=0.12, alpha=2, beta=1)

learn = language_model_learner(data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)

#load pretrained models
learn.load_pretrained(**_THWIKI_LSTM)

#train frozen
print('training frozen')
learn.freeze_to(-1)
learn.fit_one_cycle(1, 1e-3, moms=(0.8, 0.7))

#train unfrozen
print('training unfrozen')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-4, moms=(0.8, 0.7))

#save models
learn.save('wongnai_lm')
learn.save_encoder('wongnai_enc')

