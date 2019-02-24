from fastai import *    
from fastai.text import * 
from fastai.callbacks import CSVLogger
from utils import *

data_path = 'th-all-unk/'
model_path = 'thwiki_data'

#data
print('load and sanity check data')
data = load_data(model_path,'thwiki_lm_data.pkl')
data.sanity_check()
print(len(data.train_ds), len(data.valid_ds), len(data.test_ds))
print('data done')

#learner
print('create learner')
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
trn_args = dict(drop_mult=0.9, clip=0.12, alpha=2, beta=1)
print(f'config: {config} \n trn_args: {trn_args}')

learn = language_model_learner(data, AWD_LSTM, config=config, pretrained=False, **trn_args)
learn.opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
learn.callback_fns += [partial(CSVLogger, filename=f"{model_path}/logs")]
print('learner done')

#train frozen
print('training frozen')
learn.freeze_to(-1)
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

#train unfrozen
print('training unfrozen')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8, 0.7))

#train unfrozen with lower rates
print('training unfrozen')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3/3, moms=(0.8, 0.7))

learn.save('thwiki_lm')
learn.save_encoder('thwiki_enc')
print('saved')
