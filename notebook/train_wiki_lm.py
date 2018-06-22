import re
import html

from fastai.text import *
from pythainlp.tokenize import word_tokenize

import dill as pickle
from IPython.display import Image
from IPython.core.display import HTML 

DATA_PATH='/home/ubuntu/Projects/new2vec/data/'
MODEL_PATH = f'{DATA_PATH}models/'

trn_lm = np.load(f'{MODEL_PATH}trn_lm.npy')
val_lm = np.load(f'{MODEL_PATH}val_lm.npy')
itos = pickle.load(open(f'{MODEL_PATH}itos.pkl', 'rb'))

em_sz,nh,nl = 300,1150,3
wd=1e-7
bptt=70
bs=64
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
vocab_size = len(itos)
weight_factor = 0.7
drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*weight_factor

trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(path=DATA_PATH, pad_idx=1, n_tok=vocab_size, 
                       trn_dl=trn_dl, val_dl=val_dl, bs=bs, bptt=bptt)

learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
learner.metrics = [accuracy]

lr=1e-3


learner.fit(lr, n_cycle=1, wds=wd, use_clr=(20,10), cycle_len=15)
learner.save('thwiki_model')
learner.save_encoder('thwiki_enc')

learner.load('thwiki_model')
learner.fit(lr, n_cycle=1, wds=wd, use_clr=(20,10), cycle_len=10)
learner.save('thwiki_model2')
learner.save_encoder('thwiki_enc2')

