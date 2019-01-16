#trained with fastai 0.7 and converted to fastai 1.0
from fastai import *    
from fastai.text import * 
import dill as pickle

DATA_PATH='../lm_data/'
MODEL_PATH = f'{DATA_PATH}models/'


data = TextLMDataBunch.load(DATA_PATH,'lm',bs=32)
print('done data bunch')

#learner
#heuristic reference from imdb_scripts
learn = language_model_learner(data, bptt = 70, emb_sz = 400, nh = 1150, nl = 3,
                                  drop_mult = 0.5, bias = False, qrnn = False, 
                                  alpha=2, beta = 1, 
                                  pretrained_fnames = None)
learn.metrics = [accuracy]
learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
print('done learner')

#training 1
lr = 1e-3 #lr find heuristics
wd = 1e-7 #lr find heuristics
learn.wd=wd

learn.fit_one_cycle(cyc_len = 30, 
                    max_lr= lr, #learning rate
                    div_factor= 20, #factor to discount from max
                    moms = (0.8, 0.7), #momentums
                    pct_start = 0.1, #where the peak is at 
                    wd = wd #weight decay
                   ) 

learn.save_encoder('thwiki_enc_lstm')
learn.save('thwiki_model_lstm')
print('saved')
