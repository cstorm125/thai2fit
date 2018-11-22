from fastai import *    
from fastai.text import * 
from utils import *
import dill as pickle

DATA_PATH='../lm_data/'
MODEL_PATH = f'{DATA_PATH}models/'


data = TextLMDataBunch.load(DATA_PATH,'lm',bs=32)
print('done data bunch')

#learner
#heuristic reference from imdb_scripts
learn = language_model_learner(data, bptt = 70, emb_sz = 400, nh = 1550, nl = 3,
                                  drop_mult = 0.1, bias = True, qrnn = True, 
                                  alpha=2, beta = 1, clip = 0.12, 
                                  pretrained_fnames = None)
learn.metrics = [accuracy]
learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))
print('done learner')

#training 1
lr = 0.005 #lr find heuristics
wd = 1e-7 #lr find heuristics
learn.wd=wd

#train first for 20 epochs
learn.fit_one_cycle(cyc_len = 20, 
                    max_lr= lr, #learning rate
                    div_factor= 20, #factor to discount from max
                    moms = (0.8, 0.7), #momentums
                    pct_start = 0.1, #where the peak is at 
                    wd = wd #weight decay
                   ) 

learn.save('thwiki_model_qrnn')
print('saved')

#training 2; lower learning rate, dropout, and weight decays
learn = language_model_learner(data, bptt = 70, emb_sz = 400, nh = 1550, nl = 3,
                                  drop_mult = 0., bias = True, qrnn = True, 
                                  alpha=2, beta = 1, clip = 1., tie_weights=True,
                                  pretrained_fnames = None)
learn.metrics = [accuracy]
learn.opt_func = partial(optim.Adam, betas=(0.8, 0.99))

#load model
learn.load('thwiki_model_qrnn')
print('done learner')

lr = 0.001 #lower lr
wd = 1e-8 #lower wd
learn.wd=wd

learn.fit_one_cycle(cyc_len = 10, 
                    max_lr= lr, #learning rate
                    div_factor= 20, #factor to discount from max
                    moms = (0.8, 0.7), #momentums
                    pct_start = 0.1, #where the peak is at 
                    wd = wd #weight decay
                   ) 
