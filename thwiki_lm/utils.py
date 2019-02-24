import re
import numpy as np

from fastai.text.transform import *
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as normalize_char_order

class ThaiTokenizer(BaseTokenizer):
    "Wrapper around a newmm tokenizer to make it a `BaseTokenizer`."
    def __init__(self, lang = 'th'):
        self.lang = lang
    def tokenizer(self, t):
        return(word_tokenize(t,engine='ulmfit'))
    def add_special_cases(self, toks):
        pass
    
def replace_rep_after(t):
    "Replace repetitions at the character level in `t` after the repetition"
    def _replace_rep(m):
        c,cc = m.groups()
        return f'{c} {TK_REP} {len(cc)+1} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)

def rm_useless_newlines(t):
    "Remove multiple newlines in `t`."
    return re.sub('[\n]{2,}', ' ', t)

def rm_brackets(t):
    "Remove all empty brackets from `t`."
    new_line = re.sub('\(\)','',t)
    new_line = re.sub('\{\}','',new_line)
    new_line = re.sub('\[\]','',new_line)
    return(new_line)

#in case we want to add more specific rules for thai
pre_rules_th = [fix_html, replace_rep_after, normalize_char_order, 
                spec_add_spaces, rm_useless_spaces, rm_useless_newlines, rm_brackets]
post_rules_th = [replace_all_caps, deal_caps]

#get document vectors from language model
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tt = ThaiTokenizer()
# def document_vector(ss, learn, data):
#     s = tt.tokenizer(ss)
#     t = torch.tensor(data.vocab.numericalize(s), requires_grad=False).to(device)
#     m = learn.model[0].encoder.to(device)
#     res = m(t).mean(0).cpu().detach().numpy()
#     return(res)
   