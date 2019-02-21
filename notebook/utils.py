import re
import numpy as np
from fastai import *    
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
    def _replace_rep(m:Collection[str]):
        c,cc = m.groups()
        return f' {c} {TK_REP} {len(cc)+1} '
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
thai_rules = [fix_html, deal_caps, replace_rep_after, normalize_char_order, 
              spec_add_spaces, rm_useless_spaces, rm_useless_newlines, rm_brackets]

#get document vectors from language model
def document_vector(ss, m, stoi):
    s = word_tokenize(ss,engine='ulmfit')
    t = LongTensor([stoi[i] for i in s]).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    m.reset()
    pred,*_ = m[0](t)
    #get average of last lstm layer along bptt
    res = to_np(torch.mean(pred[-1],0).view(-1))
    return(res)
   