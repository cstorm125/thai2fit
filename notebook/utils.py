import re
import numpy as np
from fastai.text import *
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score

#paralellized thai tokenizer with some text cleaning
class ThaiTokenizer():
    def __init__(self, engine='newmm'):
        self.engine = engine
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.re_rep = re.compile(r'(\S)(\1{3,})')

    def sub_br(self,x): return self.re_br.sub("\n", x)

    def tokenize(self,x):
        return [t for t in word_tokenize(self.sub_br(x),engine=self.engine)]
    
    #replace aaaaaaaa
    @staticmethod
    def replace_rep(m):
        TK_REP = 'tkrep'
        c,cc = m.groups()
        return f'{TK_REP}{len(cc)+1}{c}'

    def proc_text(self, s):
        s = self.re_rep.sub(ThaiTokenizer.replace_rep, s)
        s = re.sub(r'([/#])', r' \1 ', s)
        #remvoe double space
        s = re.sub(' {2,}', ' ', s)
        return self.tokenize(s)

    @staticmethod
    def proc_all(ss):
        tok = ThaiTokenizer()
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss):
        ncpus = num_cpus()//2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(ThaiTokenizer.proc_all, ss), [])
        
#get tokenized texts
BOS = 'xbos'  # beginning-of-sentence tag
def get_texts(df):
    labels = df.iloc[:,0].values.astype(np.int64)
    texts = BOS+df.iloc[:,1].astype(str).apply(lambda x: x.rstrip())
    tok = ThaiTokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df):
    tok, labels = [], []
    for i, r in enumerate(df):
        tok_, labels_ = get_texts(r)
        tok += tok_;
        labels += labels_
    return tok, labels

#convert text dataframe to numericalized dataframes
def numericalizer(df, max_vocab = 60000, min_freq = 2, pad_tok = '_pad_',
            unk_tok = '_unk_'):
    tok, labels = get_all(df)
    freq = Counter(p for o in tok for p in o)
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, pad_tok)
    itos.insert(0, unk_tok)
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    lm = np.array([[stoi[o] for o in p] for p in tok])
    return(lm,tok,labels,itos,stoi,freq)

#get document vectors from language model
def document_vector(ss, m, stoi,tok_engine='newmm'):
    s = word_tokenize(ss,tok_engine)
    t = LongTensor([stoi[i] for i in s]).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    m.reset()
    pred,*_ = m[0](t)
    #get average of last lstm layer along bptt
    res = to_np(torch.mean(pred[-1],0).view(-1))
    return(res)
    
#load pretrained embeddings
def pretrained_wgts(em_sz, wgts, itos_pre, itos_cls):
    vocab_size = len(itos_cls)
    enc_wgts = to_np(wgts['0.encoder.weight'])
    #average weight of encoding
    row_m = enc_wgts.mean(0)
    stoi_pre = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos_pre)})
    #new embedding based on classification dataset
    new_w = np.zeros((vocab_size, em_sz), dtype=np.float32)
    for i,w in enumerate(itos_cls):
        r = stoi_pre[w]
        #use pretrianed embedding if present; else use the average
        new_w[i] = enc_wgts[r] if r>=0 else row_m
    wgts['0.encoder.weight'] = T(new_w)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    wgts['1.decoder.weight'] = T(np.copy(new_w))
    return(wgts)
    
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

#plotting
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
