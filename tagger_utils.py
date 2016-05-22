import pickle,logging,random,gensim
import sklearn,pickle
from random import shuffle
from sklearn.metrics import f1_score,recall_score,precision_score
import collections
logging.basicConfig(level=logging.INFO)
import numpy as np
import random
import string
from nltk.corpus import sentiwordnet as swn
from nltk.metrics import ConfusionMatrix

KEY_LABEL = 'ADE'
PADDING_REPLACEMENT='None'
punct_list=[x for x in string.punctuation]

def get_vocab(tagged_data):
    vocab_set=set()
    tag_set=set()
    for i,sent in enumerate(tagged_data):
        vocab_set= vocab_set | set([x[0].lower() for x,y in tagged_data[i] if x[0] !='\x00'])
        tag_set = tag_set | set([y for x,y in tagged_data[i]])
    print('Tag Set :',tag_set)
    #pickle.dump(vocab_set,open('/home/abhyuday/temp/ADE/vocab_set_short.pkl','wb'))
    #vocab_set = pickle.load(open('/home/abhyuday/temp/ADE/vocab_set_short.pkl','rb'))
    return vocab_set,tag_set


def trim_tags(tagged_data):
    for i,sent in enumerate(tagged_data):
        tagged_data[i]=[(x,'ADE') if y=='ADE+occured' or y=='adverse+effect' else (x,y) for x,y in tagged_data[i]]
        tagged_data[i]=[(x,'None') if y=='MedDRA' else (x,y) for x,y in tagged_data[i]]
    return tagged_data


def get_embedding_weights(w2i):
    i2w={i: word for word, i in w2i.iteritems()}
    logging.info('embedding sanity check (should be a word) :{0}'.format(i2w[12]))
    mdl=gensim.models.Word2Vec.load_word2vec_format('/home/abhyuday/temp/NAACL/data/pubmed+wiki+pitts-nopunct-lower.bin',binary=True)
    logging.info('{0},{1}'.format(mdl['is'].shape,len(w2i)))
    emb_i=np.array([mdl[str(i2w[i])] if i in i2w and str(i2w[i]) in mdl else np.zeros(mdl['is'].shape[0],) for i in xrange(len(w2i))])
    return emb_i


def encode_words(entire_note):
    logging.info('Reading Data Files ...')
    #tagged_data=pickle.load(open('/home/abhyuday/temp/NAACL/data/NAACL_cancer.pkl','rb'))
    tagged_data=pickle.load(open('/home/abhyuday/temp/NAACL/data/NAACL_extracted_with_punct_dummpyPOS.pkl','rb'))
    shuffle(tagged_data)
    #flattening notes into sentences.
    if entire_note:
        note_data=[]
        for notes in tagged_data:
            note=[word for sent in notes for word in sent+[(('EOS_','EOS_'),'None')]]
            note_data.append(note)
        tagged_data=note_data
    else:
        tagged_data=[sentence for notes in tagged_data for sentence in notes]
    tagged_data=trim_tags(tagged_data)
    #tagged_data=trim_fraunhofer_tags(tagged_data)
    v_set,t_set=get_vocab(tagged_data)
    logging.info('Total Word Vocabulary Size : {0}'.format(len(v_set)))
    logging.info('Total Tag Vocabulary Size : {0}'.format(len(t_set)))
    w2i={word :i+1 for i,word in enumerate(list(v_set))}
    w2i['OOV_CHAR']=0
    #print(w2i)
    t2i={word :i for i,word in enumerate(list(t_set))}
    logging.info('embedding sanity check (should be a number >1):{0}'.format(w2i['is']))
    X=[None]*len(tagged_data)
    Y=[None]*len(tagged_data)
    Z=[None]*len(tagged_data)
    logging.info('Preparing data ...')
    for i,sent in enumerate(tagged_data):
        x=[w2i[word.lower()] if word.lower() in w2i else 0 for (word,tag),label in tagged_data[i]]
        z=[word if word in w2i else 0 for (word,tag),label in tagged_data[i]]
        y=[t2i[label] if label in t2i else 0 for word,label in tagged_data[i]]
        X[i]=x
        Y[i]=y
	Z[i]=z
    emb_w=get_embedding_weights(w2i)
    #Z=generate_crf_features(tagged_data)
    return X,Z,Y,len(t_set),emb_w,t2i,w2i

def load_data(nb_words=None, skip_top=0, maxlen=100, test_split=0.2, seed=113,
          start_char=1, oov_char=2, index_from=3,shuffle=False,entire_note=False):
    X,Z,Y,numTags,emb_w,t2i,w2i =  encode_words(entire_note)
    if shuffle:
	    np.random.seed(seed)
	    np.random.shuffle(X)
	    np.random.seed(seed)
	    np.random.shuffle(Z)
	    np.random.seed(seed)
	    np.random.shuffle(Y)
    #print ('debug : length {0} shape {1}'.format(len(Y[2]),len(Y[2])))
    if maxlen:
        logging.info('Truncating {0} instances out of {1}'.format(sum(1 if len(y)>100 else 0 for y in Y),sum(1 for y in Y)))
        X=[x1[:maxlen] for x1 in X]  #to remove computation burden
        Z=[z1[:maxlen] for z1 in Z]  #to remove computation burden
        Y=[y1[:maxlen] for y1 in Y]  #to remove computation burden
    if entire_note ==False:
        pickle.dump(((X,Z,Y), numTags, emb_w ,t2i,w2i),open('/home/abhyuday/temp/NAACL/data/NAACL_cancer_processed.pkl','wb'))
    return (X,Z,Y),numTags,emb_w,t2i,w2i


def make_cross_validation_sets(data_len,n,training_percent=None):
    if training_percent ==None:
        training_percent = 1.0
    else:
        training_percent = float(training_percent)/100.0
    split_length=int(data_len/n)
    splits=[None]*n
    for i in xrange(n):
        arr=np.array(range(data_len))
        test=range(i*split_length,(i+1)*split_length)
        mask=np.ones(arr.shape,dtype=bool)
        mask[test]=0
        train=arr[mask]
        training_len=float(len(train))*training_percent
        train=train[:int(training_len)]
        splits[i]=(train.tolist(),test)
    #print "\n".join(str(x) for x in splits)
    return splits


def evaluate_f1(y_true,y_pred,verbose =False,preMsg=None):
    print('verbose value is ',verbose)
    if verbose:
        print ConfusionMatrix(y_true,y_pred)	
    z_true=y_true
    z_pred=y_pred
    label_dict={x:i for i,x in enumerate(list(set(z_true) | set(z_pred)))}
    freq_dict=collections.Counter(z_true)
    z_true=[ label_dict[x] for x in z_true]
    z_pred=[ label_dict[x] for x in z_pred]
    f1s= f1_score(z_true,z_pred, average=None)
    print(str(preMsg)+'F1 score'+str(f1s))
    rs= recall_score(z_true,z_pred, average=None)
    ps= precision_score(z_true,z_pred, average=None)
    results =[]
    f1_none=[]
    print(str(preMsg)+str(label_dict))
    for i in label_dict:
            print("{5} The tag \'{0}\' has {1} elements and recall,precision,f1 ={3},{4}, {2}".format(i,freq_dict[i],f1s[label_dict[i]],rs[label_dict[i]],ps[label_dict[i]],preMsg))
            if i!='None' and i!='|O':
                f1_none=f1_none+[(rs[label_dict[i]],ps[label_dict[i]],f1s[label_dict[i]],freq_dict[i]),]
    all_medical_words=sum([z[3] for z in f1_none])
    macro_averaged_recall= sum([float(z[0])*float(z[3]) for z in f1_none])/sum([float(z[3]) for z in f1_none])
    macro_averaged_precision= sum([float(z[1])*float(z[3]) for z in f1_none])/sum([float(z[3]) for z in f1_none])
    if (macro_averaged_recall+macro_averaged_precision) == 0.0:
        macro_averaged_f =0.0
    else:
        macro_averaged_f = 2.0* macro_averaged_recall*macro_averaged_precision/(macro_averaged_recall+macro_averaged_precision)
    print("{4} All medical tags  have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(all_medical_words,macro_averaged_recall,macro_averaged_precision,macro_averaged_f,preMsg))
    return macro_averaged_f

