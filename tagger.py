from __future__ import absolute_import
import numpy as np
import pickle,argparse,eval_metrics
import sys,random,json
import progressbar
np.random.seed(1337)  # for reproducibility

import errno
from nltk.metrics import ConfusionMatrix
import itertools
import time
import logging
import lasagne
logging.basicConfig(level=logging.INFO)
import tagger_utils as preprocess
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as sk_shuffle
import theano.tensor as T
import theano
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.layers import InputLayer, DenseLayer, prelu
import pycrfsuite
from nltk.tag import CRFTagger

params={}
sl=logging.getLogger(__name__)

wl=logging.getLogger(__name__+'_write')


def pad_and_mask(X,Y, maxlen=None,padding='pre', value=0.):
    '''
        adapted from keras pad_and_mask function
    '''
    lengths = [len(s) for s in Y]
    if maxlen is None:
        maxlen = np.max(lengths)
        global params
        params['maxlen']=maxlen
    y_dim =max([max(s) for s in Y])+1
    x_dim = 1
    identity_mask=np.eye(y_dim)
    Y=[[identity_mask[w] for w in s] for s in Y]
    nb_samples = len(Y)
    
    logging.info('Maximum sequence length is {0}\nThe X dimension is {3}\nThe y dimension is {1}\nThe number of samples in dataset are {2}'.format(maxlen,y_dim,nb_samples,x_dim))

    x = np.zeros((nb_samples, maxlen,x_dim))
    y = (np.ones((nb_samples, maxlen, y_dim)) * value).astype('int32')
    mask = (np.ones((nb_samples, maxlen,)) * 0.).astype('int32')
    
    for idx, s in enumerate(X):
        X[idx]=X[idx][:maxlen]
        Y[idx]=Y[idx][:maxlen]

    for idx, s in enumerate(X):
        if padding == 'post':
            x[idx, :len(X[idx]),0] = X[idx]
            y[idx, :len(X[idx])] = Y[idx]
            mask[idx, :len(X[i])] = 1
        elif padding == 'pre':
            x[idx, -len(X[idx]):,0] = X[idx]
            y[idx, -len(X[idx]):] = Y[idx]
            mask[idx, -len(X[idx]):] = 1

    logging.info('X shape : {0}\nY shape : {1}\nMask shape : {2}'.format(x.shape,y.shape,mask.shape))
    return x,y,mask


def setup_NN(worker,x_in,mask_in,y_in):
    batch_size = x_in.shape[0]
    print('X train batch size {0}'.format(x_in.shape))
    print('Y train batch size {0}'.format(y_in.shape))
    print('mask train batch size {0}'.format(mask_in.shape))
    xt=theano.shared(x_in)
    mt=theano.shared(mask_in)
    yt=theano.shared(y_in)
    premodel=time.time()
    
    # Setting up the NN architecture

    l_in=lasagne.layers.InputLayer(shape=(None,params['maxlen'],1),input_var=xt.astype('int32'))
    l_mask=lasagne.layers.InputLayer(shape=(None,params['maxlen'])) 
    if params['word2vec']==1:
        l_emb=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],emb_w.shape[1],W=emb_w.astype('float32'))
        l_emb.add_param(l_emb.W, l_emb.W.get_value().shape, trainable=params['emb1'])
    else:
        l_emb=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],emb_w.shape[1])
    emb_out=lasagne.layers.get_output(l_emb)
    emb_out_f=theano.function([l_in.input_var],emb_out)
    print("output shape for emb",emb_out_f(x_in.astype('int32')).shape)


    if params['emb2']>0:
        l_emb1=lasagne.layers.EmbeddingLayer(l_in,emb_w.shape[0],params['emb2'])
        l_emb=lasagne.layers.ConcatLayer([l_emb,l_emb1],axis=3)

    dropout_backward=lasagne.layers.DropoutLayer(l_emb,params['noise1'])
    dropout_forward=lasagne.layers.DropoutLayer(l_emb,params['noise1'])
    if params['rnn']=='GRU':
        backward1= lasagne.layers.GRULayer(dropout_backward,params['hidden1'],mask_input=l_mask,backwards=True,precompute_input=True,gradient_steps= params['gradient_steps'])

        forward1= lasagne.layers.GRULayer(dropout_forward,params['hidden1'],mask_input=l_mask,precompute_input=True,gradient_steps= params['gradient_steps'])


    else:
        if params['hidden2'] >0:
            backward1= lasagne.layers.LSTMLayer(dropout_backward,params['hidden2'],mask_input=l_mask,peepholes=params['peepholes'],forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=lasagne.nonlinearities.tanh,backwards=True,precompute_input=True,gradient_steps= params['gradient_steps'])
            forward1= lasagne.layers.LSTMLayer(dropout_forward,params['hidden2'],mask_input=l_mask,peepholes=params['peepholes'],forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=lasagne.nonlinearities.tanh,precompute_input=True,gradient_steps= params['gradient_steps'])
            dropout_forward=lasagne.layers.DropoutLayer(forward1,params['noise1'])
            dropout_backward=lasagne.layers.DropoutLayer(backward1,params['noise1'])

        backward1= lasagne.layers.LSTMLayer(dropout_backward,params['hidden1'],mask_input=l_mask,peepholes=params['peepholes'],forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=lasagne.nonlinearities.tanh,backwards=True,precompute_input=True,gradient_steps= params['gradient_steps'])

        forward1= lasagne.layers.LSTMLayer(dropout_forward,params['hidden1'],mask_input=l_mask,peepholes=params['peepholes'],forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),nonlinearity=lasagne.nonlinearities.tanh,precompute_input=True,gradient_steps= params['gradient_steps'])

    crf_layer=lasagne.layers.ConcatLayer([forward1,backward1],axis=2)
    if params['dense']==1:
        rlayer=lasagne.layers.ReshapeLayer(crf_layer,(-1,params['hidden1']*2))
        dropout1=lasagne.layers.DropoutLayer(rlayer,p=params['noise1'])
        dense1=lasagne.layers.DenseLayer(dropout1,numTags,nonlinearity=lasagne.nonlinearities.softmax)
        sum_layer=lasagne.layers.ReshapeLayer(dense1,(batch_size,params['maxlen'],numTags))
    else:
        sum_layer=lasagne.layers.ElemwiseSumLayer([forward1,backward1])
 
    outp=lasagne.layers.get_output(sum_layer,deterministic = False)
    eval_out=lasagne.layers.get_output(sum_layer,deterministic=True)
    crf_out=lasagne.layers.get_output(crf_layer,deterministic=True)
    #eval_out=lasagne.layers.get_output(rlayer2)
    lstm_output = theano.function([l_in.input_var,l_mask.input_var],eval_out)
    crf_output = theano.function([l_in.input_var,l_mask.input_var],crf_out)
    print("output shape for theano net",lstm_output(x_in.astype('int32'),mask_in.astype('float32')).shape)
    t_out=T.tensor3()
    eval_cost=T.mean((eval_out-t_out)**2)
    all_params = lasagne.layers.get_all_params(sum_layer,trainable=True)   #  <------ CHECK THE TRAINABLE PARAMS. You always forget to set this correctly.
    num_params = lasagne.layers.count_params(sum_layer,trainable=True)
    print('Number of parameters: {0}'.format(num_params))
    l2_cost= params['l2'] *lasagne.regularization.apply_penalty(all_params,lasagne.regularization.l2)
    l1_cost= params['l1'] *lasagne.regularization.apply_penalty(all_params,lasagne.regularization.l1)
    if params['crossentropy']==1:
        clipped_outp=T.clip(outp,1e-7,1.0-1e-7)
        cost=lasagne.objectives.categorical_crossentropy(clipped_outp,t_out)
        cost = cost.mean()
    else:
        cost=T.mean((outp-t_out)**2)#+ l2_penalty
    cost+=l2_cost+l1_cost
    updates = lasagne.updates.adagrad(cost, all_params,learning_rate=params['learning-rate'])  
    updates_m = lasagne.updates.apply_momentum(updates,all_params,momentum=0.9)
    train = theano.function([l_in.input_var,t_out,l_mask.input_var],cost,updates=updates_m)
    compute_cost = theano.function([l_in.input_var,t_out,l_mask.input_var],eval_cost)
    #acc_=T.sum(T.eq(T.argmax(eval_out,axis=2),T.argmax(t_out,axis=2))*T.sum(t_out,axis =2))/T.sum(l_mask.input_var)
    acc_=T.sum(T.eq(T.argmax(eval_out,axis=2),T.argmax(t_out,axis=2))*l_mask.input_var)/T.sum(l_mask.input_var)
    compute_acc = theano.function([l_in.input_var,t_out,l_mask.input_var],acc_)
    print('Time to build and compile model {0}'.format(time.time()-premodel))

    return crf_output,lstm_output,train,compute_cost,compute_acc

def iterate_minibatches(inputs,mask,targets, batchsize):
    indices=np.array(range(len(inputs)))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    #for start_idx in range(0, len(inputs), batchsize):
        #yield inputs[start_idx:start_idx + batchsize],mask[start_idx:start_idx + batchsize], targets[start_idx:start_idx + batchsize]
        yield inputs[indices[start_idx:start_idx + batchsize]],mask[indices[start_idx:start_idx + batchsize]], targets[indices[start_idx:start_idx + batchsize]]
    if len(indices[start_idx+batchsize:]) >0:
        last_inputs=inputs[indices[-batchsize:]]
        last_targets=targets[indices[-batchsize:]]
        last_mask=mask[indices[-batchsize:]]
        #last_mask[:-len(indices[start_idx+batchsize:]),:-1]=0
        last_mask[:-len(indices[start_idx+batchsize:])]=0
        yield last_inputs,last_mask,last_targets



def train_NN(train,crf_output,lstm_output,train_indices,compute_cost,compute_acc,worker):
    if params['patience'] !=0:
        vals=[0.0]*params['patience']
    sl.info('Dividing the training set into {0} % training and {1} % dev set'.format(100-params['dev'],params['dev']))
    train_i=np.copy(train_indices)
    np.random.shuffle(train_i)
    dev_length = len(train_indices)*params['dev']/100
    dev_i=train_i[:dev_length]
    train_i=train_i[dev_length:]
    sl.info('{0} training set samples and {1} dev set samples'.format(len(train_i),len(dev_i)))
    x_train=X[train_i]
    y_train=Y[train_i]
    mask_train=Mask[train_i]

    x_dev=X[dev_i]
    y_dev=Y[dev_i]
    mask_dev=Mask[dev_i]
    num_batches=float(sum(1 for _ in iterate_minibatches(x_train,mask_train,y_train,params['batch-size'])))
    for iter_num in xrange(params['epochs']):
        try:
            iter_cost=0.0
            iter_acc=0.0
            print('Iteration number : {0}'.format(iter_num+1))
            progr=progressbar.ProgressBar(maxval=num_batches).start()
            for indx,(x_i,m_i,y_i) in enumerate(iterate_minibatches(x_train,mask_train,y_train,params['batch-size'])):
                progr.update(indx+1)
                train(x_i.astype('int32'),y_i.astype('float32'),m_i.astype('float32'))
                iter_cost+=compute_cost(x_i.astype('int32'),y_i.astype('float32'),m_i.astype('float32'))
                iter_acc+=compute_acc(x_i.astype('int32'),y_i.astype('float32'),m_i.astype('float32'))
            progr.finish()
            print('TRAINING : acc = {0} loss : {1}'.format(iter_acc/num_batches,iter_cost/num_batches))
            val_acc=callback_NN(compute_cost,compute_acc,x_dev,mask_dev,y_dev)
            if params['patience'] !=0:
                vals.append(val_acc)
                vals = vals[1:]
                max_in=np.argmax(vals)
                print "val acc argmax {1} : list is : {0}".format(vals,max_in)
                if max_in ==0:
                    print "Stopping because my patience has reached its limit."
                    break
            if iter_num %5 ==0:
                res=evaluate_neuralnet(lstm_output,x_dev,mask_dev,y_dev)
        except IOError, e:
            if e.errno!=errno.EINTR:
                raise
            else:
                print " EINTR ERROR CAUGHT. YET AGAIN "
     
    print "Final Validation eval"
    evaluate_neuralnet(lstm_output,x_dev,mask_dev,y_dev)
 

def callback_NN(compute_cost,compute_acc,X_test,mask_test,y_test):
    num_valid_batches=float(sum(1 for _ in iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])))
    logging.info('Executing validation Callback')
    val_loss=0.0
    val_acc =0.0
    num_batches=float(sum(1 for _ in iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])))
    progr=progressbar.ProgressBar(maxval=num_batches).start()
    for indx,(x_i,m_i,y_i) in enumerate(iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])):
        val_loss+=compute_cost(x_i.astype('int32'),y_i.astype('float32'),m_i.astype('float32'))
        val_acc+=compute_acc(x_i.astype('int32'),y_i.astype('float32'),m_i.astype('float32'))
        
        progr.update(indx+1)
    progr.finish()
    print('VALIDATION : acc = {0} loss = {1}'.format(val_acc/num_valid_batches,val_loss/num_valid_batches))
    return val_acc/num_valid_batches

def evaluate_neuralnet(lstm_output,X_test,mask_test,y_test,strict=False):
    print('Mask len test',len(mask_test))
    predicted=[]
    predicted_sent=[]
    label=[]
    label_sent=[]
    original_sent=[]
    num_batches=float(sum(1 for _ in iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])))
    progr=progressbar.ProgressBar(maxval=num_batches).start()
    for indx,(x_i,m_i,y_i) in enumerate(iterate_minibatches(X_test,mask_test,y_test,params['batch-size'])):
        for sent_ind,m_ind in enumerate(m_i):
            o_sent = x_i[sent_ind][m_i[sent_ind]==1].tolist()
            #if sent_ind ==0:
            #    print x_i[sent_ind][m_i[sent_ind]==1].shape
            original_sent.append([i2w[int(l[0])] for l in o_sent])
            #original_sent.append(o_sent)
        y_p=lstm_output(x_i.astype('int32'),m_i.astype('float32'))
        for sent_ind,m_ind in enumerate(m_i):
            l_sent = np.argmax(y_i[sent_ind][m_i[sent_ind]==1],axis=1).tolist()
            p_sent = np.argmax(y_p[sent_ind][m_i[sent_ind]==1],axis=1).tolist()
            predicted_sent.append([i2t[l] for l in p_sent])
            #predicted_sent.append(p_sent)
            label_sent.append([i2t[l] for l in l_sent])
            #label_sent.append(l_sent)
        m_if=m_i.flatten()
        label+=np.argmax(y_i,axis=2).flatten()[m_if==1].tolist()
        predicted+=np.argmax(y_p,axis=2).flatten()[m_if==1].tolist()
        
        progr.update(indx+1)
    progr.finish()
    res=preprocess.evaluate_f1([i2t[l] for l in label],[i2t[l] for l in predicted],verbose=True,preMsg='NN:')
    if strict :
        print "###########EVALUATION STRICT ###################"
        eval_metrics.get_Exact_Metrics(label_sent,predicted_sent) 

    #print(predicted[0],predicted.__len__())
    #print(label[0],label.__len__())
    #preprocess.evaluate_f1(label,predicted,verbose=True)
    return res,(original_sent,label_sent,predicted_sent)

def driver(worker,(train_i,test_i)):
    if worker ==0:
        print('Embedding Shape :',emb_w.shape)
        print(len(X[train_i]), 'train sequences')
        print(len(X[test_i]), 'test sequences')
        print('Number of tags',numTags)
        print('X train Sanity check: ', np.amax(np.amax(X[train_i])))
        print('X test Sanity check :', np.amax(np.amax(X[test_i])))
        #y_test = y_test/numTags
        print('X_train shape:', X[train_i].shape)
        print('X_test shape:', X[test_i].shape)

        print('mask_train shape:', Mask[train_i].shape)
        print('mask_test shape:', Mask[test_i].shape)

        print('Y_train shape:', Y[train_i].shape)
        print('Y_test shape:', Y[test_i].shape)

    crf_output,lstm_output,train,compute_cost,compute_acc = setup_NN(0,X[0:params['batch-size']],Mask[0:params['batch-size']],Y[0:params['batch-size']])
    #setup_NN(0,X[0:params['batch-size']],Mask[0:params['batch-size']],Y[0:params['batch-size']])
    train_NN(train,crf_output,lstm_output,train_i,compute_cost,compute_acc,worker)
    print "TESTING EVALUATION"
    callback_NN(compute_cost,compute_acc,X[test_i],Mask[test_i],Y[test_i])
    _,results=evaluate_neuralnet(lstm_output,X[test_i],Mask[test_i],Y[test_i],strict=True)
    return results


def store_response(o,l,p,filename='response.pkl'):
    print "Storing responses in {0}".format(params['dump-dir']+filename)
    pickle.dump((params,o,l,p),open(params['dump-dir']+filename,'wb'))


def single_run():
    worker=0
    o,l,p=driver(worker,splits[worker])
    if params['error-analysis']==1:
        store_response(o,l,p,'single_run{0}_response.pkl'.format(params['instance']))

def cross_validation_run():
    label_sent=[]
    predicted_sent=[]
    original_sent=[]
    for worker in xrange(len(splits)):
        print "########### Cross Validation run : {0}".format(worker)
        o,l,p = driver(worker,splits[worker])
        label_sent += l
        predicted_sent += p
        original_sent +=o
    print "#######################VALIDATED SET ########"
    flat_label=[word for sentenc in label_sent for word in sentenc]
    flat_predicted=[word for sentenc in predicted_sent for word in sentenc]
    preprocess.evaluate_f1(flat_label,flat_predicted)
    print "STRICT ---"
    eval_metrics.get_Exact_Metrics(label_sent,predicted_sent)
    if params['error-analysis']==1:
        store_response(original_sent,label_sent,predicted_sent,'cv_run{0}_response.pkl'.format(params['instance']))


    
if __name__=="__main__":
    DATASET_DEFAULT='temp/NAACL/data/NAACL_cancer_processed.pkl'
    DEFAULT_DUMP_DIR='temp/ADE/'
    nonlinearity={'tanh':lasagne.nonlinearities.tanh ,'softmax':lasagne.nonlinearities.softmax}

    #parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--instance',dest='instance',type= int, default=0 ,help ='Set instance for testing purposes. Default 0')
    parser.add_argument('-m','--mode',dest='document',type= int, default=0 ,help ='Run in which mode. Sentence 0 Document 1.Default 0')
    parser.add_argument('-cv','--cross-validation',dest='cross-validation',type= int, default=0 ,help ='Cross Validation run. 0 off, 1 on. Default 0')
    parser.add_argument('-w','--word2vec',dest='word2vec',type= int, default=0 ,help ='Initialize the Network with wordvec embeddings if 1. else random initialization. defualt 0')
    parser.add_argument('-n','--epochs',dest='epochs',type= int, default=1 ,help ='Number of epochs for training. default 1')
    parser.add_argument('-e2','--emb2',dest='emb2',type= int, default=100 ,help ='Number of dimension of extra embedding layer. off if 0. default is 100')
    parser.add_argument('-h1','--hidden1',dest='hidden1',type= int, default=50 ,help ='Dimensionality of the first hidden layer. Default is 50')
    parser.add_argument('-h2','--hidden2',dest='hidden2',type= int, default=0 ,help ='Dimensionality of the first hidden layer. 0 is off. Default is zero')
    parser.add_argument('-n1','--noise1',dest='noise1',type= float, default=0.50 ,help ='Dropout Noise for first layer. Default is 0.50')
    parser.add_argument('-b','--batch-size',dest='batch-size',type= int, default=32 ,help ='Batch size. Default is 32')
    parser.add_argument('-r','--data-refresh',dest='data-refresh',type= int, default=0 ,help ='Use cached data, or reprocess the Dataset ? Default is 0: use cached data')
    parser.add_argument('-log','--log-file',dest='log-file',type= str, default=DEFAULT_DUMP_DIR+'temp_nn.log' ,help ='Log file that should be used.')
    parser.add_argument('-l','--maxlen',dest='maxlen',type= int, default=100 ,help ='Maximum Length for padding. default 100')
    parser.add_argument('-f','--folds',dest='folds',type= int, default=10,help ='Number of cross validation folds. 20  of the training data will be used for dev set. default 10')
    parser.add_argument('-d','--dev',dest='dev',type= int, default=20,help ='percentage of training data that will be used for dev set. default 20')
    parser.add_argument('-lr','--learning-rate',dest='learning-rate',type= float, default=0.1,help ='learning rate. default 0.1')
    parser.add_argument('-ce','--crossentropy',dest='crossentropy',type= int, default=0,help ='Cost function. 1 :crossentropy 0: mse. default 0')
    parser.add_argument('-de','--dense',dest='dense',type= int, default=0,help ='Add dense layer on top of lstm. default 0')
    parser.add_argument('-rnn','--rnn',dest='rnn',type= str, default='LSTM',help ='Use GRU or LSTM. default LSTM')
    parser.add_argument('-tp','--tp',dest='training-percent',type= int, default=100,help ='Percentage of training data used. default is 100')
    parser.add_argument('-e1','--emb1',dest='emb1',type= int, default=1,help ='Should the word2vec vectors be further trained. default 1')
    parser.add_argument('-l2','--l2cost',dest='l2',type= float, default=0.005,help ='Add l2 penalty. default 0.005')
    parser.add_argument('-l1','--l1cost',dest='l1',type= float, default=0.0,help ='Add l2 penalty. default 0.0, off')
    parser.add_argument('-err','--error-analysis',dest='error-analysis',type= int, default=1,help ='Dump test output in a file. default 1')
    parser.add_argument('-p','--patience',dest='patience',type= int, default=10,help ='Stop if the validation accuracy has not increased in the last n iterations.Off if 0. default 10')
    parser.add_argument('-c','--peepholes',dest='peepholes',type= int, default=0,help ='Keep peephole connections in LSTM on. Default is 0, off')
    parser.add_argument('-g','--gradient-steps',dest='gradient_steps',type= int, default=-1,help ='Number of timesteps to include in the backpropagated gradient. If -1, backpropagate through the entire sequence. Default is -1')
    parser.add_argument('-s','--shuffle',dest='shuffle',type= int, default=0,help ='Shuffle entire dataset. By default 0, means only shuffling the training and dev datasets.')
    args = parser.parse_args()
    params = vars(args)
    params['corpus']=DATASET_DEFAULT
    params['dump-dir']=DEFAULT_DUMP_DIR
    params['peepholes']=bool(params['peepholes'])
    params['emb1']=bool(params['emb1'])
    params['document']=bool(params['document'])
    if params['document']:
        params['maxlen']=1500
    if params['maxlen']==-1:
        params['maxlen']=None
 
    wl.addHandler(logging.FileHandler(params['log-file']+str(params['instance']),mode='w'))
    wl.propagate = False
    wl.info('Parameters')
    wl.info(json.dumps(params))
    sl.info('Using the parameters:\n {0}'.format(json.dumps(params,indent=2)))
    print('Using the parameters:\n {0}'.format(json.dumps(params,indent=2)))

    # Preparing Dataset

    if params['data-refresh']==1 or params['document']==True:
        sl.info('Preprocessing entire dataset ...')
        (X,Z,Y) , numTags, emb_w , t2i,w2i =preprocess.load_data(maxlen=params['maxlen'],entire_note=params['document'])
    else:
        sl.info('Loading cached dataset ...')
        (X,Z,Y) , numTags, emb_w , t2i,w2i = pickle.load(open(params['corpus'],'rb'))
    X,Y,Mask=pad_and_mask(X,Y,params['maxlen'])
    if params['shuffle']==1:
        X,Y,Mask=sk_shuffle(X,Y,Mask,random_state=0)
    i2t = {v: k for k, v in t2i.items()}
    i2w = {v: k for k, v in w2i.items()}
    splits = preprocess.make_cross_validation_sets(len(Y),params['folds'],training_percent=params['training-percent'])
    try:
        if params['cross-validation']==0:
            single_run()
        else:
            cross_validation_run()
    except IOError, e:
        if e.errno!=errno.EINTR:
            raise
        else:
            print " EINTR ERROR CAUGHT. YET AGAIN "
    sl.info('Using the parameters:\n {0}'.format(json.dumps(params,indent=2)))
    print('Using the parameters:\n {0}'.format(json.dumps(params,indent=2)))
