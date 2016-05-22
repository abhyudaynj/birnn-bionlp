import pickle,itertools
import json
from nltk.metrics import ConfusionMatrix
IGNORE_TAG='None'


x=[['None','None','ADE','ADE','Other','None','None','drugname','drugname','drugname']]
y=[['ADE','None','ADE','None','Other','indication','None','None','None','None']]

def get_labels(label,predicted):
    labels = list(set(itertools.chain.from_iterable(label)) | set(itertools.chain.from_iterable(predicted)))
    print labels
    return labels

def get_ConfusionMatrix(true,predicted):
    print "Confusion Matrix of combined folds (partial evaluation)"
    true_chain=list(itertools.chain.from_iterable(true))
    predicted_chain=list(itertools.chain.from_iterable(predicted))
    print ConfusionMatrix(true_chain,predicted_chain)

def get_Error_Metrics(true,predicted):
    labels=get_labels(true,predicted)
    error_types=['none2label','label2label','label2none','inj2inj']
    injury_tags=['ADE','indication','other+S%2FS%2FLIF']
    errors={key: 0 for key in error_types}
    total_errors=0
    for i,sent in enumerate(true):
        for j,word in enumerate(true[i]):
            if true[i][j] != predicted[i][j]:
                total_errors+=1
                if true[i][j] =='None':
                    errors['none2label']+=1
                else:
                    if predicted[i][j] == 'None':
                        errors['label2none']+=1
                    else:
                        if true[i][j] in injury_tags and predicted[i][j] in injury_tags:
                            errors['inj2inj']+=1
                        else:
                            errors['label2label']+=1
    print json.dumps(errors)
    print total_errors, sum(value for key, value in errors.iteritems())




def get_Exact_Metrics(true,predicted):
    labels=get_labels(true,predicted)
    get_ConfusionMatrix(true,predicted)
    true_positive={label:0 for label in labels}
    trues={label:0 for label in labels}
    positives={label:0 for label in labels}
    for i,sent in enumerate(true):
        if sent.__len__() ==0:
            continue
        label_tags=[]
        predicted_tags=[]
        j=0
        tag='Nothing'
        pos=[]
        while j<len(sent):
            if tag!=sent[j]:
                if tag != 'Nothing':
                    label_tags.append((tag,tuple(pos)))
                pos=[]
                pos.append(j)
                tag=sent[j]
            else:
                pos.append(j)
            j+=1
        label_tags.append((tag,tuple(pos)))

        j=0
        tag='Nothing'
        pos=[]
        psent=predicted[i]
        while j<len(psent):
            if tag!=psent[j]:
                if tag != 'Nothing':
                    predicted_tags.append((tag,tuple(pos)))
                pos=[]
                pos.append(j)
                tag=psent[j]
            else:
                pos.append(j)
            j+=1
        predicted_tags.append((tag,tuple(pos)))
        for z in predicted_tags:
            positives[z[0]]+=1
        for z in label_tags:
            trues[z[0]]+=1
        for z in list(set(label_tags)&set(predicted_tags)):
            true_positive[z[0]]+=1
    avg_recall = 0.0
    avg_precision =0.0
    num_candidates=0


    print positives,trues,true_positive
    for l in labels:
        if trues[l] ==0:
            recall =0
        else:
            recall=float(true_positive[l])/float(trues[l])
        if positives[l] ==0:
            precision =0
        else:
            precision=float(true_positive[l])/float(positives[l])
        if (recall+precision) ==0:
            f1 =0
        else:
            f1=2.0*recall*precision/(recall+precision)
        if l != IGNORE_TAG:
            avg_recall +=float(trues[l])*float(recall)
            avg_precision+=float(trues[l])*float(precision)
            num_candidates+=trues[l]
        print("The tag \'{0}\' has {1} elements and recall,precision,f1 ={2},{3}, {4}".format(l,trues[l],recall,precision,f1))
    if num_candidates >0:
        avg_recall =float(avg_recall)/float(num_candidates)
        avg_precision =float(avg_precision)/float(num_candidates)
    avg_f1 =0.0
    if (avg_recall+avg_precision) >0:
        avg_f1=2.0*float(avg_precision)*float(avg_recall)/(float(avg_recall)+float(avg_precision))
    print("All medical tags collectively have {0} elements and recall,precision,f1 ={1},{2}, {3}".format(num_candidates,avg_recall,avg_precision,avg_f1))
    return positives,trues,true_positive

def decoratr(l,p):
    p=[[xi[1] for xi in sent] for sent in p]
    get_Exact_Metrics(l,p)
    return l,p



