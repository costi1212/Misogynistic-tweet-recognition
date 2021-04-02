import string
import nltk
import sys
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import KNeighborsClassifier
import math as mat
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score,cross_val_predict
import matplotlib as matplot
#np.set_printoptions(threshold=sys.maxsize)
gnb=GaussianNB()
bnb=BernoulliNB()
mnb=MultinomialNB()
tweet_tokenizer = TweetTokenizer(strip_handles=True,reduce_len=False)
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
TXT_COL = 'text'
LBL_COL = 'label'
ID_COL='id'
train_data=pd.read_csv(TRAIN_FILE)
test_data=pd.read_csv(TEST_FILE)
labels = train_data[LBL_COL]
stop_words = stopwords.words('italian')
def tokenize(text):


    return tweet_tokenizer.tokenize(text)

def get_representation(vocabulary, how_many):

    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx,(cuvant, frecv) in enumerate (most_comm):
        wd2idx[cuvant]=idx
        idx2wd[idx]=cuvant
    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):

    counter = Counter()

    for text in corpus:
        #text = text_fara_semne_de_puntuatie(text)
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):


    features=np.zeros(len(wd2idx))
    for cuvant in tokenize(text):
        if cuvant in wd2idx:
            idx = wd2idx[cuvant]
            features[idx]+=1
    #features=normalize_vocabulary_by_lenth(features)
    return features


def corpus_to_bow(corpus, wd2idx):

    all_features = np.zeros((len(corpus),len(wd2idx)))
    for i,text in enumerate(corpus):
        bow= text_to_bow(text, wd2idx)
        all_features[i]=bow
    return all_features


def get_corpus_vocabulary_misogin(corpus1,corpus2):
    counter = Counter()
    for i in range(len(corpus1)):
        if(corpus2[i]==1):
            tokens = tokenize(corpus1[i])
            counter.update(tokens)
    return counter


def get_corpus_vocabulary_nemisogin(corpus1,corpus2):
    counter = Counter()
    for i in range(len(corpus1)):
        if (corpus2[i] == 0):
            tokens = tokenize(corpus1[i])
            counter.update(tokens)
    return counter

def get_difference_between_vocabularies(vocabular1,vocabular2):
    counter = Counter()
    #for i,(cuvant) in enumerate(vocabular1):
     ##      counter.update(cuvant)
    for cuvant in vocabular1:
        if(cuvant not in vocabular2):
            counter[cuvant]=vocabular1[cuvant]
    return counter


def most_frequent_in_misogin(vocabular_misogin,vocabular_nemisogin):
    counter= Counter()
    for cuvant in vocabular_misogin:
        if(cuvant in vocabular_nemisogin):
            if(vocabular_misogin[cuvant]>vocabular_nemisogin[cuvant] and len(cuvant)>2):
                counter[cuvant]=vocabular_misogin[cuvant]
    return counter

def write_prediction(out_file, predictions,test_data):

    np.savetxt(out_file, np.stack((test_data[ID_COL], predictions)).T, fmt="%d", delimiter=',',
               header="id,label", comments='')
    pass

def KNN(data,labels,test_data_bow):
    clf = KNeighborsClassifier()
    clf.fit(data, labels)
    predictions = clf.predict(test_data_bow)
    return predictions

def normalize_bow_by_Euclid(bow):
    radical=0
    for frecv in bow:
        radical+=frecv
    if(radical!=0):
        bow/=mat.sqrt(radical)
    return bow

def text_fara_semne_de_puntuatie(text):
    text_nopunct = "".join([char.lower() for char in text if char not in string.punctuation])
    return text_nopunct

def cross_validation_fscores(data_bow,mnb,labels):
    return (cross_val_score(mnb, data_bow, labels, cv=10, scoring="f1", n_jobs=-1))

def matrice_de_confuzie(data_bow,labels,mnb):
    array = np.array(cross_val_predict(mnb, data_bow, labels, cv=10, n_jobs=-1))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(0,len(array)):
        if labels[i]==array[i] and train_data[LBL_COL][i]==1:
            tp+=1
        elif labels[i]!=array[i] and train_data[LBL_COL][i]==1:
            fn+=1
        elif labels[i]!=array[i] and train_data[LBL_COL][i]==0:
            fp+=1
        elif labels[i]==array[i] and train_data[LBL_COL][i]==0:
            tn+=1
    return [[tp,fn],[fp,tn]]



#Create vocabularies
vocabular = get_corpus_vocabulary(train_data[TXT_COL])
#vocabular_misogin = get_corpus_vocabulary_misogin(train_data[TXT_COL] , train_data[LBL_COL])
#vocabular_nemisogin = get_corpus_vocabulary_nemisogin(train_data[TXT_COL],train_data[LBL_COL])
#vocabular_misogin_final=get_difference_between_vocabularies(vocabular_misogin,vocabular_nemisogin)
#vocabular_bun=most_frequent_in_misogin(vocabular_misogin,vocabular_nemisogin)
wd2idx, idx2wd= get_representation(vocabular,6000)
#150-221 170-216 160-211 159-210(207 cu semne de punctuatie)
#wd2idx, idx2wd=get_representation(vocabular_misogin_final,10)
#getting data ready
#data=corpus_to_bow(train_data[TXT_COL],wd2idx)
#test_data_bow=corpus_to_bow(test_data[TXT_COL],wd2idx)
data_bow=corpus_to_bow(train_data[TXT_COL],wd2idx)
test_data_bow=corpus_to_bow(test_data[TXT_COL],wd2idx)
#print(vocabular)
#for train_index, test_index in rs.split(data_bow):
    #y_pred = gnb.fit(data_bow[train_index],labels[train_index]).predict(data_bow[test_index])
    #print(accuracy_score(labels[test_index],labels[y_pred]))
print(matrice_de_confuzie(data_bow,labels,mnb))
#print(f"data de train{data_bow[train_index]} data de test{data_bow[test_index]}")
#KNN
#predictions = KNN(data_bow,labels,test_data_bow)
##Naive Bayes
y_pred = mnb.fit(data_bow, labels).predict(test_data_bow)
#print(accuracy_score(data_test_nou,y_pred))
write_prediction("D:\PITON\KAGGLE_SUBMISSION_3.csv",y_pred,test_data)

