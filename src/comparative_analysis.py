#!/usr/bin/env python3

from LDA_function import *
from LDA_class import *
import nltk
from nltk.corpus import reuters
from plsa import Corpus, Pipeline, Visualize
from plsa.algorithms import PLSA
from plsa.pipeline import DEFAULT_PIPELINE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
nltk.download('reuters')

np.random.seed(1)
ntotal=1000
num_topics=5
documents = reuters.fileids()
documents=np.random.choice(documents,ntotal)
docs=[reuters.raw(d) for d in documents]
labels1=[('earn' in reuters.categories(d)) for d in documents]
chunksize=100
tf_df, id2word = tf(docs)
lil = []
for row in tf_df.values:
    lil_sub = []
    for idx, item in enumerate(row):
        if item:
            lil_sub.append((idx, item))
    lil.append(lil_sub)
gamma_chunk=my_lda_func(corpus=lil, num_topics=num_topics, id2word=id2word, topics_only=False,chunksize=chunksize)[1]
res_lda1=np.concatenate([gamma_chunk[i] for i in np.arange(0,ntotal,chunksize)])
b=LDA2(docs)
res_lda2=b.lda(num_topics=num_topics,conv_threshold=1e-2,max_iter=100,npass=1)
res_lda2=np.array(b.gamma)
pipeline = Pipeline(*DEFAULT_PIPELINE)
corpus_plsa=Corpus(docs,pipeline)
plsa=PLSA(corpus_plsa, num_topics, True)
result = plsa.fit()
res_plsa=result.topic_given_doc
vectorizer = CountVectorizer()
vectorizer.fit_transform(docs)
analyze = vectorizer.build_analyzer()
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
res_bigram=bigram_vectorizer.fit_transform(docs).toarray()
methods='plsi,lda1,lda2,bigram'.split(',')
prop=[0.01,0.05,0.1,0.2,0.3,0.5]
acclist=[]
nmc=19
for l in range(nmc):
    acc=np.zeros((len(prop),len(methods)))
    for j,p in enumerate(prop):
        for i,x in enumerate([res_plsa,res_lda1,res_lda2,res_bigram]):
            X_train, X_test, y_train, y_test = train_test_split(x, np.array(labels1), test_size=1-p, random_state=l)
            clf = svm.SVC(kernel='linear')
            clf.fit(X_train, y_train)
            clf_predictions = clf.predict(X_test)
            acc[j,i]=clf.score(X_test, y_test)
    acclist.append(acc)
acc_all=np.array(acclist)
acc_avg=np.mean(acc_all,axis=0)
acc_sd=np.std(acc_all,axis=0)
df_avg=pd.DataFrame(acc_avg,columns=methods)
df_avg['proportion of training data']=prop
df_sd=pd.DataFrame(acc_sd,columns=methods)
df_sd['proportion of training data']=prop
avg=df_avg.melt(id_vars='proportion of training data')
avg.columns=['proportion of training data','model','avg']
sd=df_sd.melt(id_vars='proportion of training data')
sd.columns=['prop','variable','sd']
df=pd.concat([avg,sd],axis=1)
fig, ax = plt.subplots()
for key, group in df.groupby('model'):
    group.plot('proportion of training data', 'avg', yerr='sd',
        label=key, ax=ax)
plt.ylabel('accuracy')
plt.savefig('comparative_analysis.png', dpi = 1000) # the plot will be saved to the current directory
pass