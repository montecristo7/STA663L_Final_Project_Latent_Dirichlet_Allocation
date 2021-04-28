#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd
import numpy as np
import re
from scipy.special import psi  # gamma function utils
from pprint import pprint
import gensim.corpora as corpora
from gensim.corpora import Dictionary
import logging
import queue
from numba import jit,njit


# In[190]:


## Utils and Helper Class

def tf(docs):
    """
    This function is used to calculate the document-term matrix and id2word mapping
    """
    # Clean up the text
    docsc_clean = {}
    total_term = []
    for key, val in enumerate(docs):
        val_clean = re.findall(r'[a-z]+', val.lower())
        docsc_clean[f'd{key}'] = val_clean
        total_term += val_clean

    total_term_unique = sorted(set(total_term))
    # change to list
    # id2word = [(idx,word) for  idx, word in enumerate(total_term_unique)]
    id2word = {idx: word for  idx, word in enumerate(total_term_unique)}

    # Count the number of occurrences of term i in document j
    for key, val in docsc_clean.items():
        word_dir = dict.fromkeys(total_term_unique, 0)
        for word in val:
            word_dir[word] += 1
        docsc_clean[key] = word_dir

    tf_df = pd.DataFrame.from_dict(docsc_clean, orient='index')

    return tf_df, id2word


def dirichlet_expectation(sstats):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if len(sstats.shape) == 1:
        return psi(sstats) - psi(np.sum(sstats))
    else:
        return psi(sstats) - psi(np.sum(sstats, 1))[:, np.newaxis]
    
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

class LdaState:
    def __init__(self, eta, shape, dtype=np.float32):
        """
        Parameters
        ----------
        eta : numpy.ndarray
            The prior probabilities assigned to each term.
        shape : tuple of (int, int)
            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).
        dtype : type
            Overrides the numpy array default types.

        """
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def get_lambda(self):
        """Get the parameters of the posterior over the topics, also referred to as "the topics".

        Returns
        -------
        numpy.ndarray
            Parameters of the posterior probability over topics.

        """
        return self.eta + self.sstats

    def get_Elogbeta(self):
        """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        numpy.ndarray
            Posterior probabilities for each topic.
        """
        return dirichlet_expectation(self.get_lambda())

    def blend(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted average for the sufficient statistics.

        The number of documents is stretched in both state objects, so that they are of comparable magnitude.
        This procedure corresponds to the stochastic gradient update from
        `Hoffman et al. :"Online Learning for Latent Dirichlet Allocation"
        <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_, see equations (5) and (9).

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats
        self.numdocs = targetsize


# ## helper functions for my_lda_func

# In[169]:


def initalize(id2word,num_topics,dtype,random_state):
    '''
    initialize all the variables needed for LDA
    '''
    num_terms = len(id2word)

    alpha = np.array( [1.0 / num_topics for i in range(num_topics)], dtype=dtype)

    eta = np.array( [1.0 / num_topics for i in range(num_terms)], dtype=dtype)

    rand  = np.random.RandomState(random_state)

    model_states = LdaState(eta, (num_topics, num_terms), dtype=dtype)
    model_states.sstats = rand.gamma(100., 1. / 100., (num_topics, num_terms))

    expElogbeta = np.exp(dirichlet_expectation(model_states.sstats))
    
    return num_terms,alpha,eta,rand,model_states,expElogbeta


# In[170]:


def e_step_1(rand,chunk,num_topics, dtype,expElogbeta):
    '''
    e step 
    Initialize the variational distribution q(theta|gamma) for the chunk
    '''
    
    gamma = rand.gamma(100., 1. / 100., (len(chunk), num_topics)).astype(dtype, copy=False)
    tmpElogtheta = dirichlet_expectation(gamma)
    tmpexpElogtheta = np.exp(tmpElogtheta)
    sstats = np.zeros_like(expElogbeta, dtype=dtype)
    converged = 0
    
    return gamma,tmpElogtheta,tmpexpElogtheta,sstats,converged


# In[171]:


def e_step_2(chunk,gamma,tmpElogtheta,tmpexpElogtheta,expElogbeta,sstats,converged,dtype,iterations,alpha,gamma_threshold):
    '''
    e step continue
    for each document d, update d's gamma and phi
    '''
    epsilon = 1e-7

    for d, doc in enumerate(chunk):
        ids = [idx for idx, _ in doc]
        cts = np.fromiter([cnt for _, cnt in doc], dtype=dtype, count=len(doc))
        gammad = gamma[d, :]
        Elogthetad = tmpElogtheta[d, :]
        expElogthetad = tmpexpElogtheta[d, :]
        expElogbetad = expElogbeta[:, ids]

        # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
        # phinorm is the normalizer.
        phinorm = np.dot(expElogthetad, expElogbetad) + epsilon

        gammad, expElogthetad,phinorm,converged = e_step_2_inner_update(iterations,gammad,alpha,expElogthetad,cts,phinorm,expElogbetad,gamma_threshold,converged,epsilon)
        
        gamma[d, :] = gammad
        sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)
    return gamma, sstats,converged


# In[172]:


def m_step(model_states,pass_ ,num_updates, chunksize,other):
    '''
    m step
    '''
    previous_Elogbeta = model_states.get_Elogbeta()
    rho = pow(1 + pass_ + (num_updates / chunksize), -0.5)
    model_states.blend(rho, other)

    current_Elogbeta = model_states.get_Elogbeta()
    #Propagate the states topic probabilities to the inner object's attribute.
    expElogbeta = np.exp(current_Elogbeta)

    diff = np.mean(np.abs(previous_Elogbeta.ravel() - current_Elogbeta.ravel()))
    num_updates += other.numdocs
    
    return model_states,num_updates,diff


# In[ ]:





# ## Optimization on the 2 functions below

# In[193]:



def e_step_2_inner_update(iterations,gammad,alpha,expElogthetad,cts,phinorm,expElogbetad,gamma_threshold,converged,epsilon):
    '''
    explicitly updating phi
    '''
    
    for i in range(iterations):
        lastgamma = gammad
        # We represent phi implicitly to save memory and time.
        # Substituting the value of the optimal phi back into
        # the update for gamma gives this update. Cf. Lee&Seung 2001.
        gammad = (alpha + expElogthetad.astype(np.float32) * np.dot(cts.astype(np.float32) / phinorm.astype(np.float32), expElogbetad.T.astype(np.float32)))
        Elogthetad = dirichlet_expectation_numba(gammad)
        expElogthetad = np.exp(Elogthetad)
        phinorm = np.dot(expElogthetad, expElogbetad) + epsilon
        # If gamma hasn't changed much, we're done.
        if np.mean(np.abs(gammad - lastgamma)) < gamma_threshold:
            converged += 1
            break

    return gammad, expElogthetad,phinorm,converged


# In[ ]:





# In[ ]:





# ## Main LDA function

# In[174]:


def my_lda_func(corpus, num_topics, id2word, random_state=10,  passes=1, num_words=10,
                iterations=50, gamma_threshold=0.001, dtype=np.float32,  chunksize=100, topics_only=True, verbose=False):
    
    
    num_terms,alpha,eta,rand,model_states,expElogbeta = initalize(id2word,num_topics,dtype,random_state)

    # Update
    lencorpus = len(corpus)
    chunksize = min(lencorpus, chunksize)
    model_states.numdocs += lencorpus
    num_updates = 0

    for pass_ in range(passes):
        all_chunks = chunks(corpus, chunksize)

        for chunk_no, chunk in enumerate(all_chunks):
            other = LdaState(eta, (num_topics, num_terms), dtype=dtype)
            
            if len(chunk) > 1:
                if verbose:
                    print(f'performing inference on a chunk of {len(chunk) } documents')
            else:
                raise
            # e-step
            gamma,tmpElogtheta,tmpexpElogtheta,sstats,converged = e_step_1(rand,chunk,num_topics, dtype,expElogbeta)

            # e-step-2
            gamma, sstats,converged = e_step_2(chunk,gamma,tmpElogtheta,tmpexpElogtheta,expElogbeta,sstats,converged,dtype,iterations,alpha,gamma_threshold)

            if len(chunk) > 1:
                if verbose:
                    print(f"{converged}/{len(chunk)} documents converged within {iterations} iterations")

            sstats *= expElogbeta

            other.sstats += sstats
            other.numdocs += gamma.shape[0]

            # Do mstep
            if verbose:
                print('Update topics')
            model_states, num_updates,diff = m_step(model_states,pass_ ,num_updates, chunksize,other)
            
            if verbose:
                print("topic diff {}".format(diff))

    shown = []
    topic = model_states.get_lambda()

    for i in range(num_topics):
        topic_ = topic[i]
        topic_ = topic_ / topic_.sum()  # normalize to probability distribution
        bestn = topic_.argsort()[-num_words:][::-1]

        topic_ = [(id2word[id], topic_[id]) for id in bestn]
        topic_ = ' + '.join('%.3f*"%s"' % (v, k) for k, v in topic_)
        shown.append((i, topic_))

    if topics_only:
        return shown
    else:
        return shown,gamma


# In[ ]:





# ### small dataset example

# In[175]:


# Sample data for analysis
d1 = "Java is a language for programming that develops a software for several platforms. A compiled code or bytecode on Java application can run on most of the operating systems including Linux, Mac operating system, and Linux. Most of the syntax of Java is derived from the C++ and C languages."
d2 = "Python supports multiple programming paradigms and comes up with a large standard library, paradigms included are object-oriented, imperative, functional and procedural."
d3 = "Go is typed statically compiled language. It was created by Robert Griesemer, Ken Thompson, and Rob Pike in 2009. This language offers garbage collection, concurrency of CSP-style, memory safety, and structural typing."
d4 = "A young girl when she first visited magical Underland, Alice Kingsleigh (Mia Wasikowska) is now a teenager with no memory of the place -- except in her dreams."
d5 = "Her life takes a turn for the unexpected when, at a garden party for her fiance and herself, she spots a certain white rabbit and tumbles down a hole after him. Reunited with her friends the Mad Hatter (Johnny Depp), the Cheshire Cat and others, Alice learns it is her destiny to end the Red Queen's (Helena Bonham Carter) reign of terror."


# In[180]:


# Using slow version tf_df
tf_df, id2word = tf([d1, d2, d3, d4, d5])

lil = []
for row in tf_df.values:
    lil_sub = []
    for idx, item in enumerate(row):
        if item:
            lil_sub.append((idx, item))
    lil.append(lil_sub)
    
pprint(my_lda_func(corpus=lil, num_topics=2, id2word=id2word, num_words=10))


# In[115]:


get_ipython().run_line_magic('timeit', '-r3 -n2 my_lda_func(corpus=lil, num_topics=2, id2word=id2word, num_words=10)')
# without jit 


# In[181]:


get_ipython().run_line_magic('timeit', '-r3 -n2 my_lda_func(corpus=lil, num_topics=2, id2word=id2word, num_words=10)')
# with jit


# In[ ]:





# ### Real world data (from Tweet)

# In[183]:


# Real world sample data
raw_tweets = pd.read_csv('clean_tweets.csv')

tweets_list = raw_tweets.Tweets.values.tolist()

# Turn the list of string into a list of tokens
clean_tweets = [t.split(',') for t in tweets_list]

len(clean_tweets)


# In[184]:


id2word = Dictionary(clean_tweets)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in clean_tweets]


# In[165]:


my_lda_func(corpus=corpus, num_topics=10, id2word=id2word, num_words=10, chunksize=100)


# In[ ]:





# In[ ]:





# ## Compare with Gensim

# In[37]:


from gensim.models import LdaModel


# In[ ]:


lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=10, 
                   random_state=10,
                   chunksize=100,
#                    alpha='auto',```
#                    per_word_topics=True
                    )


# In[195]:


# original time comparison between plain code and genism's LDA
get_ipython().run_line_magic('timeit', '-r1 -n2 my_lda_func(corpus=corpus, num_topics=10, id2word=id2word, num_words=10,chunksize=100)')
get_ipython().run_line_magic('timeit', '-r1 -n2 LdaModel(corpus=corpus,id2word=id2word,num_topics=10, random_state=10,chunksize=100)')


# In[185]:


# new comparison between optimized and genism's LDA
get_ipython().run_line_magic('timeit', '-r1 -n2 my_lda_func(corpus=corpus, num_topics=10, id2word=id2word, num_words=10,chunksize=100)')
get_ipython().run_line_magic('timeit', '-r1 -n2 LdaModel(corpus=corpus,id2word=id2word,num_topics=10, random_state=10,chunksize=100)')


# ### before optimization stats

# In[196]:


profile = get_ipython().run_line_magic('prun', '-r -q my_lda_func(corpus=corpus, num_topics=10, id2word=id2word, num_words=10,chunksize=100)')
profile.sort_stats('cumtime').print_stats(20)
pass


# ### after optimization stats

# In[192]:


# after optimization
profile = get_ipython().run_line_magic('prun', '-r -q my_lda_func(corpus=corpus, num_topics=10, id2word=id2word, num_words=10,chunksize=100)')
profile.sort_stats('cumtime').print_stats(20)
pass


# In[ ]:




