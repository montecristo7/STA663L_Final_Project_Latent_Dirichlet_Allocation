import sys
sys.path.append('../src/')
import nltk
from nltk.corpus import reuters
nltk.download('reuters')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import LDA_function
import pandas as pd


def parse_result(result):
    """
    This function is used to reorganize the result of my_lda_func for plotting.
    """
    result_dic = {}
    for topic_num, dist in result:
        unpack = []
        for obj in dist.split('+'):
            prob, word = obj.split('*')
            unpack.append((float(prob), word.strip().strip('"')))
        prob, word = zip(*unpack)
        result_dic[topic_num] = [prob, word]
    return result_dic





# Simulated Data (Sleep & Vaccine Policy)
sleep = pd.read_csv('sleep_diet_exercise.csv', header=None)
docs = [i[0] for i in sleep.values]

tf_df, id2word = LDA_function.tf(docs)

lil = []
for row in tf_df.values:
    lil_sub = []
    for idx, item in enumerate(row):
        if item:
            lil_sub.append((idx, item))
    lil.append(lil_sub)
    
simu_result = LDA_function.my_lda_func(corpus=lil, num_topics=2, id2word=id2word, num_words=10, chunksize=20, passes=10, verbose=False)
print(simu_result)

# Bar plots for simulated data
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
cmap = ['lightsteelblue', 'pink', 'darkgrey', 'khaki', 'lightsalmon', 'darkseagreen']

simu_dic = parse_result(simu_result)


for idx, ax in enumerate(axs):
    probability, words = simu_dic[idx]

    ax.bar(words, probability, color=cmap[idx])
    ax.set_xlabel("Word")
    ax.set_ylabel("Probability")
    ax.set_title(f"Topic {idx+1}")

plt.tight_layout()

plt.savefig('simulated_data_result.jpg')
print('simulated_data_result figure has been saved!')







# Real-world Data 1: Reuters
np.random.seed(1)
ntotal=1000
documents = reuters.fileids()
documents=np.random.choice(documents,ntotal)
docs=[reuters.raw(d) for d in documents]

tf_df, id2word = LDA_function.tf(docs)

lil = []
for row in tf_df.values:
    lil_sub = []
    for idx, item in enumerate(row):
        if item:
            lil_sub.append((idx, item))
    lil.append(lil_sub)
    
real_result_1 = LDA_function.my_lda_func(corpus=lil, num_topics=4, id2word=id2word, num_words=10, chunksize=20, passes=10)
print(real_result_1)

# Bar plots for real-world data 1
fig, axs = plt.subplots(2, 2, figsize=(17, 7))
cmap = ['lightsteelblue', 'pink', 'darkgrey', 'khaki', 'lightsalmon', 'darkseagreen']

real_result_1_dic = parse_result(real_result_1)

for idx, ax in enumerate(axs.ravel()):
    probability, words = real_result_1_dic[idx]

    ax.bar(words, probability, color=cmap[idx])
    ax.set_xlabel("Word")
    ax.set_ylabel("Probability")
    ax.set_title(f"Topic {idx+1}")

plt.tight_layout()

plt.savefig('reuters_data_result.jpg')
print('reuters_data_result figure has been saved!')










