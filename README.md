# STA663L Final Project Replementation of Latent Dirichlet Allocation

This project re-Implementated the LDA algorithm described in [link to paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?TB_iframe=true&width=370.8&height=658.8) by Blei, et al.    
  Latent Dirichlet Allocation is a statistical model that identifies previously unknown grouping of a collection of discrete data and assigns each member of the collection a mixture of topics. In this paper, we will be focusing on the re-implementation of LDA, and the application of LDA in topic modeling. The application of topic modeling using LDA will classify each text into particular topics with a probability by predefining a fixed number of topics we want to categorize. We report results and speeds in document modeling, comparing our naive implementations with optimized implementation, probabilistic latent semantic indexing (pLSI) model, and Biterm topic model.

## Installation instructions

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package. We recommend to use a vitualenv to install.

```bash
git clone https://github.com/montecristo7/STA663L_Final_Project_Latent_Dirichlet_Allocation.git
pip install .
```


## Directory Tree

```
.
├── example
│   ├── example_LDA_function.py
│   └── sleep_diet_exercise.csv
├── LICENSE
├── README.md
├── setup.py
├── src
│   ├── comparative_analysis.py
│   ├── LDA_class.py
│   ├── LDA_function_optimization_process.py
│   └── LDA_function.py
├── STA663_final_paper.pdf
└── test
    └── test_LDA_function.py
 
```

## Support
The open-source library for unsupervised topic modeling and natural language processing **Gensim** had implemented plain LDA model, and LDA multicore model using CPU cores to parallelize and speed up model training.  
You can find more about the library on :
- [gensim LDA model](https://radimrehurek.com/gensim/models/ldamodel.html)
- [gensim LDA multicore](https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore)

