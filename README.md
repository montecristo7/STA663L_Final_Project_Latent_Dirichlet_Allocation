# STA663L Final Project Replementation of Latent Dirichlet Allocation

This project re-Implementated the LDA algorithm described in https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?TB_iframe=true&width=370.8&height=658.8 by Blei, et al. 
  Latent Dirichlet Allocation is a statistical model that identifies previously unknown grouping of a collection of discrete data and assigns each member of the collection a mixture of topics. In this paper, we will be focusing on the re-implementation of LDA, and the application of LDA in topic modeling. The application of topic modeling using LDA will classify each text into particular topics with a probability by predefining a fixed number of topics we want to categorize. We report results and speeds in document modeling, comparing our naive implementations with optimized implementation, probabilistic latent semantic indexing (pLSI) model, and Biterm topic model.

## Installation instructions

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package.

```bash
-- change later
pip install foobar
```

## Usage

```python
import LDA
my_lda_func(xxxx)
```

# Directory Tree

```
project
│   README.md
│   final report    
│
└───src
│   │   LDA_class.py
|   |   LDA_function.py
│   │   LDAModel_Optimization.py
│   │   comparative_analysis.py
└───test
    │   test_LDA_function.py
    │   
```

.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html
