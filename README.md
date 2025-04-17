# Query Expansion for Information Retrieval System

This project implements query expansion techniques to enhance search results in an Information Retrieval (IR) system. It uses the 20 Newsgroups dataset for testing the functionality and applies a TF-IDF-based search engine with WordNet-based query expansion.

## Prerequisites

Before running the application, make sure to install the following dependencies:

### 1. **Python Version**
Make sure you're using Python 3.8+.

### 2. **Conda Environment Setup**

Create a new Conda environment:

```bash
conda create -n query_expansion_ir python=3.8
conda activate query_expansion_ir





pip install scikit-learn nltk numpy
pip install gensim







run this particular in python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

back to terminal

python -m nltk.downloader punkt stopwords wordnet



python app.py



requirements.txt --->


scikit-learn==0.24.2
nltk==3.6.3
numpy==1.21.2
gensim==4.1.2
