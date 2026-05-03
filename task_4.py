#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import re # 'regex' module
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[34]:


col = ['target', 'ids', 'date', 'flag', 'user', 'text']
data=pd.read_csv("C:/Users/Hafiz Zuhaib Idrees/.cache/kagglehub/datasets/kazanova/sentiment140/versions/2/training.1600000.processed.noemoticon.csv", names=col, encoding='ISO-8859-1')
data=data.drop(['flag'], axis=1)
data.head()


# In[40]:


# define a function which will handle and clean the text of my dataset
# initialize the tools
stop_word=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()
# function
def clean_dataset(text):
    # clean @mentions
    text=re.sub(r"@[a-zA-Z0-9_]+", " ", text)
    # remove url's and links
    text=re.sub(r"http\s+|www\s+|https\s+", " ", text, flags=re.MULTILINE)
    # remove non letters and keep only words
    text=re.sub(r"[~a-zA-Z\s]", " ", text)
    # Lowercase, Tokenize, Remove Stopwords, and Lemmatize
    words=text.lower().split()
    clean_word=[lemmatizer.lemmatize(w) for w in words if w not in stop_word]
    return " ".join(clean_word)


# In[41]:


# take a random sample of 50000 rows to speed up 
data_sample=data.sample(n=50000, random_state=42)
# apply the function on the text column
print('cleaning the data, please wait...')
data_sample['clean_text']=data_sample['text'].apply(clean_dataset)
print('clean successfully')


# In[45]:


data_sample.clean_text.head()

