#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
#%%
##read the orignal text file

#file = open("tt.txt", "r")
#file2 = open("ww.txt","w")
#filedata = file.readlines()
#article = filedata[0].split(". ")
#"""
#for w in article:
#    file2.write(w)
#    file2.write("\n")
#file2.close()
#print(filedata)
#print(article)
#print("\n-------------------\n")
#"""
#sentences = []
#for sentence in article:
#    print(sentence)
#    sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
#    sentences.append(re.sub('[^a-zA-Z]',' ', sentence).split(' '))
#sentences.pop()
#stopwords = stopwords.words('english')
##
####read_article()


#%%

#def sentence_similarity(sent1, sent2, stopwords=None):
sent1=['For', 'years,', 'Facebook', 'gave', 'some', 'of', 'the', "world's", 'largest', 'technology', 'companies', 'more', 'intrusive', 'access', 'to', "users'", 'personal', 'data', 'than', 'it', 'has', 'disclosed,', 'effectively', 'exempting', 'those', 'business', 'partners', 'from', 'its', 'usual', 'privacy', 'rules,', 'according', 'to', 'internal', 'records', 'and', 'interviews']
sent2=['The', 'special', 'arrangements', 'are', 'detailed', 'in', 'hundreds', 'of', 'pages', 'of', 'Facebook', 'documents', 'obtained', 'by', 'The', 'New', 'York', 'Times']
nltk.download("stopwords")
stopwords = stopwords.words('english')
if stopwords is None:
    stopwords = []
sent1 = [w.lower() for w in sent1]
sent2 = [w.lower() for w in sent2]
 
all_words = list(set(sent1 + sent2))
 
vector1 = [0] * len(all_words)
vector2 = [0] * len(all_words)
 
# build the vector for the first sentence
for w in sent1:
    if w in stopwords:
        continue
    vector1[all_words.index(w)] += 1
 
# build the vector for the second sentence
for w in sent2:
    if w in stopwords:
        continue
    vector2[all_words.index(w)] += 1
 
##return 1 - cosine_distance(vector1, vector2)
#
##sentence_similarity(sent1,sent2,stop_words)

#%%
#create summary from the matrix
