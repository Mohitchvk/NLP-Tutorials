# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:55:09 2021

@author: MOHIT CHVK
"""
import re
import nltk
import pandas as pd 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
data=pd.read_csv('Data.csv',encoding="ISO-8859-1")

headlines=[]
stemmer=PorterStemmer()

for i in range( len(data)):
    headlines.append(' '.join(str(x) for x in data.iloc[i,2:]))
    
for i in range(len(data)):
    headlines[i]=re.sub('[^a-zA-z]',' ',headlines[i])
    headlines[i]=headlines[i].lower()
    headlines[i]=nltk.word_tokenize(headlines[i])
    headlines[i]=[stemmer.stem(word) for word in headlines[i] if word not in set(stopwords.words('english'))]
    headlines[i]=' '.join(headlines[i])

y=data.iloc[:,1]

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
X=cv.fit_transform(headlines).toarray()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

