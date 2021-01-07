# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:21:00 2021

@author: MOHIT CHVK
"""
#data importing and cleaning
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
data=pd.read_csv(r'fake news\train.csv')
data=data.dropna()
titles=[]
y=data.iloc[:,4]
lemmatizer=WordNetLemmatizer()
for i in range(len(data)):
    review=re.sub('[^a-zA-Z]',' ',data.iloc[i,1])
    review=review.lower()
    review=nltk.word_tokenize(review)
    review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    titles.append(review)

#vocab size
vocab_size=5000

#one hot encoding
from tensorflow.keras.preprocessing.text import one_hot
onehot=[one_hot(word,vocab_size) for word in titles]

#padding sentences
sent_len=20
from tensorflow.keras.preprocessing.sequence import pad_sequences
embedded=pad_sequences(onehot,padding='pre',maxlen=sent_len)

#creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
embedding_vector_features=40
model=Sequential()

model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np
X=np.array(embedded)
y=np.array(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

#predict and accuracy scores
y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

model.save(r'fake news\fakeNews.h5')