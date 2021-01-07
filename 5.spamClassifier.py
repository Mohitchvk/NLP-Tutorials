import pandas as pd
messages=pd.read_csv('SMSSpamCollection',sep='\t',names=["label","message"])

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

lemmatizer=WordNetLemmatizer()
corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detection_model=MultinomialNB().fit(X_train,y_train)

y_pred=spam_detection_model.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)