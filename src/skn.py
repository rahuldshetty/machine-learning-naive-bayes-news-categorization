from nb import *
from nlp import *
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

le = preprocessing.LabelEncoder()
count_vect = CountVectorizer()


df = pd.read_csv('bbc-text.csv')

x = list(df['text'])
y = list(df['category'])

x = process_doc(x)

LIMIT = 95
size = int(len(x) * LIMIT / 100 ) 
x_train = x[:size]
y_train = y[:size]
x_test = x[size:]
y_test = y[size:]

count_vect.fit(x_train)
x_train =  count_vect.transform(x_train)
le.fit(y_train)
y_train = le.transform(y_train) 
clf = MultinomialNB()
clf.fit(x_train, y_train)
x_test = count_vect.transform(x_test)
y_test = le.transform(y_test) 

res = clf.predict(x_test)

count_correct = 0
for i in range(len(res)):
    output = res[i]
    act = y_test[i]
    if output == act:
        count_correct +=1 

acc = 100 * (count_correct/len(res))

print("Accuracy:",acc)
