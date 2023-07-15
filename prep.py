import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split





df1 = pd.read_csv("dataset.csv")


def data_cleaner(Text):
    Text=re.sub(r'[\/`!@#$%^&*()_+{}<>,.?/":;0-9]',' ',Text)
    Text=Text.lower()
    return Text

data=df1.copy()
data["cleaned_data"]=""




data["cleaned_data"]=data["Text"].apply(lambda x:data_cleaner(x))

x=np.array(data["cleaned_data"],)
y=np.array(data["language"])

le=LabelEncoder()
y=le.fit_transform(y)

cv=CountVectorizer()
X=cv.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)


model=MultinomialNB()
model.fit(X_train,y_train)

vec_file = 'vectorizer.pickle'
pickle.dump(cv, open(vec_file, 'wb'))

pickle.dump(le,open("label_encoder",'wb'))


# pickle.dump(model, open('model.pkl', 'wb'))