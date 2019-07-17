#team 1917: pbajpai@mt.iitr.ac.in, shrustikhot12@gmail.com, kirtisethi22@gmail.com

#importing libraries
import pandas as pd
import numpy as np

#reading data
data = pd.read_csv("AC03.csv")

#data cleaning
X = data.iloc[:,0:9]

X.drop('Ratio',axis=1,inplace=True)

Y = data['Follow Back']

#splitting,training and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=123)


#importing logistic regresion model
from sklearn.linear_model import LogisticRegression


#training logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

model.predict(X_test)


#model validation
model.score(X_test,Y_test)


#importing random forest classifer model

from sklearn.ensemble import RandomForestClassifier

#traing random forest classifier model
model2 = RandomForestClassifier(n_estimators=500,max_depth=10)
model2.fit(X_train,Y_train)
model2.predict(X_test)

#model validation
model12.score(X_test,Y_test)



