import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('divorce.csv')

df.drop(df.iloc[:, 10:54], inplace=True, axis=1)

from sklearn.model_selection import train_test_split
X = df.drop(columns='Divorce_Y_N')
y = df['Divorce_Y_N']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=10)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

filename = 'divorce-pred-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))