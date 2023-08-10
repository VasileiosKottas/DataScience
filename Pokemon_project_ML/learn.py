#%% imports
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


df = pd.read_csv("pokemon.csv")
# train and test set df
df_train = df[0:800]
df_test = df[0:800]
# train and test set attack, defense
features = ['Total','Sp. Atk']
Class_Y = 'Legendary'
Xtrain = df_train[features]
Xtest = df_test[features]
Ytrain = df_train[Class_Y]
Ytest = df_test[Class_Y]
#%% Logistic Regression

lr = LogisticRegression()
lr.fit(Xtrain, Ytrain)
Y_pred_lr = lr.predict(Xtest)
score_lr = round(accuracy_score(Y_pred_lr, Ytest)*100, 2)

#%% Gaussian Naive Bayes

nb = GaussianNB()
nb.fit(Xtrain, Ytrain)
Y_pred_nb = nb.predict(Xtest)
score_nb = round(accuracy_score(Y_pred_nb, Ytest)*100, 2)

#%% SVM

sv = svm.SVC(kernel = 'linear')
sv.fit(Xtrain, Ytrain)
Y_pred_sv = sv.predict(Xtest)
score_sv = round(accuracy_score(Y_pred_sv, Ytest)*100, 2)


