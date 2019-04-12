#Environment Setup
import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier

#Ensemble for machine learning
class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2016))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred

#Loading data set
features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slop','ca','thal','pred_attribute']
dataset = pd.read_csv(r"C:\Users\BDO\Desktop\uci-heart-disease\data\processed_cleveland_data.csv",na_values="?", low_memory = False)
dataset.columns=features
dataset["pred_attribute"].replace(inplace=True, value=[1, 1, 1, 1], to_replace=[1, 2, 3, 4])
#selection of features
np_dataset = np.asarray(dataset)
dataset_select = dataset[['age','sex','fbs','thalach','cp','exang','pred_attribute']]

X = dataset_select.iloc[:, :-1].values  
y = dataset_select.iloc[:, -1].values

#my_imputer = SimpleImputer()
#my_imputer = my_imputer.fit(X[:,0:6])   
#X[:, 0:6] = my_imputer.transform(X[:, 0:6])
#print(X)

#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=43)

kfold = KFold(n_splits=10 ,random_state=None, shuffle=False)

gnb = GaussianNB()

lr = LogisticRegression(
    C=0.01,
    penalty='l2',
    dual=True,
    tol=0.0001, 
    fit_intercept=True,
    intercept_scaling=1.0, 
    class_weight=None,
    random_state=43)

'''lr_pg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lr_gscv = GridSearchCV(lr,param_grid=lr_pg, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
lr_gscv.fit(X_train, y_train)'''
#print(X_test)
gnb_lr=VotingClassifier(estimators=[('Guassian Naive Bayes', gnb),('Logistic Regression', lr)], voting='soft', weights=[2,1]).fit(X_train,y_train)
print('The accuracy for Guassian Naive Bayes and Logistic Regression:',gnb_lr.score(X_test,y_test))

m=[]
for i in range(6):
    print(dataset_select.columns[i], end=" ")
    m.append(int(input()))
    if(m[i]<=1):
        m[i]=bool(m[i])
m=np.array(m)
inp=m.reshape(1,-1)
#inp=pd.DataFrame({'age':m[0],'sex':bool(m[1]),'fbs':bool(m[2]),'thalach':m[3],'cp':m[4],'exang':bool(m[5])})
#print(inp)
test_healthy = pd.Series(gnb_lr.predict(inp), name="Healthy")
#print(test_healthy)
a=list(test_healthy)
print(*a)
#print(a[0])
#submission = pd.DataFrame({'Healthy': test_healthy })
#submission.to_csv("Results_df_NSR.csv", index=True)
#print("The person is",submission.iloc[0,0])
