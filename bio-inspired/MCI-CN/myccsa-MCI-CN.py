
# coding: utf-8

# In[ ]:


#import random,math,copy,time
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
#from sklearn import tree
#import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split,cross_val_score
#import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
#from tqdm import tqdm
#import time
from sklearn.model_selection import GridSearchCV


## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/707/original/Good_Job!.wav', autoplay=True))
## Insert whatever audio file you want above


# In[ ]:

print('----reading mci-cn.csv----')
df=pd.read_csv("MCI-CN.csv")



# In[ ]:


#df=df.drop(labels=['Unnamed: 0'],axis=1)
df = df.dropna(how='all')
allDone()



# In[ ]:

print('----reading mci-cn_val.csv----')

val_Data = pd.read_csv("MCI-CN_val.csv")
val_Data = val_Data.dropna(how='all')
#val_Data=val_Data.drop(labels=['Unnamed: 0'],axis=1)
allDone()


features = [i for i in range(0,67600)]


y_train=df['label']
x_train=df[df.columns[1:]]


y_test = val_Data['label']
x_test = val_Data[val_Data.columns[1:]]


x_train.columns=features
x_test.columns=features


# ## WITH GRID-SEARCH WITHOUT BIO-INSPIRED

# In[ ]:

print('----starting svm----')

clf1 = svm.SVC()

svcparams = {'C':[1.0,10,100,1000], 'kernel':['rbf','linear', 'poly'], 'gamma':['auto',0.1,0.01,0.001]}
grid_svc = GridSearchCV(clf1, svcparams,cv=2, verbose=2)
grid_svc.fit(x_train, y_train)
allDone()

grid_svc.predict(x_test)
print(grid_svc.best_score_)
print(grid_svc.best_params_)

grid_svcacc=grid_svc.score(x_test,y_test)
print(grid_svcacc)

allDone()

f=open("MCI-CN.txt",'a')
f.write('SVM='+str(grid_svc.best_params_)+'\n')
f.flush()
f.close()


print("-----------------------------------------------------------------")
print("-----------------------------------------------------------------")

print('--starting knn---')
neigh1 = KNeighborsClassifier()
knnparams = {'n_neighbors':[2,3, 4, 5, 6, 7], 'leaf_size':[10, 20, 30, 40], 'p':[2, 3]}
grid_knn = GridSearchCV(neigh1, knnparams,cv=2, verbose=2)
grid_knn.fit(x_train, y_train)
allDone()

grid_knn.predict(x_test)
print(grid_knn.best_score_)
print(grid_knn.best_params_)


grid_knnacc=grid_knn.score(x_test,y_test)
print(grid_knnacc)

allDone()

f=open("MCI-CN.txt",'a')
f.write('KNN'+str(grid_knn.best_score_)+'\n')
f.flush()
f.close()



