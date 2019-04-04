
# coding: utf-8

# In[1]:



from sklearn import tree
import pandas as pd

from sklearn.model_selection import GridSearchCV
## Import up sound alert dependencies
from IPython.display import Audio, display

def allDone():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/707/original/Good_Job!.wav', autoplay=True))
## Insert whatever audio file you want above


# In[3]:

print('train data')
df=pd.read_csv("AD-MCI.csv")
df = df.dropna(how='all')

df=df.drop(columns=['Unnamed: 0'])



print('test data')
val_Data = pd.read_csv("AD-MCI_val.csv")

val_Data = val_Data.dropna(how='all')
val_Data=val_Data.drop(columns=['Unnamed: 0'])
# df=df.reset_index().drop(columns='index')

allDone()


# In[19]:


features = [i for i in range(0,67600)]


y_train=df['label']
x_train=df[df.columns[1:]]


# In[25]:


y_test = val_Data['label']
x_test = val_Data[val_Data.columns[1:]]



x_train.columns=features
x_test.columns=features

print("dtree fitting")
dtree1 = tree.DecisionTreeClassifier()
# criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
dtparams = {'criterion':['gini', 'entropy'], 'splitter':['best','random'], 'max_depth':[30,40,50], 'min_samples_split':[10,20,30,40], 'min_samples_leaf':[4,6,8]}
grid_dt = GridSearchCV(dtree1, dtparams, cv=2,verbose=2)
grid_dt.fit(x_train, y_train)
grid_dt.predict(x_test)
print(grid_dt.best_score_)
print(grid_dt.best_params_)
allDone()

f=open('AD-MCI.txt','a')
f.write('dtree='+str(grid_dt.best_params_))
f.flush()
f.close()