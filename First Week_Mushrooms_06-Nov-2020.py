#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_mushrooms=pd.read_csv("mushrooms.csv")
df_mushrooms.head()


# In[3]:


df_mushrooms.shape


# In[4]:


df_mushrooms.info()


# In[5]:


df_mushrooms.describe(include="all").transpose()


# In[6]:


df_mushrooms=df_mushrooms.drop(["veil-type"],axis=1)


# In[7]:


features=df_mushrooms.columns
target="class"
features=list(features.drop(target))
features


# In[8]:


fig,axs=plt.subplots(nrows=11,ncols=2,figsize=(11,66))
for f, ax in zip(features,axs.ravel()):
    sns.countplot(x=f,hue="class",data=df_mushrooms,ax=ax)


# In[9]:


from sklearn import preprocessing
labelencoder=preprocessing.LabelEncoder()
for col in df_mushrooms.columns:
    df_mushrooms[col]=labelencoder.fit_transform(df_mushrooms[col])


# In[10]:


df_mushrooms.describe().transpose()


# In[11]:


plt.figure(figsize=(14,12))
sns.heatmap(df_mushrooms.corr(),linewidths=.1,cmap="GnBu",annot=True)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_mushrooms.loc[:,features],df_mushrooms.loc[:,target],test_size=0.3,random_state=0)
print ('Train data set', X_train.shape)
print ('Test data set', X_test.shape)


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
LR = LogisticRegression(random_state = 0)
LR.fit(X_train,y_train)


# In[14]:


y_pred = LR.predict(X_test)


# In[15]:


print(classification_report(y_test, y_pred))


# In[16]:


print(confusion_matrix(y_test, y_pred))


# In[17]:


LR_eval = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 5)
LR_eval.mean()


# In[ ]:




