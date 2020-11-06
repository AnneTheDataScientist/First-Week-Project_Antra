#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_wine=pd.read_csv("wine_dataset.csv")
df_wine.head()


# In[2]:


sns.heatmap(df_wine.isnull(),annot=True)
plt.show()


# In[3]:


df_wine.isnull().sum()


# In[4]:


from scipy.stats import zscore
z_score=abs(zscore(df_wine))
print(df_wine.shape)
df_wine_final=df_wine.loc[(z_score<3).all(axis=1)]
print(df_wine_final.shape)


# In[5]:


df_wine.skew()


# In[6]:


df_wine.plot.scatter(x='Alcohol',y='Class')


# In[7]:


sns.lineplot(x="Malic acid",y="Class",data=df_wine)


# In[8]:


Y=df_wine.iloc[:,0].values
X=df_wine.iloc[:,1:13].values


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1)


# In[10]:


df_wine.describe()


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
ppn = Perceptron(max_iter = 1500, eta0 = 0.1, random_state = 1)
ppn.fit(X_train, Y_train)
y_pred = ppn.predict(X_test)
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
print("\nConfusion matrix: \n" + str(confusion_matrix(Y_test, y_pred)))
print("\nClassification report: \n" + str(classification_report(Y_test, y_pred)))


# In[13]:


from sklearn.linear_model import LogisticRegression
ppn = LogisticRegression(C = 100, max_iter = 1500, random_state = 1)
ppn.fit(X_train, Y_train)
y_pred = ppn.predict(X_test)
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
print("\nConfusion matrix: \n" + str(confusion_matrix(Y_test, y_pred)))
print("\nClassification report: \n" + str(classification_report(Y_test, y_pred)))


# In[14]:


from sklearn.tree import DecisionTreeClassifier
ppn = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 1)
ppn.fit(X_train, Y_train)
y_pred = ppn.predict(X_test)
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
print("\nConfusion matrix: \n" + str(confusion_matrix(Y_test, y_pred)))
print("\nClassification report: \n" + str(classification_report(Y_test, y_pred)))


# In[15]:


from sklearn.svm import SVC
ppn = SVC(C= 10000, kernel = 'rbf', degree = 3)
ppn.fit(X_train, Y_train)
y_pred = ppn.predict(X_test)
print("Accuracy score: " + str(accuracy_score(Y_test, y_pred)))
print("\nConfusion matrix: \n" + str(confusion_matrix(Y_test, y_pred)))
print("\nClassification report: \n" + str(classification_report(Y_test, y_pred)))


# In[ ]:




