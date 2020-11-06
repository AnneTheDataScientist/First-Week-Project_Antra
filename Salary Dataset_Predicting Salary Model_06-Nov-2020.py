#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Predicing Salary Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_salary=pd.read_csv("Salary_Data.csv")
df_salary.head()


# In[2]:


sns.heatmap(df_salary.isnull(),annot=True)
plt.show()


# In[3]:


df_salary.isnull().sum()


# In[4]:


from scipy.stats import zscore
z_score=abs(zscore(df_salary))
print(df_salary.shape)
df_salary_final=df_salary.loc[(z_score<3).all(axis=1)]
print(df_salary_final.shape)


# In[5]:


X=df_salary.iloc[:,:-1].values
y=df_salary.iloc[:,:1].values


# In[6]:


sns.distplot(df_salary['YearsExperience'],kde=False,bins=10)


# In[7]:


sns.countplot(y='YearsExperience',data=df_salary)


# In[8]:


sns.barplot(x='YearsExperience',y='Salary',data=df_salary)


# In[9]:


sns.heatmap(df_salary.corr())


# In[10]:


sns.distplot(df_salary.Salary)


# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[13]:


X_train.shape


# In[14]:


y_train.shape


# In[15]:


y_pred=lr.predict(X_test)
y_pred


# In[16]:


plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.title('Salary vs Years of Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()


# In[17]:


plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,lr.predict(X_test),color='red')
plt.title('Salary vs Years of Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary of an employee')
plt.show()


# In[18]:


from sklearn import metrics
print('Mean Absolute Error of the Model:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# In[19]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




