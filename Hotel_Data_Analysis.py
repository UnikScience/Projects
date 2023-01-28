#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing dependencies needed
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn


# In[2]:


# Reading dataset and printing top 5 rows to get the feel of the data
data = pd.read_csv("hotel_bookings.csv")
data.head()


# In[3]:


# To get the rows and columns present in the data set
data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.drop(['company','is_canceled','reservation_status','reservation_status_date','assigned_room_type','booking_changes',], axis=1, inplace=True) # Dropping null columns and unwanted fields
data = data.dropna(axis=0, subset=['country'])  # Dropping rows where country is null
data = data.fillna(0) # Replaced null values with 0 in children and agent fields
data.isnull().sum()


# In[7]:


sns.countplot(x="hotel",data=data) # Classification classes are imbalanced


# In[8]:


data['hotel'].value_counts()


# In[9]:


SampledData = data.groupby('hotel', group_keys=False).apply(lambda x: x.sample(35000)) #Balacing clases by stratified sampling (Majority Under sampling)


# In[10]:


SampledData.shape


# In[11]:


sns.countplot(x="hotel",data=SampledData)


# In[12]:


SampledData.head()


# In[13]:


sns.countplot(x="hotel",hue="customer_type",data=SampledData)


# In[14]:


SampledData.head(50)


# In[15]:


#Converting string values to categorical
d = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
SampledData.arrival_date_month = SampledData.arrival_date_month.map(d)   

rrt = pd.get_dummies(SampledData['reserved_room_type'])
dt = pd.get_dummies(SampledData['deposit_type'],drop_first=True)
ct = pd.get_dummies(SampledData['customer_type'])
ht = pd.get_dummies(SampledData['hotel'],drop_first=True)
cnty = pd.get_dummies(SampledData['country'])
ml = pd.get_dummies(SampledData['meal'])
ms = pd.get_dummies(SampledData['market_segment'])
dc = pd.get_dummies(SampledData['distribution_channel'])

SampledData = pd.concat([SampledData,rrt,dt,ct,ht,cnty,ml,ms,dc],axis=1)
SampledData = SampledData.drop(['reserved_room_type','deposit_type','customer_type','hotel','country','Undefined','meal','market_segment','distribution_channel'],axis=1)
SampledData.rename(columns={'Resort Hotel': 'hotel'}, inplace=True)


# ## Modeling

# In[35]:


# y is the dependent variable or target variable while X is the independent variables
y = SampledData['hotel']
X = SampledData.drop("hotel",axis=1)
#Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs',max_iter=3000)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)


# In[36]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[37]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[ ]:




