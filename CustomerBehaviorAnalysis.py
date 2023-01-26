#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing required dependencies
import pandas as pd
import numpy as np
import seaborn as sns


# In[11]:


# Reading Data
data = pd.read_csv("Customer_Behaviour.csv")
data.shape


# In[12]:


# Displaying top rows of Data
data.head() # To see the feel of the data before diving into analysis


# In[13]:


# Getting info on null objects in the dataset for cleaning
data.info()


# In[15]:


# 5 number summary of all numerical characteristics
data.describe()


# In[16]:


# Finding Correlation between characteristics
data.corr() 


# In[17]:


sns.histplot(data['Age']) # To Check if age distribution to see if the dataset is skewed towards 


# In[8]:


# Filtering people that made a purchase
dp = data[data.Purchased==1]
dp.head()  


# In[9]:


# Age distribution of people who made a purchase
sns.histplot(data=dp,x='Age')


# In[ ]:




