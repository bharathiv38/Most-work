#!/usr/bin/env python
# coding: utf-8

# In[1]:


#lets load dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing dataset
df_train=pd.read_csv(r"C:\Users\User\Downloads\train.csv")
df_test=pd.read_csv(r"C:\Users\User\Downloads\test.csv")


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


#merge both train and test data
df=df_train.append(df_test)
df.head()


# In[6]:


#information
df.info()


# In[7]:


df.describe()


# In[8]:


df.drop(['User_ID'],axis=1,inplace=True)  # removing user id as it is not much important


# In[9]:


df.head()


# # Data Pre-processing

# In[10]:


#converting age into numerical
#df['Gender']=pd.get_dummies(df['Gender'],drop_first=1)


# In[11]:


#df.head()


# ### One more way of handling categorical into numerical

# In[12]:


df['Gender'] = df['Gender'].map({'F':0,'M':1})
df.head()


# In[13]:


#handling age
df['Age'].unique()


# In[14]:


# pd.get_dummies(df['Age'],drop_first=True) but this is not correct way
df['Age'] = df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[15]:


df.head()


# In[16]:


## Second technique for AGE
#from sklearn import preprocessing

#label_encoder=preprocessing.LabelEncoder()

#df['Age'] = label_encode.fit_transform(df['Age'])

#df['Age'].unique()


# In[17]:


# fixing categorical city category 
df_city=pd.get_dummies(df['City_Category'],drop_first=True)#if we have three categories two categories is sufficient sodrop=t


# In[18]:


df_city.head()


# In[19]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[20]:


df.drop('City_Category',axis=1,inplace=True)


# In[21]:


df.head()


# ## Checking missing values

# In[22]:


df.isnull().sum()


# In[23]:


# Focus on replacing missing values
df['Product_Category_1'].unique() # here we can see discrete features


# In[24]:


df['Product_Category_2'].unique() # here also we can see discrete features


# In[25]:


df['Product_Category_1'].value_counts()


# In[26]:


df['Product_Category_2'].value_counts()


# In[27]:


df['Product_Category_2'].mode()


# In[28]:


df['Product_Category_2'].mode()[0]


# In[29]:


## Best way to replace discrete missing values is with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[30]:


df['Product_Category_2'].value_counts()


# In[31]:


## Product_Category_3 replace missing values
df['Product_Category_3'].unique()


# In[32]:


df['Product_Category_3'].value_counts()


# In[33]:


df['Product_Category_3'].mode()


# In[34]:


df['Product_Category_3'].mode()[0]


# In[35]:


df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[36]:


df.head()


# In[37]:


df['Stay_In_Current_City_Years'].unique()


# In[38]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+',' ')


# In[39]:


df.head()


# In[40]:


df.info()


# In[41]:


#convert Stay_In_Current_City_Years is an object we have to convert into integer
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df.info()


# In[42]:


df['B'] = df['B'].astype(int)
df['C'] = df['C'].astype(int)


# In[43]:


df.info()


# # Visualization

# In[44]:


#sns.pairplot(df)


# In[45]:


sns.barplot(data=df,x='Age',y='Purchase',hue='Gender')


# Observation:
# 1. Male has purchased more compare to female
# 2. Purchasing range is all equal in all ages

# In[46]:


#visualization of purchase with occupation
sns.barplot(data=df,x='Occupation',y='Purchase',hue='Gender')


# Observation:
# 1. Seems irrespective of occupation purchases are equal

# In[47]:


sns.barplot(data=df,x='Product_Category_1',y='Purchase',hue='Gender')


# In[48]:


sns.barplot(data=df,x='Product_Category_2',y='Purchase',hue='Gender')


# In[49]:


sns.barplot(data=df,x='Product_Category_3',y='Purchase',hue='Gender')


# In[50]:


## Feature Scaling
df_test = df[df['Purchase'].isnull()]


# In[51]:


df_train = df[df['Purchase'].isnull()]


# In[52]:


X = df_train.drop(columns=['Product_ID','Purchase'],axis=1)


# In[53]:


X.head()


# In[54]:


Y = df_train['Purchase']


# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)


# In[56]:


#feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[57]:


from sklearn.linear_model import LinearRegression


# In[ ]:




