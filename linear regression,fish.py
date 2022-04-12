#!/usr/bin/env python
# coding: utf-8

# In[3]:


#MULTIPLE LINEAR REGRESSION MODEL FOR WEIGHT ESTIMATION FROM MEASUREMENTS OF THE FISH

#The aim of this case study is to estimate weight of the fish individuals from their measurements
#through using linear regression model.
#This study can be improved to use in fish farms. Individual fish swimming in front of the camera can
#be measured from the video image and the weight of the fish can be estimated through the linear
#regression model.


# In[4]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


data = pd.read_csv('/Users/Admin/Downloads/Fish_dataset.csv')
df=data.copy()
df.sample(10)


# In[6]:


df.rename(columns= {'Length1':'LengthVer', 'Length2':'LengthDia', 'Length3':'LengthCro'}, inplace=True)
df.head()


# In[7]:


sp = df['Species'].value_counts()
sp = pd.DataFrame(sp)
sp.T


# In[8]:


# 1.Plot a bar chart showing count of individual species
sns.barplot(x=sp.index, y=sp['Species']);
plt.xlabel('Species')
plt.ylabel('Counts of Species')
plt.show()


# In[9]:


# 2.Identify outliers and remove if any

sns.boxplot(x=df['Weight'])


# In[10]:


dfw = df['Weight']
dfw_Q1 = dfw.quantile(0.25)
dfw_Q3 = dfw.quantile(0.75)
dfw_IQR = dfw_Q3 - dfw_Q1
dfw_lowerend = dfw_Q1 - (1.5 * dfw_IQR)
dfw_upperend = dfw_Q3 + (1.5 * dfw_IQR)


# In[11]:


dfw_outliers = dfw[(dfw < dfw_lowerend) | (dfw > dfw_upperend)]
dfw_outliers


# In[12]:


sns.boxplot(x=df['LengthVer'])


# In[13]:


dflv = df['LengthVer']
dflv_Q1 = dflv.quantile(0.25)
dflv_Q3 = dflv.quantile(0.75)
dflv_IQR = dflv_Q3 - dflv_Q1
dflv_lowerend = dflv_Q1 - (1.5 * dflv_IQR)
dflv_upperend = dflv_Q3 + (1.5 * dflv_IQR)

dflv_outliers = dflv[(dflv < dflv_lowerend) | (dflv > dflv_upperend)]
dflv_outliers


# In[14]:


sns.boxplot(x=df['LengthDia'])


# In[15]:


dfdia = df['LengthDia']
dfdia_Q1 = dfdia.quantile(0.25)
dfdia_Q3 = dfdia.quantile(0.75)
dfdia_IQR = dfdia_Q3 - dfdia_Q1
dfdia_lowerend = dfdia_Q1 - (1.5 * dfdia_IQR)
dfdia_upperend = dfdia_Q3 + (1.5 * dfdia_IQR)

dfdia_outliers = dfdia[(dfdia < dfdia_lowerend) | (dfdia > dfdia_upperend)]
dfdia_outliers


# In[16]:


sns.boxplot(x=df['LengthCro'])


# In[17]:


dfcro = df['LengthCro']
dfcro_Q1 = dfcro.quantile(0.25)
dfcro_Q3 = dfcro.quantile(0.75)
dfcro_IQR = dfcro_Q3 - dfcro_Q1
dfcro_lowerend = dfcro_Q1 - (1.5 * dfcro_IQR)
dfcro_upperend = dfcro_Q3 + (1.5 * dfcro_IQR)

dfcro_outliers = dfcro[(dfcro < dfcro_lowerend) | (dfcro > dfcro_upperend)]
dfcro_outliers


# In[18]:


df1 = df.drop([142,143,144])
df1.describe().T


# In[19]:


sns.boxplot(x=df1['LengthCro'])


# In[20]:


# 3.Build a regression model and print regression equation

# Dependant (Target) Variable:
y = df1['Weight']
# Independant Variables:
X = df1.iloc[:,2:7]


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[22]:


print('How many samples do we have in our test and train datasets?')
print('X_train: ', np.shape(X_train))
print('y_train: ', np.shape(y_train))
print('X_test: ', np.shape(X_test))
print('y_test: ', np.shape(y_test))


# In[23]:


reg = LinearRegression()
reg.fit(X_train,y_train)


# In[24]:


# My model's parameters:
print('Model intercept: ', reg.intercept_)
print('Model coefficients: ', reg.coef_)


# In[25]:


print('y = ' + str('%.2f' % reg.intercept_) + ' + ' + str('%.2f' % reg.coef_[0]) + '*X1 ' + str('%.2f' % reg.coef_[1]) + '*X2 ' +
      str('%.2f' % reg.coef_[2]) + '*X3 + ' + str('%.2f' % reg.coef_[3]) + '*X4 + ' + str('%.2f' % reg.coef_[4]) + '*X5')


# In[26]:


#4.
y_pred = reg.predict(X_test)


# In[27]:


from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(reg, X_train, y_train, cv=5, scoring='r2')
print(cross_val_score_train)


# In[28]:


cross_val_score_train.mean()


# In[29]:


#5.compare real and predicted weights and give a conclusion statement based on it
y_pred1 = pd.DataFrame(y_pred, columns=['Estimated Weight'])
y_pred1.head()


# In[30]:


y_test1 = pd.DataFrame(y_test)
y_test1 = y_test1.reset_index(drop=True)
y_test1.head()


# In[32]:


ynew = pd.concat([y_test1,y_pred1],axis = 1)
ynew


# Conclusion Statement : From the results above,one can see there is a tendency towards errorous estimations when the weight is small 

# In[ ]:





# In[ ]:




