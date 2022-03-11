#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

## import csc to dataframe
df=pd.read_csv('titanic.csv')
df.head()


# In[3]:


df.info()


# In[7]:


# define chi-square check function
def my_chisquare(df,col1,col2,dp=3.841):
# create contingency table
    df_crosstab = pd.crosstab(df[col1],
                                df[col2],
                               margins=True, margins_name="Total")
    print(df_crosstab)
    df_crosstab = pd.crosstab(df[col2],
                                df[col1],
                               margins=False)
    d1=list(df_crosstab[0])
    d2=list(df_crosstab[1])
    from scipy.stats import chi2_contingency
    # defining the table
    data = [d1, d2]
    stat, p, dof, expected = chi2_contingency(data)
    print('chi-square value is {}'.format(stat))
    print('df is {} and dp is {}'.format(dof,dp))
    # interpret stat vs expected
    if stat > dp:
        print('Dependent (reject H0)')
    else :
        print('Independent (H0 holds true)')
    # interpret p-value
    alpha = 0.05
    print("p value is " + str(p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')


# In[8]:


# A - extracting class info, plot and chi-square

df_class=df.loc[:,['Survived','Pclass']]


# In[19]:


survived_class = df_class[df_class['Survived']==1]['Pclass'].value_counts()
dead_class = df_class[df_class['Survived']==0]['Pclass'].value_counts()
temp_df = pd.DataFrame([survived_class,dead_class])
temp_df.index = ['Survived','Dead']
temp_df.plot(kind='bar',stacked=True, figsize=(10,5), color=['#0072BD','darkblue','#D95319'])
plt.savefig('class_bar.png')


# In[39]:


dp=5.991
my_chisquare(df_class,'Survived','Pclass',dp)


# In[11]:


# B - extracting Gender info, plot and chi-square

df_gender=df.loc[:,['Survived','Sex']]


# In[12]:


survived_gender = df_gender[df_gender['Survived']==1]['Sex'].value_counts()
dead_gender = df_gender[df_gender['Survived']==0]['Sex'].value_counts()
temp_df = pd.DataFrame([survived_gender,dead_gender])

temp_df.index = ['Survived','Dead']

temp_df.plot(kind='bar',stacked=True, figsize=(10,5), color=['#0072BD','darkblue'])
plt.savefig('gender_bar.png')


# In[13]:


survived_gender = df_gender[df_gender['Survived']==1]['Sex'].value_counts()/891
dead_gender = df_gender[df_gender['Survived']==0]['Sex'].value_counts()/891
temp_df = pd.DataFrame([survived_gender,dead_gender])

temp_df.index = ['Survived','Dead']

temp_df.plot(kind='bar',stacked=False, figsize=(10,5), color=['#0072BD','darkblue'])


# In[14]:


dp=3.841
my_chisquare(df_gender,'Survived','Sex',dp)


# In[15]:


# C extract Age info, drop null, plot histogram and perform t test

df2=df.loc[:,['Survived','Age']]
df_Age=df2.dropna()
survived_Age = df_Age[(df_Age['Survived']==1)]['Age']
dead_Age = df_Age[(df_Age['Survived']==0)]['Age']

figure = plt.figure(figsize=(10,5))
plt.hist([survived_Age,dead_Age],bins=40,stacked=True,label=['Survived','Dead'], color=['#0072BD','darkblue'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.savefig('age_hist.png')


# In[17]:


stats.ttest_ind(df_Age['Age'][df_Age['Survived'] == 0],
                df_Age['Age'][df_Age['Survived'] == 1])


# In[16]:


# experiment on seaborn box plot
sns.boxplot(x="Survived", y="Age", data=df_Age)


# In[ ]:




