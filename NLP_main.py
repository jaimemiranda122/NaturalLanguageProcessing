
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


yelp = pd.read_csv('yelp.csv')


# In[3]:


yelp.head()


# In[4]:


yelp.info()


# In[5]:


yelp.describe()


# In[6]:


yelp['text length'] = yelp['text'].apply(len)


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')


# In[8]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')


# In[9]:


sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')


# In[10]:


sns.countplot(x='stars',data=yelp,palette='rainbow')


# In[11]:


stars = yelp.groupby('stars').mean()


# In[12]:


stars


# In[13]:


stars.corr()


# In[14]:


sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# In[15]:


yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[16]:


X = yelp_class['text']
y = yelp_class['stars']


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[18]:


X = cv.fit_transform(X)


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.29,random_state=90)


# In[21]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[22]:


nb.fit(X_train,y_train)


# In[23]:


predictions = nb.predict(X_test)


# In[24]:


from sklearn.metrics import confusion_matrix,classification_report


# In[25]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[26]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[27]:


from sklearn.pipeline import Pipeline


# In[28]:


pipeline = Pipeline([
    ('bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])


# In[29]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=90)


# In[30]:


pipeline.fit(X_train,y_train)


# In[31]:


predictions = pipeline.predict(X_test)


# In[32]:


print(confusion_matrix(y_test,predictions))


# In[33]:


print(classification_report(y_test,predictions))

