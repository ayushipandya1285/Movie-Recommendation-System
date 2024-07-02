#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[11]:


movies = pd.read_csv('dataset (1).csv')


# In[12]:


movies.head()


# In[13]:


movies.columns


# In[14]:


movies.info()


# In[15]:


movies['tags'] = movies['genre'] + movies['overview']


# In[16]:


movies.head()


# In[17]:


new_df = movies[['id','title','genre','overview','tags']]


# In[18]:


new_df = new_df.drop(columns=['genre','overview'])


# In[19]:


new_df.head()


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer


# In[21]:


cv = CountVectorizer(max_features=10000, stop_words='english')


# In[22]:


cv


# In[23]:


vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray()


# In[24]:


vec


# In[26]:


vec.shape


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


sim = cosine_similarity(vec)


# In[30]:


sim


# In[32]:


new_df[new_df['title']=='The Godfather']


# In[33]:


dist = sorted(list(enumerate(sim[2])),reverse=True, key=lambda vec:vec[1])


# In[34]:


dist


# In[38]:


for i in dist[0:5]:
  print(new_df.iloc[i[0]].title)


# In[46]:


def recommend(movies):
    index = new_df[new_df['title']==movies].index[0]
    distance = sorted(list(enumerate(sim[index])),reverse=True, key=lambda vec:vec[1])
    for i in distance[0:5]:
        print(new_df.iloc[i[0]].title)


# In[47]:


recommend


# In[48]:


recommend("Iron Man")


# In[ ]:




