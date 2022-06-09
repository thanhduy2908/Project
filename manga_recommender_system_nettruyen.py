#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('manga_nettruyen_2.csv')
df


# In[3]:


df1= df.iloc[: , 1:]
df1


# In[4]:


len(df['the_loai'].unique())


# In[5]:


df1.info()


# In[6]:


df1["luot_xem"] = df1["luot_xem"].str.replace(".","")


# In[7]:


df1


# In[8]:


df1['so_chuong'] = df1['so_chuong'].astype(int)
df1['luot_xem'] = df1['luot_xem'].astype(int)


# In[9]:


df1


# In[10]:


df1['so_chuong'] = df1.so_chuong.round().astype(int)
df1


# In[11]:


df1['so_chuong'].replace({0 : 1}, inplace=True)


# In[12]:


df1[df1["so_chuong"] == 0 ]


# In[13]:


df1['avg_view'] = df1['luot_xem']/df1['so_chuong']
df1


# In[14]:


df2 = df1.drop(['luot_like','so_binh_luan','luot_xem'],axis = 1)


# In[15]:


df2


# In[16]:


df2.info()


# In[17]:


df2.describe()


# In[18]:


dangtienhanh = df2.loc[df['the_loai'] == 'Đang tiến hành']
len(dangtienhanh)


# In[19]:


dangtienhanh


# In[20]:


df3 = df2.loc[df['the_loai'] != 'Đang tiến hành']
df3


# In[21]:


df3['the_loai'].unique()


# In[22]:


df4 = df3['the_loai'].str.replace(' - ',',')
df4


# In[23]:


list_theloai  = list(df4)
list_theloai


# In[24]:


final_list = []
def to_list(material):
    for i in range(0,19650) :
        sub_list = material[i].split(',')
        final_list.append(sub_list)
    return final_list


# In[25]:


test_list = to_list(list_theloai)
test_list


# In[26]:


df5 = df3.reset_index().iloc[: , 1:]
df5


# In[27]:


df5['genres'] = test_list
df5.drop('the_loai', axis=1, inplace=True)
df5


# In[28]:


df5['name'] = df5['name'].astype(str)
df5['name'] = df5['name'].apply(lambda x: [i.replace("'",'') for i in x])
df5['name'] = df5['name'].apply(lambda x: [i.replace("'",'') for i in x])
df5['name'] = df5['name'].apply(lambda x: [i.replace("[",'') for i in x])
df5['name'] = df5['name'].apply(lambda x: [i.replace("]",'') for i in x])
df5['name'] = df5['name'].apply(lambda x :"".join(x))


# In[29]:


df5


# In[30]:


df6 = df5[df5['avg_view']>=1 ].sort_values('avg_view',ascending=False)
df6


# In[31]:


df_train = df5[['name','genres']]
df_train


# In[32]:


df_train['genres'][0]


# In[33]:


df_train['genres'] = df_train['genres'].apply(lambda x: [i.replace(' ','') for i in x])
df_train['genres'] = df_train['genres'].apply(lambda x: [i.replace('-','') for i in x])
df_train['genres'] = df_train['genres'].apply(lambda x :" ".join(x))
df_train['genres'] = df_train['genres'].apply(lambda x:x.lower())


# In[34]:


df_train['genres'][0]


# In[35]:


df_train['genres'] = df_train['genres'].apply(lambda x :x.removesuffix('đangcậpnhật'))


# In[36]:


df_train['genres'][0]


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer


# In[38]:


ps = PorterStemmer()


# In[39]:


cv = CountVectorizer(max_features = 5000,stop_words = 'english')


# In[40]:


vectors = cv.fit_transform(df_train['genres']).toarray()


# In[41]:


cv.get_feature_names()


# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity = cosine_similarity(vectors)


# In[44]:


sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x:x[1])[1:6]


# In[70]:


def recommend(manga):
    manga_index = df_train[df_train['name'] == manga].index[0]
    distance = similarity[manga_index]
    manga_list = sorted(list(enumerate(distance)),reverse = True,key = lambda x:x[1])[1:6]
    recommend_manga = []
    for i in manga_list :
        sub_index = df_train.iloc[i[0]].name
        print(df_train['name'].iloc[sub_index])
        recommend_manga.append(df_train.iloc[i[0]].name)


# In[71]:


recommend("TÔI ĐÃ CHUYỂN SINH THÀNH SLIME")


# In[ ]:





# In[ ]:




