
# coding: utf-8

# In[1]:

from numpy import *
num_movies=10
num_users=5
ratings=random.randint(11,size=(num_movies,num_users))
print ratings


# In[2]:

didrate =(ratings!=0)*1
print didrate


# In[3]:

rajeev_ratings= zeros((num_movies,1))
print rajeev_ratings


# In[4]:

rajeev_ratings[6]=9
rajeev_ratings[5]=7
rajeev_ratings[2]=4
rajeev_ratings[9]=3
print rajeev_ratings


# In[5]:

ratings=append(rajeev_ratings,ratings,axis=1)
didrate=append(((rajeev_ratings!=0)*1),didrate,axis=1)
print ratings
print didrate


# In[6]:

a=[10,20,30]
avg=mean(a)
print avg


# In[7]:

a=[10-avg,20-avg,30-avg]
print a


# In[19]:

i=2
print didrate[i]


# In[20]:

idx = where(didrate[i] == 1)[0]
print idx


# In[21]:

ratings_mean=zeros(shape=(10,1))
ratings_mean[i] = mean(ratings[i, idx])
print ratings_mean[i]


# In[22]:

ratings_norm=zeros(shape=ratings.shape)
ratings_norm[i,idx]=ratings[i,idx]-ratings_mean[i]
print ratings_norm[i]


# In[ ]:

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean



# In[ ]:

ratings,ratings_mean=normalize_ratings(ratings,didrate)



