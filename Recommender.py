
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




# In[14]:

ratings,ratings_mean_eachmovie=normalize_ratings(ratings,didrate)


# In[15]:

print ratings                        #full normalize ratings for each movie


# In[16]:

print ratings_mean_eachmovie


# In[17]:

num_users=ratings.shape[1]
num_features=3     #no of features on which recommendation will be based for eg(romance,comedy,action)
print num_features


# In[18]:

movie_features=random.randn(num_movies,num_features)
user_prefs=random.randn(num_users,num_features)
initial_X_and_Theta=r_[movie_features.T.flatten(),user_prefs.T.flatten()]


# In[19]:

print movie_features


# In[20]:

print user_prefs


# In[21]:

initial_X_and_Theta.shape


# In[22]:

def unroll_params(x_and_theta,num_users,num_movies,num_features):
    
    #finding the x and theta values separately

    #get the first (30) values out of 48   
    first_30=x_and_theta[:num_movies * num_features]
  
    #convert the 3*10 matrix into 10*3 matrix  
    x=first_30.reshape((num_features,num_movies)).transpose()
    
    #get the rest (18) values out of 48    
    last_18=x_and_theta[num_movies * num_features :]

    #convert the 3*6 matrix into 6*3 matrix  
    theta=last_18.reshape(num_features,num_users).transpose()

    return x,theta       #return the values (x and theta are as such the x and y parameter of the regression equation)    


# In[36]:

def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta



# In[35]:

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization


# In[25]:

print initial_X_and_Theta


# In[54]:

from scipy import optimize

reg_param=50

minimizedcost_and_optimal_params=optimize.fmin_cg(calculate_cost,fprime=calculate_gradient,x0=initial_X_and_Theta,
                                                 args=(ratings,didrate,num_users,num_movies,num_features,reg_param),
                                                 maxiter=100,disp=True,full_output=True
                                                 )


# In[59]:

cost,optimal_movie_features_and_user_prefs=minimizedcost_and_optimal_params[1], minimizedcost_and_optimal_params[0]


# In[61]:

movie_features,user_prefs=unroll_params(optimal_movie_features_and_user_prefs,num_users,num_movies,num_features)


# In[62]:

print movie_features


# In[63]:

print user_prefs


# In[64]:

all_predictions=movie_features.dot(user_prefs.T)


# In[65]:

print all_predictions


# In[69]:

predictions_for_rajeev=all_predictions[ :,0:1 ] + ratings_mean_eachmovie
print predictions_for_rajeev





