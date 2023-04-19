#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def clusters(df, v1, v2):
    """ This function takes in a dataframe and two 
    independent variables from that dataframe and applies 
    Kmeans with 3 clusters"""
    # Creating df with my 2 independent variables that I want to cluster
    dataframe = df[[v1, v2]]
    # Making the thing
    kmeans = KMeans(n_clusters = 3, random_state= 123)
    # Fitting the thing
    kmeans.fit(dataframe)
    # Predicting 
    kmeans.predict(dataframe)
    # add cluster predictions to df as a new column
    dataframe['cluster'] = kmeans.predict(dataframe)
    return dataframe, kmeans


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




