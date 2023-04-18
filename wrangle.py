#!/usr/bin/env python
# coding: utf-8

# wrangle

# In[ ]:

############ Acquire #####################
def acquire_wine(df):
    """ This function will access and concat the red wine and white wine csvs acquired from the data.world dataframes
    for preparation"""
    if os.path.isfile('wine.csv'):
        return pd.read_csv('wine.csv')
    else: 
        # Calling in my dfs from csv link
        red = pd.read_csv('https://query.data.world/s/azffrkwaoqlfrd3srbnuwjp24hvlj4?dws=00000')
        white = df = pd.read_csv('https://query.data.world/s/6ao5pdvepveo2qeeafwdfia6bl5mou?dws=00000')
        
        # Adding 'type' categories on both dataframes before concating
        red['type'] = 'red'
        white['type'] = 'white'
        
        # The two become one
        df = wine = pd.concat([red, white], index=False)
        return df
      
  
############ Prepare #####################
def prep_wine(df):
    """ This function will concat the red wine and white wine csvs acquired from the data.world dataframes
    and prepare them for exploration and analysis"""
    # Creating variable to categorize quality
    df['quality_bins'] = pd.cut(df.quality,[0,5,7,9], labels=['low_quality', 'mid_quality', 'high_quality'])
    # Creating dummy variables for type
    dummies = pd.get_dummies(data=df[['type']], dummy_na= False, drop_first=False)
    df = pd.concat([df, dummies], axis = 1)
    return df
  
############ Split #####################
def split_wine(df, target):
    '''
    take in a DataFrame return train, validate, test split on wine DataFrame.
    '''
# Reminder: I don't need to stratify in regression. I don't remember why, but Madeleine said 
# it
     train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])
    return train, val, test  

 ########### Scaling ################## 
 def split_scaled(train, val, test):
    """ This function returns scaled versions of certain features in train/val/test using a MinMax scaler """
    # Features I want to scale
    to_scale= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
    # Features I want to keep unscaled
    not_to_scale = ['quality', 'type_red',
       'type_white']
    
    # Making copies to scale
    train_sc = train.copy()
    val_sc = val.copy()
    test_sc =test.copy()

    # Make the scaler
    minmax= MinMaxScaler()

    # Fit the scaler
    minmax.fit(train[to_scale])
    
    # Make dataframes applying my minmax scaler & setting index
    train_sc = pd.DataFrame(minmax.transform(train[to_scale]), columns = train[to_scale].columns.values).set_index([train.index.values])
    val_sc = pd.DataFrame(minmax.transform(val[to_scale]), columns = val[to_scale].columns.values).set_index([val.index.values])
    test_sc = pd.DataFrame(minmax.transform(test[to_scale]), columns = test[to_scale].columns.values).set_index([test.index.values])
    
    # Concating my scaled sets with my non_scaled
    train_sc = pd.concat([train_sc, train[not_to_scale]], axis=1)
    val_sc = pd.concat([val_sc, val[not_to_scale]], axis=1)
    test_sc = pd.concat([test_sc, test[not_to_scale]], axis=1)

    # (Notes to self:
    # Make sure to put dataframes into brackets before tryng to concat!!!
    # And indicate that you want it on one axis!!!!)
    
    return train_sc, val_sc, test_sc 
  
############# Clustering ################
def clusters_sc(df, v1, v2):
  """ This function takes in a dataframe and two independent variables from that dataframe and applies Kmeans
  with 3 clusters"""
    # Creating df with my 2 independent variables that I want to cluster. No need to scale since I'm using train_sc
    df = train_sc[[v1, v2]]
    
    # Making the thing
    kmc = KMeans(n_clusters = 3, random_state= 123)
    # Fitting the thing
    kmeans.fit(df)
    # Predicting 
    kmeans.predict(df)
    
    df['cluster_sc'] = kmeans.predict(df)
    
    return df

def change_clusters(df, v1, v2):
    inertia= []
    for n in range (2, 10):
        # Making object
        kmc = KMeans(n_clusters = n, random_state= 123 )
        # Fitting object
        kmc.fit(df[[v1, v2]])
        inertia.append(kmc.inertia_)
    # return inertia
    i_results = pd.DataFrame({'n_clusters': n,
                                  'inertia': inertia})

    return i_results
