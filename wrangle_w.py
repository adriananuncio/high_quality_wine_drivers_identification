#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# wrangle

# In[ ]:

#Imports
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

############ Acquire #####################
def acquire_wine():
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
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])
    return train, val, test  

########### Scaling ################## 
def split_scaled(train, val, test):
    """ This function returns scaled versions of certain features in train/val/test using a MinMax scaler """
    # Features I want to scale
    to_scale= ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
               'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']

    # Features I want to keep unscaled
    not_to_scale = ['quality', 'type_red','type_white']
    
    # Making copies to scale
    train_sc = train.copy()
    val_sc = val.copy()
    test_sc =test.copy()

    # Make the scaler
    minmax= MinMaxScaler()

    # Fit the scaler
    minmax.fit(train[to_scale])
    
    # Make dataframes applying my minmax scaler & setting index
    train_sc = pd.DataFrame(minmax.transform(train[to_scale]), 
    columns = train[to_scale].columns.values).set_index([train.index.values])
    val_sc = pd.DataFrame(minmax.transform(val[to_scale]), 
    columns = val[to_scale].columns.values).set_index([val.index.values])
    test_sc = pd.DataFrame(minmax.transform(test[to_scale]), 
    columns = test[to_scale].columns.values).set_index([test.index.values])
    
    # Concating my scaled sets with my non_scaled
    train_sc = pd.concat([train_sc, train[not_to_scale]], axis=1)
    val_sc = pd.concat([val_sc, val[not_to_scale]], axis=1)
    test_sc = pd.concat([test_sc, test[not_to_scale]], axis=1)

    # (Notes to self:
    # Make sure to put dataframes into brackets before tryng to concat!!!
    # And indicate that you want it on one axis!!!!)
    
    return train_sc, val_sc, test_sc 

############# Statistical Analysis ######

def sulphates_redwhite(train):
    """ This function returns a bar chart that shows how sulphate content in wine affects it's quality,
    controlling for type"""
    ax = plt.axes()
    ax.set_facecolor("bisque")
    sb.barplot(data= train, x='quality', y= 'sulphates', hue='type', color = 'firebrick', ec='grey')
    plt.title(' Sulphate Range in Red and White Wine Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Sulphate Range')
    plt.show()


def stats_q2(train):
    """ This function returns my statistical testing data for quality vs alcohol evaluation"""
    # Correlation for quality and alcohol
    r, p_value = stats.pearsonr(train['alcohol'], train['quality'])
    print(f"Correlation coefficient: {r:.3f}")
    print(f"P-value: {p_value:.3f}")

    # T_test for quality of wines with high and low alcohol content

    high_alcohol = train[train['alcohol'] >= train['alcohol'].mean()]
    low_alcohol = train[train['alcohol'] < train['alcohol'].mean()]

    t, p_value = stats.ttest_ind(high_alcohol['quality'], low_alcohol['quality'], equal_var=False)
    print(f"T-statistic: {t:.3f}")
    print(f"P-value: {p_value:.3f}")

    # Testing for similar significance as  t-test, but more bins/ ANOVA

    bins = [8, 10, 12, 14, 16]
    labels = ['Low', 'Medium', 'High', 'Very high']
    train['alcohol_group'] = pd.cut(train['alcohol'], bins=bins, labels=labels)

    anova = stats.f_oneway(train[train['alcohol_group'] == 'Low']['quality'],
                           train[train['alcohol_group'] == 'Medium']['quality'],
                           train[train['alcohol_group'] == 'High']['quality'],
                           train[train['alcohol_group'] == 'Very high']['quality'])
    print(f"F-statistic: {anova.statistic:.3f}")
    print(f"P-value: {anova.pvalue:.3f}")
  
 ############# Clustering ################
def clusters_sc(df, v1, v2):
    """This function takes in a dataframe and two independent variables from that dataframe and applies     Kmeans with 3 clusters"""
    # Creating df with my 2 independent variables that I want to cluster. No need to scale since I'm using train_sc
    df = train_sc[[v1, v2]]
    
    # Making the thing
    kmeans = KMeans(n_clusters = 3, random_state= 123)
    # Fitting the thing
    kmeans.fit(df)
    # Predicting 
    kmeans.predict(df)
    
    df['cluster_sc'] = kmeans.predict(df)
    
    return df

def cluster_vis(df):
    plt.figure(figsize=[10,5])
    plt.subplot(121)
    plt.title('Clustered % Alcohol x Wine Quality')
    sb.scatterplot(data=df, x='quality', y='alcohol', hue='cluster', palette='rocket')
    plt.subplot(122)
    plt.title('Clustered % Alcohol x Wine Quality')
    sb.barplot(data=df, x='quality', y='alcohol', hue='cluster', palette='rocket')
    plt.tight_layout()
    plt.show()
    
    
def centroid_cluster(df):
    kmean = KMeans(n_clusters=3)
    kmean.fit(df[['quality', 'alcohol']])
    df['cluster'] = kmean.labels_
    centroids = pd.DataFrame(kmean.cluster_centers_, columns=['quality', 'alcohol'])
    colors = ['palevioletred', 'lightsalmon', 'maroon']
    plt.figure(figsize=(14, 9))
    for i, (cluster, subset) in enumerate(df.groupby('cluster')):
        plt.scatter(subset['quality'], subset['alcohol'], label='cluster ' + str(cluster), alpha=.6, c=colors[i])
    centroids.plot.scatter(y='alcohol', x='quality', c='black', marker='x', s=700, ax=plt.gca(), label='centroid')
    plt.legend()
    plt.title('Unscaled Clusters and Cluster Centroids \nfor % Alcohol x Wine Quality')
    plt.xlabel('Wine Quality')
    plt.ylabel('% Alcohol')
    plt.show()

def change_clusters(df, v1, v2):
    """ This function returns a list showing how changing the number of clusters impacts the inertia"""
    inertia= []
    for n in range (2, 10):
        # Making object
        kmc = KMeans(n_clusters = n, random_state= 123 )
        # Fitting object
        kmc.fit(df[[v1, v2]])
        inertia.append(kmc.inertia_)
    # return inertia
    i_results = pd.DataFrame({'n_clusters': list(range(2, 10)),
                                  'inertia': inertia})

    return i_results

def inertia_change(df):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
    pd.Series({k: KMeans(k).fit(df).inertia_ for k in range(2, 12)}).plot(marker='x')
    plt.xticks(range(2, 12))
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.title('Change in inertia as k increases')
    colors = ['palevioletred', 'lightsalmon', 'maroon', 'thistle', 'peru']
    # create a figure with subplots for each value of k
    fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)
    # iterate over the subplots and create a scatter plot for each value of k
    for ax, k in zip(axs.ravel(), range(2, 6)):
        clusters = KMeans(k).fit(df).predict(df)
        ax.scatter(df.quality, df.alcohol, c=[colors[c] for c in clusters])
        ax.set(title='k = {}'.format(k), xlabel='quality', ylabel='alcohol')
    fig.suptitle('Variation in k Cluster for\n % Alcohol x Wine Quality', fontsize=20)
    plt.show()


########### *MODELING* ###############

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
def dt__comp_train_test(X_train, y_train, X_val, y_val): 
    christmas=[]
    for i in range(1,15):
        # make the thing
        tree=DecisionTreeClassifier(max_depth= i, random_state= 123)
        # fit the thing 
        tree.fit(X_train, y_train)
        # use the thing to evaluate model performance
        out_of_sample= tree.score(X_val, y_val)
        in_sample=tree.score(X_train, y_train)
        difference= round((in_sample - out_of_sample) * 100,2)

        #labeling columns for table
        heads= {'max_depth': {i}, 
                'train_accuracy' : in_sample,
                'val_accuracy' : out_of_sample,
                'Percentage Difference' : difference}
        christmas.append(heads)
    willow = pd.DataFrame(christmas)
    return willow

def dtmodel(X_train, y_train, X_val, y_val):   
    tree=DecisionTreeClassifier(max_depth= 6, random_state= 123)
    # fit the thing 
    tree.fit(X_train, y_train)
    # use the thing to evaluate model performance
    out_of_sample= tree.score(X_val, y_val)
    in_sample=tree.score(X_train, y_train)
    difference= round((in_sample - out_of_sample) * 100,2)
    
    labels = pd.DataFrame(data=[{'model': 'DecisionTree(6)',
                   'Train Accuracy': in_sample,
                   'Validate Accuracy': out_of_sample,
                   'Percentage Difference': difference
                   }])
    return labels

def dtmodel2(X_train, y_train, X_val, y_val):   
    tree=DecisionTreeClassifier(max_depth= 4, random_state= 123)
    # fit the thing 
    tree.fit(X_train, y_train)
    # use the thing to evaluate model performance
    out_of_sample= tree.score(X_val, y_val)
    in_sample=tree.score(X_train, y_train)
    difference= round((in_sample - out_of_sample) * 100,2)
    
    labels = pd.DataFrame(data=[{'model': 'DecisionTree(4)',
                   'Train Accuracy': in_sample,
                   'Validate Accuracy': out_of_sample,
                   'Percentage Difference': difference
                   }])
    return labels


def rt_multi_val(X_train, y_train, X_val, y_val):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the max_depth and min_samples
    are changed"""
    little_john=[]
    # set range for max_Depth starting at 1, up to 15, counting by 2
    for i in range(1,15,2):
    # set range forin_samples starting at 3, up to 20, counting by 3
        for x in range(3,20,3):
    # fit a Random Forest classifier
            sherwood = RandomForestClassifier(max_depth= i, min_samples_leaf= x, random_state=123)

            rftestfit = sherwood.fit(X_train, y_train)

    # make predictions on the test set
            rftest_pred = sherwood.predict(X_train)

    # calculate model scores
            val_score = sherwood.score(X_val, y_val)
            train_score= sherwood.score(X_train, y_train)
            difference = round((train_score - val_score) * 100, 2)

            labels = {'max_depth': i,
                           'min_samples_leaf': x,
                           'Train Accuracy': train_score,
                           'Validate Accuracy': val_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            little_john.append(labels)
    return pd.DataFrame(little_john)


def rfmodel(X_train, y_train, X_val, y_val):
# fit a Random Forest classifier
    sherwood = RandomForestClassifier(max_depth= 3, min_samples_leaf= 15, random_state=123)

    rftestfit = sherwood.fit(X_train, y_train)

# make predictions on the test set
    rftest_pred = sherwood.predict(X_train)

# calculate model scores
    val_score = sherwood.score(X_val, y_val)
    train_score= sherwood.score(X_train, y_train)
    difference = round((train_score - val_score) * 100, 2)

    labels = pd.DataFrame(data=[{'model': 'RandomForest',
                   'Train Accuracy': train_score,
                   'Validate Accuracy': val_score,
                   'Percentage Difference': difference
                   }])
    return labels

def knn_multi_val(X_train, y_train, X_val, y_val):
    """This function takes in the train and test datasets and computes their respective
    accuracy scores and the difference between those scores when the number of neighbors is
    changed"""
    wont_you_be=[]
    # set range for n_neighbors starting at 5, up to 25
    for i in range(5,25):
    # fit a KNN classifier
            nextdoor = KNeighborsClassifier(n_neighbors= i)

            nextdoor.fit(X_train, y_train)

    # make predictions on the test set
            y_pred = nextdoor.predict(X_train)

    # calculate model scores
            val_score = nextdoor.score(X_val, y_val)
            train_score= nextdoor.score(X_train, y_train)
            difference = round((train_score - val_score) * 100, 2)

            labels = {'n_neighbors': i,
                           
                           'Train Accuracy': train_score,
                           'Validate Accuracy': val_score,
                           'Percentage Difference': difference
                           }
    # create df that measures train score, test score, and the difference between them
            wont_you_be.append(labels)
    return pd.DataFrame(wont_you_be)


def knn_model(X_train, y_train, X_val, y_val):

    # fit a KNN classifier
    nextdoor = KNeighborsClassifier(n_neighbors= 24)

    nextdoor.fit(X_train, y_train)

# make predictions on the test set
    y_pred = nextdoor.predict(X_train)

# calculate model scores
    val_score = nextdoor.score(X_val, y_val)
    train_score= nextdoor.score(X_train, y_train)
    difference = round((train_score - val_score) * 100, 2)

    labels = pd.DataFrame(data=[{'model': 'KNN',
                   'Train Accuracy': train_score,
                   'Validate Accuracy': val_score,
                   'Percentage Difference': difference
                   }])
    return labels



def americasnexttopmodel(X_train, y_train, X_val, y_val):
    
    ## Decision Tree, 6 depth

    dtmodel1=DecisionTreeClassifier(max_depth= 6, random_state= 123)
    # fit the thing 
    dtmfit = dtmodel1.fit(X_train, y_train['quality'])
    # use the thing to evaluate model performance
    dt1out_of_sample= dtmodel1.score(X_val, y_val['quality'])
    dt1in_sample=dtmodel1.score(X_train, y_train['quality'])
    dt1difference= round((dt1in_sample - dt1out_of_sample) * 100,2)

    # Decision Tree, 4 depth

    dtmodel2=DecisionTreeClassifier(max_depth= 4, random_state= 123)
    # fit the thing 
    dtm2fit = dtmodel2.fit(X_train, y_train['quality'])
    # use the thing to evaluate model performance
    dt2out_of_sample= dtmodel2.score(X_val, y_val['quality'])
    dt2in_sample=dtmodel2.score(X_train, y_train['quality'])
    dt2difference= round((dt2in_sample - dt2out_of_sample) * 100,2)


    # fit a Random Forest classifier
    rfmodel = RandomForestClassifier(max_depth= 3, min_samples_leaf= 15, random_state=123)
    # fit a Random Forest classifier
    rftestfit = rfmodel.fit(X_train, y_train['quality'])
    # make predictions on the test set
    rftest_pred = rfmodel.predict(X_train)
    # calculate model scores
    rtval_score = rfmodel.score(X_val, y_val['quality'])
    rttrain_score= rfmodel.score(X_train, y_train['quality'])
    rfdifference = round((rttrain_score - rtval_score) * 100, 2)

    # KNN

    knnmodel = KNeighborsClassifier(n_neighbors= 24)
    # fit a KNN classifier
    knnfit = knnmodel.fit(X_train, y_train['quality'])
    # make predictions on the test set
    y_pred = knnmodel.predict(X_train)
    # calculate model scores
    knnval_score = knnmodel.score(X_val, y_val['quality'])
    knntrain_score= knnmodel.score(X_train, y_train['quality'])
    knndifference = round((knntrain_score - knnval_score) * 100, 2)

    # Model features to concat

    baseline_acc = round((y_train['quality'] ==6.0).mean(),2)
    namelist= ['Baseline Accuracy','Decision Tree(6)', 'Decision Tree(4)', 'Random Forest', 'KNearest']
    train_acc= [baseline_acc, dt1in_sample, dt2in_sample,rttrain_score, knntrain_score]
    val_acc= ['N/A', dt1out_of_sample, dt2out_of_sample,rtval_score, knnval_score]
    difference= ['N/A', dt1difference,dt2difference,rfdifference,knndifference]

    model_comp2 = pd.DataFrame()

    model_comp2['Model']= namelist
    model_comp2['Train Accuracy']= train_acc
    model_comp2['Validation Accuracy']= val_acc
    model_comp2['Difference']= difference

    
    ax = sb.barplot(data=model_comp2, y= 'Model', x= 'Train Accuracy', palette = "deep")
    ax.bar_label(ax.containers[-1], fmt='Model Score:\n%.2f', label_type='center', c='white')
    plt.title('Model Scores for Classification on Train Data')


    # [{'model':namelist, 'Train Accuracy':train_acc,
    #                             'Validate Accuracy': val_acc, 'Percentage Difference': difference}]
    return model_comp2
