# standard imports
import numpy as np
import pandas as pd
import pandas as pd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

def cluster_vis(df):
    plt.figure(figsize=[10,5])
    plt.subplot(121)
    plt.title('Clustered % Alcohol x Wine Quality')
    sns.scatterplot(data=df, x='quality', y='alcohol', hue='cluster', palette='rocket')
    plt.subplot(122)
    plt.title('Clustered % Alcohol x Wine Quality')
    sns.barplot(data=df, x='quality', y='alcohol', hue='cluster', palette='rocket')
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