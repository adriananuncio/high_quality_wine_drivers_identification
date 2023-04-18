#!/usr/bin/env python
# coding: utf-8

# In[2]:


# standard imports
import numpy as np
import pandas as pd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# models
import visuals


# In[3]:


def qual_x_alc():
    # quality x alcohol for all wine
    plt.figure(figsize=[10,10])
    plt.subplot(321)
    plt.title('Wine Quality x %Alcohol for All Wines')
    sns.barplot(data=train, x='quality_bins', y='alcohol', palette="rocket")
    plt.subplot(322)
    plt.title('Wine Quality x %Alcohol for All Wines')
    sns.boxplot(data=train, x='quality_bins', y='alcohol', palette="rocket")
    # red wine
    plt.subplot(323)
    plt.title('Wine Quality x %Alcohol for Red Wines')
    sns.barplot(data=train[train['type']=='red'], x='quality_bins', y='alcohol', palette='gist_heat')
    plt.subplot(324)
    plt.title('Wine Quality x %Alcohol for Red Wines')
    sns.boxplot(data=train[train['type']=='red'], x='quality_bins', y='alcohol', palette='gist_heat')
    # white wine
    plt.subplot(325)
    plt.title('Wine Quality x %Alcohol for White Wines')
    sns.barplot(data=train[train['type']=='white'], x='quality_bins', y='alcohol', palette='Wistia')
    plt.subplot(326)
    plt.title('Wine Quality x %Alcohol for White Wines')
    sns.boxplot(data=train[train['type']=='white'], x='quality_bins', y='alcohol', palette='Wistia')
    plt.tight_layout()
    plt.show()


# In[ ]:




