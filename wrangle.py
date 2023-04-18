#!/usr/bin/env python
# coding: utf-8

# wrangle

# In[ ]:


# after the free sulfur dioxide outlier has been removed
plt.figure(figsize=[15,15])
plt.subplot(421)
plt.title('Free Sulfur Dioxide Distibution Values \nw/o High FSD Val')
train.free_sulfur_dioxide.hist(color='lightsalmon', alpha = 0.7)
plt.subplot(421)
train[train['type']== 'white'].free_sulfur_dioxide.hist(color='rosybrown', alpha = 0.7)
plt.subplot(421)
train[train['type']== 'red'].free_sulfur_dioxide.hist(color='maroon', alpha = 0.9)
plt.subplot(422)
plt.title('pH Distribution Values \nw/o High FSD Val')
train.pH.hist(color='lightsalmon', alpha = 0.7)
plt.subplot(422)
train[train['type']== 'white'].pH.hist(color='rosybrown', alpha = 0.7)
plt.subplot(422)
train[train['type']== 'red'].pH.hist(color='maroon', alpha = 0.9)
plt.subplot(423)
plt.title('Free Sulfer Dioxide x Wine Quality Distribution Values \nw/o High FSD Val')
sns.boxplot(data=train, x='quality_bins', y='free_sulfur_dioxide', palette='rocket')
plt.subplot(424)
plt.title('pH x Wine Quality Distributions Values \nw/o High FSD Val')
sns.boxplot(data=train, x='quality_bins', y='pH', palette='rocket')
plt.subplot(425)
plt.title('Free Sulfer Dioxide x pH \nw/o High FSD Val')
sns.scatterplot(data=train, x='pH', y='free_sulfur_dioxide', hue='quality_bins', palette='rocket')
plt.subplot(426)
plt.title('Free Sulfur Dioxide x pH \nw/o High FSD Val')
sns.scatterplot(data=train, x='pH', y='free_sulfur_dioxide', hue='type', palette='rocket')
plt.subplot(427)
plt.title('Free Sulfer Dioxide x pH \nw/o High FSD Val')
sns.scatterplot(data=train, x='pH', y='free_sulfur_dioxide', color='chocolate')
plt.tight_layout()
plt.show()

# min and max values for free sulfur dioxide and pH
print(f'Free Sulfur Dioxide Min Value: {train.free_sulfur_dioxide.min()}')
print(f'Free Sulfur Dioxide Max Value: {train.free_sulfur_dioxide.max()}')
print('-'*50)
print(f'pH Min Value: {train.pH.min()}')
print(f'pH Max Value: {train.pH.max()}')

