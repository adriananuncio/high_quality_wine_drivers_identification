# imports
# standard imports
import numpy as np
import pandas as pd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# models
import wine_visuals
# split
from sklearn.model_selection import train_test_split
# stats
from scipy import stats
#warnings
import warnings
warnings.filterwarnings('ignore')

def import_dataframes(a,b):
    '''This function takes in string literal csv files a and b, 
        reads the individual csv files into a dataframe, concatinates 
        those dataframes together, and returns a single combined data frame.
        Print statements verify the combined dataframe has the same value
        counts as a+b'''
    # import red wine data set
    red = pd.read_csv(a)
    # create a columnto indicate red wine
    red['type'] = 'red'
    # import white wine data set
    white = pd.read_csv(b)
    # create a column to indicate white wine
    white['type'] = 'white'
    # concatenate the datasets
    df_combined = pd.concat([red, white])
    # write the combined dataframe to a new CSV file
    df_combined.to_csv('wine_quality_combined.csv', index=False)
    # import the combined dataset as df
    df = pd.read_csv('wine_quality_combined.csv')
    # value count of red wine data set
    a = len(red)
    print(f'Value count of the red wine dataframe: {a}')
    print('_'*50)
    # value count of white wine data set
    b = len(white)
    print(f'Value count of the white wine dataframe: {b}')
    print('_'*50)
    # value count of the combined dataset
    c = len(df)
    print(f'Value count of the combined dataframes: {c}')
    print('_'*50)
    # does the value count of the data sets combined 
    # -- equal the value count of red+white
    d = ((a+b) == c)
    if d == True:
        print(f'The value count of the combined dataframes equal the value counts of red + white: {d}')
        print('_'*50)
    else:
        print(f'The value count of the combined dataframes DOES NOT equal the value counts of red + white: {d}')
        print('_'*50)
    # summarize data/ inital glace at data
    print('_'*50)
    print(f'Data Frame: \n{df.sort_index().head(2).T.to_markdown()}')
    print('_'*50)
    print(f'Shape: \n{df.shape}')
    print('_'*50)
    print(f'Stats: \n{df.describe().T}')
    print('_'*50)
    print('Info: ')
    print(df.info())
    print('_'*50)
    print(f'Data Types: \n{df.dtypes}')
    print('_'*50)
    print(f'Null Values: \n{df.isnull().sum()}')
    print('_'*50)
    print(f'NA Values: \n{df.isna().sum()}')
    print('_'*50)
    print(f'Unique Value Count: \n{df.nunique()}')
    print('_'*50)
    print(f'Columns: \n{df.columns}')
    print('_'*50)
    print(f'Column Value Counts: \n{df.columns.value_counts(dropna=False)}')
    print('_'*50)
    return df

def data_summary(df):
    # summarize data/ inital glace at data
    print('_'*50)
    print(f'Data Frame: \n{df.sort_index().head(2).T.to_markdown()}')
    print('_'*50)
    print(f'Shape: \n{df.shape}')
    print('_'*50)
    print(f'Stats: \n{df.describe().T}')
    print('_'*50)
    print('Info: ')
    print(df.info())
    print('_'*50)
    print(f'Data Types: \n{df.dtypes}')
    print('_'*50)
    print(f'Null Values: \n{df.isnull().sum()}')
    print('_'*50)
    print(f'NA Values: \n{df.isna().sum()}')
    print('_'*50)
    print(f'Unique Value Count: \n{df.nunique()}')
    print('_'*50)
    print(f'Columns: \n{df.columns}')
    print('_'*50)
    print(f'Column Value Counts: \n{df.columns.value_counts(dropna=False)}')
    print('_'*50)

def clean_data(df):
    # change dtype of quality column to float
    df.quality = df.quality.astype(float)
    # format column names in snake case
    df.columns = df.columns.str.replace(' ', '_')
    # binn wine quality
    df['quality_bins'] = pd.cut(df.quality,[0,5,7,9], labels=['low_quality', 'mid_quality', 'high_quality'])
    # initial visuals of df
    vis = wine_visuals.initial_visuals(df)
    # train, test, split df
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.quality)
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train.quality)
    return df, vis, train, val, test

def split_wine(df):
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.quality)
    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train.quality)
    return train, val, test


def hyp_chi2(df, x, y):
    # create cross table of wine quality and alc bins
    observed = pd.crosstab(df[x], df[y])
    # chi2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # set alpha
    α = 0.05
    # hypothesis test for wine quality and % alcohol
    if p < α:
        print(f'We reject the null hypothesis. With p value: {p:.4f}, there is enough evidence to support a statistical relationship between {x} and {y}.')
    else:
        print(f'We fail to reject the null hypothesis. With p value: {p:.4f}, this is not enough evidence to support a statistical relationship between {x} and {y}')
        

def chi_alc_x_qual(df, x, y):
    # create cross table of wine quality and alc bins
    observed = pd.crosstab(df[x], df[y])
    # chi2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # set alpha
    α = 0.05
    if p < α:
        print(f'We reject the null hypothesis. With p value: {p:.4f}, there is enough evidence to support a statistical relationship between {x} and {y}.')
    else:
        print(f'We fail to reject the null hypothesis. With p value: {p:.4f}, this is not enough evidence to support a statistical relationship between {x} and {y}')
    return ((print('\nChi2 performed on % Alcohol x Wine Quality Bins'))
             , (print('-'*50))
             , (print(f'\nAll Wine'))
             , (print('-'*8))
             , (print(hyp_chi2(df, x, y)))
             , (print('='*50))
             , (print(f'\nRed Wine'))
             , (print('-'*8))
             , (print(hyp_chi2(df[df['type']=='red'], x, y)))
             , (print('='*50))
             , (print(f'\nWhite Wine'))
             , (print('-'*8))
             , (print(hyp_chi2(df[df['type']=='white'], x, y))))
 