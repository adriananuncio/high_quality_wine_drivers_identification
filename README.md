![wine_project](https://github.com/adriananuncio/high_quality_wine_drivers_identification/assets/122935207/5d892f9a-c254-4e1d-b430-2022fa6712d5)

# You're Wine in a Million
## Using Classification Modeling to Predict the Quality of Vino Verde Wine Variants
Adriana Nuncio, Mack McGlenn

Codeup: O'Neil Cohort

_____________________________________________________________________________________

### Project Overview

This project is designed to identify key drivers in scoring wine quality and predict wine score accuracy for red and white variants of the Portuguese "Vinho Verde" wine.

**Objectives**

- Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter notebook final report.

- Create modules (wrangle.py) that make your process repeateable and your report (notebook) easier to read and follow.

- Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.

- Construct a classification model to predict wine scores with greater accuracy than the baseline.

- Deliver a presentation to share our findings with the stakeholders of the California Wine Institute

_____________________________________________________________________________________

**Deliverables**

- a complete readme.md which walks through the details of the project
- a final report (.ipynb) which concisely displays the finished version of the project
- wrangle.py file to hold all data acquisition, preparation, testing, and modeling functions
- exploratory workbook showing the unpolished groundwork and insights that lead to the final notebook
- link to stakeholder presentation at [canva.com](https://www.canva.com/design/DAFgfDBJnEw/IjxepVNWP28uNtntcINGkw/edit?analyticsCorrelationId=82740c74-ad06-41fa-8b4b-e0dbf78ccb57)


_____________________________________________________________________

#### Outline:
1. Acquire & Prepare Data
2. Exploration/ Correlation Analysis
3. Statistical Analysis
4. Preprocessing
5. Modeling
6. Conclusion

_____________________________________________________________________________________


## 1. Acquire & Prepare

![3F7DF990-5F9F-482F-8E15-AEFC10E47B9B](https://github.com/adriananuncio/high_quality_wine_drivers_identification/assets/122935207/61dcd8c6-93dc-4248-9f60-c508fb57794a)

#### Acquire Actions
- Data acquired as csvs from data.world's Food database as two dataframes for red and white wines, respectively
-  6497 rows x 12 columns total
- Metrics called: fixed_acidity, volatile_acidity, citric_acid, 
  residual_sugar, chloride, free_sulfur_dioxide, total_sulfur_dioxide,
  density, ph, sulphates, alcohol, quality
  
#### Functions called from wrangle.py:
1. wr.acquire_wine()


#### Prepare Actions:
- Add 'type' categories on both dataframes before concating to indicate whether wine is red or white
- Concat dataframes for red and white wine metrics
- Create a variable 'quality_bins' to categorize quality
- Create dummy variables for type
- Keep outliers from dataset to apply anomaly detection
- Split data into train/val/test datasets, stratifying on quality

#### Functions called from wrangle.py:
1. wr.prep_wine(df)

## 2. Exploration/ Correlation Analysis

First, we took a look at the correlation between wine quality and other features. This helped us identify how impactful a feature is in determining wine quality. Overall, alcohol content had the highest correlation at .44. We performed our initial correlation analysis on the combined dataframe, but we decided to re-run the same analysis with the individual datasets for red and white wine. Here's what we found:

High correlation for red wine:
- alchohol, sulphates, citric acids

High correlation for white wine:
- alcohol, pH, sulphates

## 3. Statistical Analysis

**Questions Addressed:**

1. Are higher sulphates more important in determining quality of red wine than in white wine?
2. Do higher quality wines contain more alcohol than mid-quality or low quality wines?
3. Does the quality of a wine's score correlate with whether the wine is a red or a white?

**Methodology:**
1. Are higher sulphates more important in determining quality of red wine than in white wine?

    - Null hypothesis: Higher sulphates are no more important in determining quality of red wine than in white wine
    - Hypothesis: Higher sulphates are more important in determining quality of red wine than in white wine
    Test: mannwhitneyu
    Results:
    - p < 0
    - Reject the null hypothesis
    - It appears that red wines have a consistently higher levels of sulphates across all quality levels. The highest gap of sulphate content appears in higher quality wines.
     
2. Do higher quality wines contain more alcohol than mid-quality or low quality wines?

    - Null Hypothesis: Higher quality wines contain less alcohol than lower or mid-quality wines
    - Hypothesis: Higher quality wines contain more alcohol than lower or mid-quality wines
    Test: T-Test
    Results:
    - p < 0.05
    - We reject the null hypothesis.
    - It appears that higher quality wines have a higher average alcohol content than low quality wines, and a slightly higher alcohol  content than mid-quality wines.
   
   
_____________________________________________________________________________________


### Modeling
We decided to go with classification modeling; after trying both classification and regression, it returned the most significant results. The means used for modeling are Decision Tree, KNearest Neighbors, and RandomForest. We kept all features for modeling except quality_bins and type. 

![BCF6580C-71E1-4B92-9231-0B95B92E95CA](https://github.com/adriananuncio/high_quality_wine_drivers_identification/assets/122935207/56399a7b-45cf-4801-921c-a9dfac307044)

Our baseline was set to a quality score of 6, with a **44% baseline accuracy.** We ran each model with various depths and parameters to determine the best fit.

#### Results: 
The overall best performing model algorithm was Decision Tree with a max depth of 6. It returned an accuracy score of .61 on our train data, and an accuracy score of .54, both of which outperform our baseline accuracy score. It was also the model with the 2nd lowest difference between the train and validation accuracy scores. Using this model, which is based on all of our X_train features, we can predict wine quality with over 15% higher accuracy.

_____________________________________________________________________________________


### Conclusion

Here's what we learned from this project:

Sulphates are a more significant driver of wine quality for red wines than for whites wines. They're also the most significant driver overall, followed by alcohol. During this project, we attempted to use clustering during our analysis, but it proved not to be a significant tool in this case. We were able to improve on our baseline accuracy for determining wine quality by 17%, with our best model being a Decision Tree with a max depth of 6.

As far as future improvements for this project, we believe that the dataset could benefit from adding more features to explore. Specifically, if we knew the regions the wine represented (name/location of vinyard), we would measure the impact that it has on the other features. Overall, we enjoyed the opportunity to work on this project.


_________________________________________________________________________________________________________________________

### Steps to Reproduce

   1. Clone this repo.
   2. Acquire the data from [data.world](https://data.world/food/wine-quality)
   3. Put the data in the file containing the cloned repo.
   4. Run notebook.
