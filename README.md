# Ex-06-Feature-Transformation

# AIM :

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION :'

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM :

## STEP 1:

Read the given Data

## STEP 2:

Clean the Data Set using Data Cleaning Process

## STEP 3:

Apply Feature Transformation techniques to all the features of the data set

## STEP 4:

Print the transformed features

# PROGRAM :

## DEVELOPED BY : ABRIN NISHA A
## REG NO : 212222230005

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```

# OUTPUT:

## DATA :

![Screenshot 2023-05-17 234557](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/ec468a68-5437-4389-a954-1dfa7df00786)

## df.head() :

![Screenshot 2023-05-17 234916](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/13ca199e-1905-48c4-9b93-4535f9e94b4d)

## df.isnull().sum() :

![Screenshot 2023-05-17 235047](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/d3a10c10-df90-4acf-abf0-2356e0a3077a)

## df.info() :

![Screenshot 2023-05-17 235109](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/44b8242e-2a8c-4f75-b331-b048835a3d35)

## df.describe() : 

![Screenshot 2023-05-17 235227](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/b7b57aea-d6ea-4ee7-a7e6-110d440db24c)

## HIGHLY POSITIVE SKEW :

![Screenshot 2023-05-08 113107](https://user-images.githubusercontent.com/118889454/236745822-e2631dc7-3e75-43b1-a763-cc250ba6e39e.png)

![Screenshot 2023-05-08 113139](https://user-images.githubusercontent.com/118889454/236745919-c87b21e5-e153-44db-af7d-6e87370ebfde.png)

## HIGHLY NEGATIVE SKEW : 

![Screenshot 2023-05-08 113230](https://user-images.githubusercontent.com/118889454/236746020-98e98b8f-c1c1-42a2-a2b0-fce23a6e3843.png)

## MODERATE POSITIVE SKEW :

![Screenshot 2023-05-08 113319](https://user-images.githubusercontent.com/118889454/236746140-b53e1185-45e0-4cbd-8f1e-9fe331609a5b.png)

## MODERATE NEGATIVE SKEW : 

![Screenshot 2023-05-08 113400](https://user-images.githubusercontent.com/118889454/236746253-2aea507b-872a-41de-b1a8-df77d8acfadf.png)

## LOG OF MODERATE POSITIVE SKEW :

![Screenshot 2023-05-08 113447](https://user-images.githubusercontent.com/118889454/236746393-47855ac2-a498-4e96-911c-f9a28219de15.png)

## LOG OF HIGHLY POSITIVE SKEW :

![Screenshot 2023-05-08 113533](https://user-images.githubusercontent.com/118889454/236746516-f8f59943-0bf5-4569-a076-2c9787661b19.png)

## RECIPROCAL OF HIGHLY POSITIVE SKEW : 

![Screenshot 2023-05-08 113620](https://user-images.githubusercontent.com/118889454/236746648-acac828d-b9b9-4302-8da3-1392164c1c80.png)

## SQUARE ROOT TRANSFORMATION :

![Screenshot 2023-05-08 113655](https://user-images.githubusercontent.com/118889454/236746755-4909d2dd-bd10-4547-848a-732afd203a9e.png)

## POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW :

![Screenshot 2023-05-08 113744](https://user-images.githubusercontent.com/118889454/236746892-04f4f145-c7ed-4f76-8430-8e0fed552643.png)

## QUANTILE TRANSFORMATION :

![Screenshot 2023-05-08 113832](https://user-images.githubusercontent.com/118889454/236747022-b9ee1319-5020-4249-8ab4-5d06dba11f3c.png)

# RESULT:

Thus, Feature transformation is performed and executed successfully for the given dataset.
