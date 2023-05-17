# Ex-06-Feature-Transformation

# AIM :

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION :

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

## BEFORE TRANSFORMATION :

![Screenshot 2023-05-18 000326](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/026b6ff9-8370-4936-9c55-8ec65ee35707)

![Screenshot 2023-05-18 000345](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/1381b257-c371-437f-9124-6bbb5520706a)

![Screenshot 2023-05-18 000547](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/55e368f3-4b98-4f86-bf69-efa43d636f7c)

![Screenshot 2023-05-18 000616](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/985aec93-f1a4-4935-a6c9-3c610f3a9a2c)

## LOG TRANSFORMATION :
 
![Screenshot 2023-05-18 000852](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/7640e2be-1404-45b9-815b-74e1697d6980)

## RECIPROCAL TRANSFORMATION :

![Screenshot 2023-05-18 001016](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/3ea97e42-84ab-478c-9be8-a9d7e6cb067a)

## SQAURE RROT TRANSFORMATION :

![Screenshot 2023-05-18 001114](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/fe4607f9-31d0-4522-a5cf-74dc3d9e2513)

![Screenshot 2023-05-18 001126](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/6d92c687-1818-4be2-8873-e5244cc8fedb)

## POWER TRANSFORMATION :

![Screenshot 2023-05-18 001437](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/b51df7d8-0e69-46c1-a054-3fb5c47aa701)

## QUANTILE TRANSFORMATION :

![Screenshot 2023-05-18 001634](https://github.com/Abrinnisha6/Ex-06-Feature-Transformation/assets/118889454/fa32b257-3062-4636-a3f7-85f144d995f4)

# RESULT:

Thus, Feature transformation is performed and executed successfully for the given dataset.
