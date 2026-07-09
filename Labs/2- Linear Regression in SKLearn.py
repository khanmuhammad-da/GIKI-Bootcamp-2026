import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


np.random.seed(42)

#load data
data = pd.read_csv('50_Startups.csv') #dataframe as data

#print(data.head())

#data cleaning
#check for missing values
print(data.isnull().sum())

#check for duplicates
print(data.duplicated().sum())

#EDA
#Distribution of numerical values

plt.figure(figsize=(12,8))
for i, col in enumerate(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']):
    plt.subplot(2,2,i+1)
    sns.histplot(data[col],kde=False)
    plt.title("Distribution of " + col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
a= data.corr(numeric_only=True)
sns.heatmap(a,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            annot_kws={'size': 10},
            )
plt.title('Correlation Matrix')
plt.show()

#scatter plot

plt.figure(figsize=(12, 4))
for i, col in enumerate(['R&D Spend', 'Administration', 'Marketing Spend'], 1):
    plt.subplot(1, 3, i)
    plt.scatter(data[col], data['Profit'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('Profit')
    plt.title(f'{col} vs. Profit')
plt.tight_layout()
plt.show()


#Box plot for data plot
plt.figure(figsize=(12,8))
sns.boxplot(x='State',y='Profit',data=data)
plt.title('Box Plot of Profit')
plt.show()

#preprocessing
x = data.drop('Profit', axis=1)
y = data['Profit']

#define numerical and categorical columns
num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
cat_cols = ['State']

#create a preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols),
    ]
)

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#apply preprocessing
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)
#get features name
feature_names = num_cols + [f'State_{state}' for state in preprocessor.named_transformers_['cat'].categories_[0][:]]


#linear regression
x_train_rd = x_train[['R&D Spend']]
x_test_rd = x_test[['R&D Spend']]
#without normalization
lr1=LinearRegression()
lr1.fit(x_train_rd, y_train)

y_pred_lr1 = lr1.predict(x_test_rd)
print(y_pred_lr1)