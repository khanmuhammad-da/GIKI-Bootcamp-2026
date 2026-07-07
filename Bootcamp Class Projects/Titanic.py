import matplotlib
import pandas as pd
from narwhals import corr

titanic = pd.read_csv('train_and_test2.csv')
pd.set_option('display.width', None)
#print(titanic.head())
#print(titanic.info())
#print(titanic.describe())

titanic = titanic.drop(columns=['zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6','zero.7','zero.8','zero.9','zero.10','zero.11','zero.12','zero.13','zero.14','zero.15','zero.16','zero.17','zero.18'])
#print(titanic.info())
#print(titanic.describe())

#print(titanic.isnull().sum())
#print(titanic.head())
#print(titanic['Embarked'].value_counts())
titanic['Embarked'] = titanic['Embarked'].fillna(2)
#print(titanic.isnull().sum())
titanic = titanic.rename(columns = {'2urvived':"Survived"})
#print(titanic.head())

#print(titanic.corr(numeric_only=True))

X = titanic[['Pclass', 'Sex', 'Age', 'Fare', 'sibsp', 'Parch', 'Embarked']]
y = titanic['Survived']

from matplotlib import pyplot as plt
titanic.plot(x='Pclass', y='Survived', title = 'line plot')
plt.show()

plt.scatter(titanic['Pclass'], titanic['Survived'])
plt.show()

plt.hist(titanic['Age'], bins=20)
plt.show()

titanic.to_csv('titanic cleaned.csv')
print("done")