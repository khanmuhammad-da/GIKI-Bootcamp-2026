import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris=pd.read_csv('iris.csv')
pd.set_option('display.width', None)
#print(iris.head())
#print(iris.tail())
#print(iris.describe())
#print(iris.info())

#check if there is any null value
#print(iris.isnull().sum())

#drop any duplicate value
iris = iris.drop_duplicates()

#print(iris.info())

#sort by id i.e, column[0]

iris = iris.sort_values(by=iris.columns[0])
#print(iris.head())
#print(iris.tail())
#print(iris.describe())

iris.corr(numeric_only=True)
iris.plot(x=iris.columns[0],y=iris.columns[1],title = 'Line Plot')
plt.show()

iris.plot(kind = 'scatter', x= iris.columns[0], y= iris.columns[1])
plt.show()

iris[iris.columns[0]].plot(kind='hist')
plt.show()

iris.head().plot(kind='bar',x=iris.columns[0])
plt.show()

if "species" in iris.columns:
    iris["species"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.show()
iris.boxplot()
plt.show()

fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(
    iris['SepalLengthCm'],
    iris['SepalWidthCm'],
    iris['PetalLengthCm']
)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")
plt.show()

x=iris.iloc[:,0:4]
print(x.head())
y=iris.iloc[:,4]
print(y.head())

iris.to_csv('iris_cleaned.csv',index = False)
print("done")