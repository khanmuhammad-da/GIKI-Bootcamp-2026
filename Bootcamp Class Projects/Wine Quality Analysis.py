import numpy as np
import pandas as pd

wine = pd.read_csv('wineQT.csv')
pd.set_option('display.width',None)
#print(wine.head())
#print(wine.info())
#print(wine.describe())
#print(wine.isnull().sum())

a= wine.corr(numeric_only=True)
x= wine.drop(columns=['quality','Id'])
y=wine['quality']

from matplotlib import pyplot as plt
import seaborn as sns
# plt.figure(figsize=(12,8))
# sns.heatmap(
#     a,
#     annot=True,
#     #cmap='viridis',
#     #cmap='coolwarm',
#     cmap = 'RdYlBu',
#     vmin = -1,
#     vmax = 1,
#     center = 0,
#     fmt='.2f',
#     linewidths=.5,
# )
# #plt.show()
#
#
# mask=np.triu(np.ones_like(a,dtype=bool))
# plt.figure(figsize=(12,8))
# sns.set_theme(style="darkgrid")
# #sns.set_theme(style="white")
# #sns.set_theme(style="dark")
# #sns.set_theme(style="ticks")
# sns.heatmap(
#     a,
#     mask=mask,
#     annot=True,
#     cmap = 'RdYlBu',
#     vmin = -1,
#     vmax = 1,
#     center = 0,
#     fmt='.2f',
#     linewidths=.5,
#
# )
# plt.tight_layout()
# #plt.show()
# #print(wine.info())

#histogram
# plt.figure(figsize=(10,10))
# plt.hist(
#     wine['alcohol'],
#     bins=10,
#     color='green',
#     edgecolor='black',
#     linewidth=2,
#     alpha=0.5,
#     #histtype='bar'
#     # histtype='stepfilled',
#     # histtype='step',
#     histtype='barstacked',
#     density=False,
#     # cumulative=True,
#     # orientation='horizontal',
#     hatch = "/"
# )
# plt.title("Alcohol Distribution")
#
# plt.xlabel("Alcohol (%)")
#
# plt.ylabel("Frequency")
# plt.show()

import matplotlib.pyplot as plt

# plt.figure(figsize=(12,6))
#
# plt.hist(
#     wine["alcohol"],
#     bins=20,
#     color="royalblue",
#     edgecolor="black",
#     linewidth=1.5,
#     alpha=0.8,
#     rwidth=0.9
# )
#
# plt.title("Distribution of Alcohol Content", fontsize=16)
#
# plt.xlabel("Alcohol (%)", fontsize=12)
#
# plt.ylabel("Frequency", fontsize=12)
#
# plt.grid(axis="y", alpha=0.3)
#
# plt.tight_layout()
#
# plt.show()
#
# plt.figure(figsize=(10,6))
#
# plt.hist(wine["alcohol"],
#          bins=20,
#          alpha=0.5,
#          label="Alcohol")
#
# plt.hist(wine["pH"],
#          bins=20,
#          alpha=0.5,
#          label="pH")
#
# plt.legend()
#
# plt.show()
#
# features = wine.columns[:-1]
#
# wine[features].hist(
#     figsize=(15,12),
#     bins = 20
# )
# plt.tight_layout()
# plt.show()
plt.figure(figsize=(8,5))
plt.boxplot(
    wine['alcohol'],
    # vert=False
    showmeans=True,
    meanprops={
        'marker':'D',
        'markerfacecolor':'red',
        'markersize':8
    },
    patch_artist = True,
    boxprops={
        'facecolor':'lightblue',
    },
    medianprops={
        'color':'red',
        'linewidth':3
    },
    whiskerprops={
        'color':'green',
        'linewidth':3
    },
    capprops={
        'color':'black',
        'linewidth':2
    },
    flierprops={
        'marker':'o',
        'markerfacecolor':'red',
        'markersize':8
    },
    notch=True,
    showfliers=False,
    whis=2
)
plt.title('Alcohol Distribution')
plt.ylabel('Alcohol')
plt.show()

plt.figure(figsize=(10,6))
plt.boxplot([
    wine['alcohol'],
    wine['pH'],
    wine['density']
])

plt.xticks(
    [1,2,3],
    ['alcohol','pH','density']
)
plt.show()

plt.figure(figsize=(15,8))

wine.drop(columns="Id").boxplot()

plt.xticks(rotation=45)

plt.show()

plt.figure(figsize=(12,6))

plt.boxplot(
    wine["alcohol"],
    patch_artist=True,
    showmeans=True,
    notch=True,
    boxprops={"facecolor":"skyblue"},
    medianprops={"color":"red","linewidth":2},
    meanprops={"marker":"D","markerfacecolor":"black"},
    flierprops={"marker":"o","markerfacecolor":"orange"}
)

plt.title("Alcohol Box Plot")

plt.ylabel("Alcohol (%)")

plt.grid(axis="y", alpha=0.3)

plt.show()