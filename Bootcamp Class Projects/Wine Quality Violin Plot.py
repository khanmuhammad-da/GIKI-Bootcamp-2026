import pandas as pd

wine = pd.read_csv('wineQT.csv')
pd.set_option('display.width',None)

from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.violinplot(data=wine,x='quality',y='alcohol')
#plt.show()

sns.violinplot(data=wine,
               x='quality',
               y='alcohol',
               #palette='pastel',
               inner='box',
               #inner='quartile',
               #inner='point',
               #inner='stick',
               cut =2,
               linewidth=2,
               #density_norm="area",
               density_norm='count',
               bw_adjust =1)
plt.show()

plt.figure(figsize=(10,6))

sns.violinplot(
    x="quality",
    y="alcohol",
    data=wine,
    palette="Set2",
    inner="quart",
    cut=0,
    linewidth=1.5
)

plt.title("Alcohol Distribution by Wine Quality", fontsize=15)
plt.xlabel("Quality")
plt.ylabel("Alcohol (%)")

plt.grid(axis="y", alpha=0.3)

plt.show()

fig,ax = plt.subplots(2,3,figsize=(16,10))
sns.violinplot(x='quality',y='alcohol',data=wine,ax=ax[0,0])
ax[0,0].set_title("Alcohol Distribution by Wine Quality", fontsize=15)

sns.violinplot(x='quality',y='pH',data=wine,ax=ax[0,1])
ax[0,1].set_title("Alcohol Distribution by pH", fontsize=15)

sns.violinplot(x='quality',y='sulphates',data=wine,ax=ax[0,2])
ax[0,2].set_title("Sulphates Distribution by sulphates", fontsize=15)

sns.violinplot(x="quality", y="volatile acidity", data=wine, ax=ax[1,0])
ax[1,0].set_title("Volatile Acidity")

sns.violinplot(x="quality", y="residual sugar", data=wine, ax=ax[1,1])
ax[1,1].set_title("Residual Sugar")

# Hide the unused subplot
ax[1,2].axis("off")

plt.tight_layout()
plt.show()


features =[
    'alcohol',
    'sulphates',
    'pH',
    'volatile acidity',
    'residual sugar',

]
fig,ax = plt.subplots(2,3,figsize=(16,10))
ax = ax.flatten()
for i,feature in enumerate(features):
    sns.violinplot(data=wine,
                   x='quality',
                   y = feature,
                   ax=ax[i])

ax[-1].axis("off")
plt.tight_layout()
plt.show()


import math
import matplotlib.pyplot as plt
import seaborn as sns

features = [col for col in wine.columns if col not in ["quality", "Id"]]

n_cols = 3
n_rows = math.ceil(len(features) / n_cols)

fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
ax = ax.flatten()

for i, feature in enumerate(features):
    sns.violinplot(
        x="quality",
        y=feature,
        data=wine,
        inner="quart",
        cut=0,
        #palette="Set2",
        ax=ax[i]
    )
    ax[i].set_title(feature.title())
    ax[i].grid(axis="y", alpha=0.3)

# Hide any extra subplots
for j in range(len(features), len(ax)):
    ax[j].axis("off")

plt.tight_layout()
plt.show()