# ######CLUSTERING#########
# 1. Repeat the above exercise for different values of k
#  - How do the inertia and silhouette scores change?
#  - What if you don't scale your features?
#  - Is there a 'right' k? Why or why not?


# 2. Repeat the following exercise for food nutrients dataset

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# ## iris
# df = pd.read_csv("iris.csv")
# cols = df.columns[:-1]
# X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])

# # train
# k = 4
# kmeans = cluster.KMeans(n_clusters=k)
# kmeans.fit(X_scaled)

# # metrics 
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
# inertia = kmeans.inertia_
# silhouette_score = metrics.silhouette_score(X_scaled, labels, metric='euclidean')
# print("inertia: ", inertia)
# print("silhouette_score: ", silhouette_score)
# # Larger the k, smaller the inertia, larger the silhouette_score. This makes sense based on their definitions.

# # Normalization is necessary, otherwis the distance for different features may be interpreted diffrently.
# # This would be a bias favoring features with small values.

# # the value of k depends on the nature of the question.
# # One could say 3 is a reasonable number because it is the number of names.
# # One could also consider 2 is a reasonable number, since it gives a very good silhouette_score.
# # The choice of k really depends.

# # pairplot
# df['label'] = labels
# cols = df.columns[:-2]
# sns.pairplot(df, x_vars=cols, y_vars= cols, hue='label')
# sns.pairplot(df, x_vars=cols, y_vars= cols, hue='Name')
# plt.show()

## nutrients
df = pd.read_csv("nutrients.txt", delim_whitespace=True)
cols = df.columns[1:]
X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])

# # train
k = 3
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X_scaled)

# # metrics
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
silhouette_score = metrics.silhouette_score(X_scaled, labels, metric='euclidean')
print("inertia: ", inertia)
print("silhouette_score: ", silhouette_score)

df['Label'] = labels
sns.pairplot(df, x_vars=cols, y_vars= cols, hue='Name')
sns.pairplot(df, x_vars=cols, y_vars= cols, hue='Label')
plt.show()