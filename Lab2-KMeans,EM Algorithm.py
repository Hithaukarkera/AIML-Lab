
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
data = pd.read_csv(r'C:\Users\MCA_PC31\Downloads\driver-data.csv')  
print(data)
X = data.values
em = GaussianMixture(n_components=5)  
em_labels = em.fit_predict(X)

kmeans = KMeans(n_clusters=3)  
kmeans_labels = kmeans.fit_predict(X)

data['EM_Labels'] = em_labels
print("\nEM Labels:")
print(data['EM_Labels'].value_counts())

silhouette_em =silhouette_score(X,em_labels)
silhouette_em =silhouette_score(X,kmeans_labels)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='rainbow')
plt.title("K-Means Clustering")
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='rainbow')
plt.title("EM Clustering")
data['KMeans_Labels'] = kmeans_labels
print("K-Means Labels:")
print(data['KMeans_Labels'].value_counts())

plt.show()