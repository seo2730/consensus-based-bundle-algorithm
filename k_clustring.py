import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
# plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
# plt.show()

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

k=2
df = Kmean.fit_predict(X)
cluster = np.array([])
print(np.where(df==1))
for i in range(k):
    cluster = np.append(cluster,np.where(df==i))

print(cluster)
    
# plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
# plt.scatter(Kmean.cluster_centers_[0][0], Kmean.cluster_centers_[0][1], s=200, c='g', marker='s')
# plt.scatter(Kmean.cluster_centers_[1][0], Kmean.cluster_centers_[1][1], s=200, c='r', marker='s')
# plt.show()

plt.figure(figsize = (8, 8))
for i in range(k):
    plt.scatter(X[np.where(df==i),0],X[np.where(df==i),1])
plt.show()