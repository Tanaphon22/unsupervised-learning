import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, decomposition
from sktime.datasets import load_longley
from sklearn.decomposition import PCA

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

#fig = plt.figure(1, figsize=(8, 6))
#plt.clf()
#plt.cla()
#plt.scatter(X[:, 0], X[:, 2])
#plt.show()

#pca = decomposition.PCA(n_components=2)
#pca.fit(X)
#X1 = pca.transform(X)
#print(X1.shape) #(150, 2)
#fig = plt.figure(1, figsize=(8, 6))
#plt.clf()
#plt.cla()
#plt.scatter(X1[:, 0], X1[:, 1])
#plt.show()

#pca = None
#pca = decomposition.PCA(n_components=4)
#pca.fit(X)
#X1 = pca.transform(X)
#var_rat = pca.explained_variance_ratio_
#print(var_rat)
#num_com = pca.n_components
#print(num_com)
#PC_value = np.arange(num_com)
#print(PC_value)

#plt.plot(PC_value, var_rat, 'ro-', linewidth=2)
#plt.title("Scree plot")
#plt.show()

import pandas as pd
glass = pd.read_csv(r"D:\glass.csv")
#print(glass.shape)

X3 = glass[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
#print(X3.shape)

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
plt.cla()
plt.scatter(X3[['RI']], X3[['Na']])
#plt.show()

pca = PCA(n_components=2)
pca.fit(X3)
X4 = pca.transform(X3)
#print(X4)

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
plt.cla()
plt.scatter(X4[:, 0], X4[:, 1])
plt.show()

pca2 = PCA(n_components=5)
pca2.fit(X3)
X5 = pca2.transform(X3)
print(X5.shape)

PC_values = np.arange(pca2.n_components_) + 1
print(PC_values)
plt.plot(PC_values, pca2.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.show()