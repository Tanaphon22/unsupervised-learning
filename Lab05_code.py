from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:, 0]
y = iris.target

fig, axs = plt.subplots(figsize =(6, 4))
axs.hist(x, bins=10,color='pink')
axs.set_title('Length of sepal')
axs.set_xlabel('cm')
axs.set_ylabel("Frequency")
#plt.show()

for i in range(3):
    axs.hist(x[y == i], bins=10, color=['b','g','r'][i], alpha=0.5, label=iris.target_names[i])
axs.set_title('Length of sepal by class')
axs.set_xlabel('cm')
axs.set_ylabel("Frequency")
axs.legend()
#plt.show()
data = iris.data
print(data.shape)

from sklearn.cluster import DBSCAN
dbscanD = DBSCAN(eps = 0.5, min_samples = 2).fit(data) # fitting the model
labelsD = dbscanD.labels_
print(labelsD)
print(y)