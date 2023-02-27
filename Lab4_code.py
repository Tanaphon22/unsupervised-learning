from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from matplotlib import pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
#print(y)

#groups = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
#print(groups .fit_predict(X))

#full_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,distance_threshold=0)
#full_model = full_model.fit()

def plot_dendrogram(X):
    linkage_iris = linkage(X, 'ward')
    dendrogram(linkage_iris)

#plt.title("Agnes Clustering Dendrogram")
#plot_dendrogram(X)
#plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit_predict(X)
print(kmeans)