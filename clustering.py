import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GM


class Clustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k
    
    def clustering_kmeans(self) -> pd.DataFrame:
        X = self.data
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)

        '''
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
        '''
        return self.data
    
    def clustering_hierarchical(self) -> pd.DataFrame:
        X = self.data
        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()

        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)

        '''
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()
        '''
        return self.data


    def clustering_dbscan(self) -> pd.DataFrame:
        data = self.data

        # Plotto i dati
        plt.figure(figsize = [6, 6])
        plt.scatter(data['Coord_1'], data['Coord_2'])
        plt.xlabel('Coord_1', size = 20);
        plt.ylabel('Coord_2', size = 20);

        # Instanzio ed addestro il DBSCAN
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        db = DBSCAN(eps = 3, min_samples = 3)
        db.fit(data)

        # Determino le etichette dei cluster più i punti isolati (-1)
        groups = np.unique(db.labels_)
        Nclusters = len(groups) - (1 if -1 in groups else 0)

        # Costruisco una colormap di tanti colori quanti cluster più i punti isolati (nero)
        colors = plt.cm.Wistia(np.linspace(0, 1, Nclusters))
        colors = np.concatenate([[[0, 0, 0, 1]], colors], axis = 0)

        '''
        # Plotto i dati
        plt.figure(figsize = [6, 6])
        cores_mask = np.zeros_like(db.labels_, dtype = bool)
        cores_mask[db.core_sample_indices_] = True
        for group, color in zip(groups, colors):
        group_mask = (db.labels_ == group)

        # Plotto i punti cores
        X = np.array(data.iloc[cores_mask & group_mask, :])
        plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor = color, markeredgecolor = 'k', markersize = 14)

        # Plotto i punti reachable ed i punti isolati
        X = np.array(data.iloc[~cores_mask & group_mask, :])
        plt.plot(X[:, 0], X[:, 1], 'o', markerfacecolor = color, markeredgecolor = 'k', markersize = 6)

        plt.title(f'# cluster: {Nclusters}', size = 20)
        plt.xlabel('Coord_1', size = 20);
        plt.ylabel('Coord_2', size = 20);
        '''
        return self.data
    
    def clustering_expectationMaximisation(self) -> pd.DataFrame:
        X = self.data  # Utilizza il dataset definito nell'oggetto self
        
        # Plotto i dati
        plt.figure(figsize = [6, 6])
        plt.scatter(X[:, 0], X[:, 1], cmap = 'brg')  # Assumi che self.data non abbia etichette
        plt.xlabel('x', size = 20)
        plt.ylabel('y', size = 20)

        # Instanzio ed addestro l'EM
        em = GM(n_components = 3, n_init = 10)
        em.fit(X)
        clusters = em.predict(X)

        '''
        # Plotto i dati
        plt.figure(figsize = [6, 6])
        plt.scatter(X[:, 0], X[:, 1], c = labels_true, cmap = 'brg')
        plt.scatter(X[:, 0], X[:, 1], s = 10, marker = '*', c = clusters, cmap = 'cool')
        plt.xlabel('x', size = 20);
        plt.ylabel('y', size = 20);

        # Plotto le linee di livello
        x = np.linspace(-4, 7)
        y = np.linspace(-8, 5)
        xx, yy = np.meshgrid(x, y)
        MM = np.array([xx.ravel(), yy.ravel()]).T
        zz = -em.score_samples(MM)
        zz = zz.reshape(xx.shape)
        CS = plt.contour(xx, yy, zz, levels = np.logspace(0, 1, 15))
        '''
        return self.data