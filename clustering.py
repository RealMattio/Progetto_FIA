import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as GM
from sklearn.cluster import AgglomerativeClustering
from kmodes.kmodes import KModes


class Clustering:
    def __init__(self, data, n_cluster, clustering_model = 'kmodes', dbscan_eps = 3, dbscan_min_samples = 15, GM_n_init = 10):
        '''
        Costruttore della classe Clustering
        :param data: dataset da clusterizzare, già preprocessato e organizzato rispetto al modello di clustering che si intendere utilizzare
        :param n_cluster: numero di cluster da identificare
        :param clustering_model: modello di clustering da utilizzare, possibili valori sono 'kmodes'. 'kmeans', 'hc', 'dbscan', 'em' (default: 'kmodes')
        :param dbscan_eps: valore di epsilon da utilizzare nel DBSCAN (default: 3)
        :param dbscan_min_samples: numero minimo di campioni per formare un cluster nel DBSCAN (default: 15)
        :param GM_n_init: numero di inizializzazioni da utilizzare nell'EM (default: 10)
        '''
        self.data_to_cluster = data
        self.n_cluster = n_cluster
        self.clustering_model = clustering_model
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.GM_n_init = GM_n_init
        self.cluster = None
    
    def clustering_kmeans(self) -> pd.DataFrame:
        X = self.data_to_cluster
        kmeans = KMeans(n_clusters = self.n_cluster, init = 'random', random_state = 42)
        #kmeans = KMeans(n_clusters = self.n_cluster, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)
        self.cluster = pd.DataFrame({'Cluster_Kmeans': y_kmeans})
    
    def clustering_hierarchical(self) -> pd.DataFrame:
        X = self.data
        '''
        n_components = math.ceil(len(X.columns)/4)
        pca = PCA(n_components = n_components) 
        X = pca.fit_transform(X)
        print(f'Original: {X.shape}')
        print(f'PCA: {df_reduced.shape}')
        '''
        hc = AgglomerativeClustering(n_clusters = self.n_cluster, metric = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        self.cluster = pd.DataFrame({'Cluster_HC': y_hc})


    def clustering_dbscan(self) -> pd.DataFrame:
        data = self.data_to_cluster
        # Instanzio ed addestro il DBSCAN
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        db = DBSCAN(eps = self.dbscan_eps, min_samples = self.dbscan_min_samples)
        db.fit(data)
        y_db = db.fit_predict(data)
        self.cluster = pd.DataFrame({'Cluster_DBSCAN': y_db})
    

    def clustering_expectationMaximisation(self, GM_n_init = 2) -> pd.DataFrame:
        X = self.data_to_cluster  # Utilizza il dataset definito nell'oggetto self
        self.GM_n_init = GM_n_init
        # Instanzio ed addestro l'EM
        em = GM(n_components = self.n_cluster , n_init = self.GM_n_init)
        em.fit(X)
        clusters = em.predict(X)
        self.cluster = pd.DataFrame({'Cluster_EM': clusters})
    
    def clustering_kmodes(self) -> pd.DataFrame:
        X = self.data_to_cluster
        km = KModes(n_clusters = self.n_cluster, init='Huang', n_init=4, verbose=0)
        clusters = km.fit_predict(X)
        self.cluster = pd.DataFrame({'Cluster_Kmodes': clusters})

    def execute(self):
        if self.clustering_model == 'kmodes':
            self.clustering_kmodes()
            return self.cluster
        elif self.clustering_model == 'kmeans':
            self.clustering_kmeans()
            return self.cluster
        elif self.clustering_model == 'hc':
            self.clustering_hierarchical()
            return self.cluster
        elif self.clustering_model == 'dbscan':
            self.clustering_dbscan()
            return self.cluster
        elif self.clustering_model == 'em':
            self.clustering_expectationMaximisation()
            return self.cluster
        else:
            raise Exception('Modello di clustering non riconosciuto')

    
