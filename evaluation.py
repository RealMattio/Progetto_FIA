import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from statistics import mean
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ClusteringEvaluation:
    def __init__(self, data:pd.DataFrame, labels:pd.DataFrame, predictions:pd.DataFrame, clustering_type:str, reduction:bool = False, n_components:int = 3):
        '''
        :param data: DataFrame contenente i dati
        :param labels_col: Dataframe contenente le etichette reali
        :param predictions_col: Dataframe contenente le etichette predette
        :param clustering_type: Tipo di algoritmo di clustering utilizzato
        :param reduction: Booleano che indica se andra' stata fatta una riduzione di dimensionalità con PCA per velocizzare il calcolo della shilouette
        :param n_components: Numero di componenti principali da considerare per la PCA
        '''
        self.data = data
        self.labels = labels
        self.predictions = predictions
        self.purity = None
        self.silhouette_mean = None
        self.silhouette_std = None
        self.final_metric = None
        self.clustering_type = clustering_type
        self.reduction = reduction
        self.n_components = n_components

    def purity_score(self, y_true, y_pred) -> float:
        '''
        Compute the purity score.
        :param y_true: the true labels
        :param y_pred: the predicted labels
        :return: the purity score
        '''
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #sulle righe ci sono le etichette vere, sulle colonne le etichette predette
        # shape = [n_classes_true, n_classes_pred] = [n_labels, n_clusters]
        
        #print(contingency_matrix)
        # return purity
        # se sulle riche ci sono le label e sulle colonne i cluster allora il massimo deve essere calcolato lungo le colonne. In altre parole, fissato il cluster
        # sulla colonna, scorro le righe in cerca del massimo. Perciò axis=0
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) # 0 = scorre le righe fissando le colonne, 1 = viceversa! NOI LA DOBBIAMO FISSARE LE COLONNE!
        return purity, contingency_matrix
    
    
    def eval(self) -> dict:
        '''
        Valuta la bontà del clustering calcolando solo la purezza
        :return: Dizionario con i valori di purezza, silhouette media e deviazione standard
        '''
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        return {"purity": self.purity.item(), "contingency_matrix": str(self.contingency_matrix)}
    
    
    def calcolo_metrica_finale(self, purity, silhouette_mean, n_clusters):
        '''
        Calcola la metrica finale combinando purezza, silhouette mean e un termine di penalità.
        1)param purity: Purezza del clustering
        2)param silhouette_mean: Punteggio medio di silhouette
        3)param n_clusters: Numero di cluster
        '''
        penalty = 0.05 * n_clusters
        final_metric = mean([purity, silhouette_mean]) - penalty
        return final_metric
        #una volta studiati i risultati bisogna normalizzare tale metrica
        
    def eval2(self) -> dict:
        
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        features_array = self.dati_binari
        start_silhouette = time.time()
        # il primo parametro è il dataframe con tutti i valori (nel nostro caso binari) dei punti
        # il secondo parametro è il vettore con le etichette predette
        silhouette_vals = silhouette_score(features_array, self.predictions)
        end_silhouette = time.time()

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         # Calcolo il numero di cluster
        n_clusters = len(np.unique(self.predictions))
        # Calcola la metrica finale
        self.final_metric = self.calcolo_metrica_finale(self.purity, self.silhouette_mean, n_clusters)

        return {"purity": self.purity.item(), "contingency_matrix": self.contingency_matrix, "silhouette": silhouette_vals, "time_silhouette": end_silhouette - start_silhouette} #"final_metric": self.final_metric

    def calculate_silhouette(self) -> pd.DataFrame: #OK
        '''
        Calcola i valori di silhouette per ogni punto
        Se il dataset è dotato di valori features categoriche è necessario convertirli in valori numerici, questo sarà vero solo se è stato usato il KModes
        :return: DataFrame con i valori di silhouette per ogni punto aggiunti al dataframe originale
        '''
        if self.clustering_type == 'kmodes':
            encoded_data = pd.get_dummies(self.data).astype(float)
            if self.reduction:
                if min(encoded_data.shape) > self.n_components:
                    pca = PCA(n_components = self.n_components)
                    reduced_data = pca.fit_transform(encoded_data)
                else:
                    pca = PCA(n_components = min(encoded_data.shape))
                    reduced_data = pca.fit_transform(encoded_data)
            else:
                reduced_data = encoded_data
        start_time = time.time()
        silhouette_vals = silhouette_samples(reduced_data, self.predictions)
        end_time = time.time()
        self.time_silhouette = end_time - start_time
        self.silhouette_vals = pd.DataFrame(silhouette_vals, columns = ['silhouette'])
        return self.silhouette_vals
    
    def calculate_final_metric(self) -> float: #OK
        '''
        La metrica finale è calcolata come la media tra purezza e silhouette_mean meno 0.05 volte il numero di cluster
        '''
        penalty = 0.05 * len(self.predictions.iloc[:,0].unique())
        final_metric = mean([self.purity, self.silhouette_mean]) - penalty
        return final_metric
    
    def evaluate(self) -> dict: #OK
        '''
        Calcolo la purezza, la silhouette media standardizzata e la metrica finale
        '''
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        scaler = MinMaxScaler()
        silh_normalized = scaler.fit_transform(self.silhouette_vals)
        self.silhouette_mean = silh_normalized.mean().item()
        final_metric = self.calculate_final_metric()
        return {"purity": self.purity.item(), "contingency_matrix": np.array2string(self.contingency_matrix), "silhouette_mean": self.silhouette_mean, "final_metric": final_metric, "time_silhouette": self.time_silhouette}