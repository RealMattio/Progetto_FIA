import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from statistics import mean
import time


class ClusteringEvaluation:
    def __init__(self, data:pd.DataFrame, labels_name:str, predictions_name:str, dati_binari):
        '''
        :param data: DataFrame contenente i dati
        :param labels_col: Nome della colonna contenente le etichette vere
        :param predictions_col: Nome della colonna contenente le etichette predette
        '''
        self.data = data
        self.labels_name = labels_name
        self.labels = data[labels_name]
        self.predictions_name = predictions_name
        self.predictions = data[predictions_name]
        self.purity = None
        self.contingency_matrix = None
        self.silhouette_mean = None
        self.silhouette_std = None
        self.final_metric = None
        #self.data_to_compute = data.drop(columns=['data_erogazione', 'anno', 'quadrimestre', 'incremento_teleassistenze', self.predictions_name])
        self.dati_binari = dati_binari
        
    '''
    def calculate_purity(true_labels, predicted_labels):
        # Creiamo una tabella di contingenza
        contingency_matrix = pd.crosstab(true_labels, predicted_labels)
        
        # Troviamo la classe con il massimo valore in ogni colonna
        max_values = contingency_matrix.max(axis=0)
        
        # La purezza è la somma di questi massimi valori divisa per il numero totale di punti
        purity = np.sum(max_values) / np.sum(contingency_matrix.values)
        return purity
    '''

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
    
    '''
    def calculate_silhouette(self) -> pd.DataFrame:
        ''
        Calcola i valori di silhouette per ogni punto
        :return: DataFrame con i valori di silhouette per ogni punto aggiunti al dataframe originale
        ''
        # Creiamo un array con i dati
        features_array = self.data_to_compute.values
        silhouette_vals = silhouette_samples(features_array, self.predictions)
        # Aggiungiamo i valori di silhouette al DataFrame
        self.data['silhouette'] = silhouette_vals
        return self.data
    
    
    def evaluate(self) -> tuple:
        ''
        Valuta la bontà del clustering
        :return: tupla che contiene come primo elemento il DataFrame al quale sono stati aggiunti i valori di 
        silhouette per ogni punto e come secondo elemento un dizionario con i valori di purezza, silhouette media e deviazione standard
        ''
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        self.purity = self.purity.item()
        #self.purity = purity_score(self.labels, self.predictions)
        self.data = self.calculate_silhouette()
        self.silhouette_mean = self.data['silhouette'].mean().item()
        self.silhouette_std = self.data['silhouette'].std().item()
        # calcolo il numero di cluster
        N = len(self.data[self.predictions_name].unique())
        # calcolo la media tra purezza e silhouette_mean e sottraggo 0.05 volte il numero di cluster

        self.final_metric = mean([self.purity, self.silhouette_mean]) - 0.05*N
        return self.data, {"purity": self.purity, "silhouette_mean": self.silhouette_mean, "silhouette_std": self.silhouette_std, "final_metric": self.final_metric}
    '''
    def eval(self) -> dict:
        '''
        Valuta la bontà del clustering
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

    def eval2(self) -> dict:
        
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        features_array = self.dati_binari
        start_silhouette = time.time()
        # il primo parametro è il dataframe con tutti i valori (nel nostro caso binari) dei punti
        # il secondo parametro è il vettore con le etichette predette
        silhouette_vals = silhouette_score(features_array, self.predictions)
        end_silhouette = time.time()

         # Calcolo il numero di cluster
        n_clusters = len(np.unique(self.predictions))
        # Calcola la metrica finale
        self.final_metric = self.calcolo_metrica_finale(self.purity, self.silhouette_mean, n_clusters)

        return {"purity": self.purity.item(), "contingency_matrix": self.contingency_matrix, "silhouette": silhouette_vals, "time_silhouette": end_silhouette - start_silhouette, "final_metric": self.final_metric}
