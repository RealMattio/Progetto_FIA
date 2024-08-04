import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_samples#, silhouette_score
from statistics import mean


class ClusteringEvaluation:
    def __init__(self, data:pd.DataFrame, labels_name:str, predictions_name:str):
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
        self.silhouette_mean = None
        self.silhouette_std = None
        
    
    def calculate_purity(true_labels, predicted_labels):
        # Creiamo una tabella di contingenza
        contingency_matrix = pd.crosstab(true_labels, predicted_labels)
        
        # Troviamo la classe con il massimo valore in ogni colonna
        max_values = contingency_matrix.max(axis=0)
        
        # La purezza è la somma di questi massimi valori divisa per il numero totale di punti
        purity = np.sum(max_values) / np.sum(contingency_matrix.values)
        return purity
    

    def purity_score(self, y_true, y_pred) -> float:
        '''
        Compute the purity score.
        :param y_true: the true labels
        :param y_pred: the predicted labels
        :return: the purity score
        '''
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(contingency_matrix)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    def calculate_silhouette(self) -> pd.DataFrame:
        '''
        Calcola i valori di silhouette per ogni punto
        :return: DataFrame con i valori di silhouette per ogni punto aggiunti al dataframe originale
        '''
        # Creiamo un array con i dati
        features_array = self.data.drop(columns=[self.labels_name, self.predictions_name]).values
        silhouette_vals = silhouette_samples(features_array, self.predictions)
        # Aggiungiamo i valori di silhouette al DataFrame
        self.data['silhouette'] = silhouette_vals
        return self.data
    
    
    def evaluate(self) -> dict:
        '''
        Valuta la bontà del clustering
        :return: Dizionario con i valori di purezza, silhouette media e deviazione standard
        '''
        self.purity = self.purity_score(self.labels, self.predictions).item()
        #self.purity = purity_score(self.labels, self.predictions)
        self.data = self.calculate_silhouette()
        self.silhouette_mean = self.data['silhouette'].mean().item()
        self.silhouette_std = self.data['silhouette'].std().item()
        # calcolo il numero di cluster
        N = len(self.data[self.predictions_name].unique())
        # calcolo la media tra purezza e silhouette_mean e sottraggo 0.05 volte il numero di cluster

        self.final_metric = mean([self.purity, self.silhouette_mean]) - 0.05*N
        return {"purity": self.purity, "silhouette_mean": self.silhouette_mean, "silhouette_std": self.silhouette_std, "final_metric": self.final_metric}
