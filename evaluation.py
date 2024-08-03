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
        self.contingency_matrix = None
        self.silhouette_mean = None
        self.silhouette_std = None
        self.final_metric = None
        self.data_to_compute = data.drop(columns=['data_erogazione', 'anno', 'quadrimestre', 'incremento_teleassistenze', self.predictions_name])

        
    
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
        #sulle righe ci sono le etichette vere, sulle colonne le etichette predette
        # shape = [n_classes_true, n_classes_pred] = [n_labels, n_clusters]
        
        #per ogni colonna calcolo il massimo e sommo i massimi
        max_value = 0
        for i in range(contingency_matrix.shape[1]):
            max_value += max(contingency_matrix[:, i])
        print(f'{max_value} / {len(y_true)}')
        max_value = max_value / len(y_true)
        


        #print(contingency_matrix)
        # return purity
        purity = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix) # 0 = righe, 1 = colonne! NOI LA DOBBIAMO FARE PER COLONNE!
        return max_value, contingency_matrix
    
    def calculate_silhouette(self) -> pd.DataFrame:
        '''
        Calcola i valori di silhouette per ogni punto
        :return: DataFrame con i valori di silhouette per ogni punto aggiunti al dataframe originale
        '''
        # Creiamo un array con i dati
        features_array = self.data_to_compute.values
        silhouette_vals = silhouette_samples(features_array, self.predictions)
        # Aggiungiamo i valori di silhouette al DataFrame
        self.data['silhouette'] = silhouette_vals
        return self.data
    
    
    def evaluate(self) -> tuple:
        '''
        Valuta la bontà del clustering
        :return: tupla che contiene come primo elemento il DataFrame al quale sono stati aggiunti i valori di 
        silhouette per ogni punto e come secondo elemento un dizionario con i valori di purezza, silhouette media e deviazione standard
        '''
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
    
    def eval(self) -> dict:
        '''
        Valuta la bontà del clustering
        :return: Dizionario con i valori di purezza, silhouette media e deviazione standard
        '''
        self.purity, self.contingency_matrix = self.purity_score(self.labels, self.predictions)
        return {"purity": self.purity.item(), "contingency_matrix": self.contingency_matrix}
