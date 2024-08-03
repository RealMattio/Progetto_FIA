# aggiungere tutti gli import necessari per eseguire opgni step della pipeline
import pandas as pd
import data_preprocessing as dp
import feature_extraction as fe
import clustering as cl
import evaluation as ev
import itertools
from tqdm import tqdm
import time

    # Funzione per ottenere tutte le combinazioni degli elementi di una lista
def all_combinations(input_list):
    result = []
    # Iteriamo su tutte le lunghezze possibili
    for r in range(1, len(input_list) + 1):
        # Otteniamo le combinazioni di lunghezza r
        combinations = itertools.combinations(input_list, r)
        result.extend(combinations)
    return result

# Description: Classe che si occupa di eseguire tutti i passaggi del pipeline
class Pipeline:

    # Costruttore della classe che si occuperà di eseguire tutti i passaggi del pipeline
    def __init__(self, path ='data/challenge_campus_biomedico_2024_sample.csv'):
        self.path = path
        self.data = self.load_data()


    #funzione per legge il dataset
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
    

    # Metodo che esegue la pipeline
    def run(self):
        print(f"Running pipeline from data in {self.path}")
        print("Reading file")
        data_preprocessing = dp.DataPreprocessing(self.data)
        print("Data preprocessing")
        dati = data_preprocessing.preprocessing_data()

        # ciclo per la selezione delle migliori features da utilizzare nel clustering
    
        #lista_di_features=['fascia_eta']
        lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'fascia_eta']
        #lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'sesso', 'asl_erogazione', 'fascia_eta']
        #lista_di_features=['regione_residenza', 'asl_residenza', 'codice_descrizione_attivita', 'sesso', 'tipologia_professionista_sanitario', 'regione_erogazione', 'asl_erogazione', 'fascia_eta']
        
        # lista di liste dove ogni lista interna è l'elenco delle features da utilizzare in quell'iterazione
        features = all_combinations(lista_di_features)
        # Convertiamo le tuple in liste per una visualizzazione più chiara
        features = [list(comb) for comb in features]
        print(f"Number of combinations: {len(features)}")
        #print(f"Features: {features}")
        feature_extractor = fe.FeatureExtraction(dati)
        dati = feature_extractor.extract()
        #print(dati.columns)
        dati = dati[dati['anno'] != 2019]
        print(dati.shape)

        risultati = []
        for feature in tqdm(features):
            # selezioniamo le colonne del dataset il cui nome inizia con le feature scelte
            #print(f"Features: {feature}")
            data = pd.DataFrame()
            for f in feature:
                data1 = dati[[col for col in dati.columns if col.startswith(f)]]
                data = pd.concat([data, data1], axis=1)
            #print(data.head())
            # feature extraction: in questa fase si calcolano le features 'incremento' e 'incremento_teleassistenze'
            # vogliamo in ingresso un dataframe e in uscita verrà fornito lo stesso dataframe con le colonne 'incremento' e 'incremento_teleassistenze' aggiunte
            data = pd.concat([data, dati[['data_erogazione', 'anno', 'quadrimestre', 'incremento_teleassistenze']]], axis=1)
            '''
            feature_extractor = fe.FeatureExtraction(data)
            data = feature_extractor.extract()
            print(data.shape)
            data = data[data['anno'] != 2019]
            print(data.shape)
            '''
            # clustering: in questa fase si esegue il clustering utilizzando le features appena selezionate
            # vogliamo in ingresso un dataframe e restituisce lo stesso dataframe con N (numero di tipologie di cluster) colonne in più, una per ogni tipologia di cluster

            # ATTENZIONE : quando viiene eseguito il clustering bisogna eliminare dal dataframe le colonne che non sono numeriche e rendere numeriche le colonne booleane
            for n in tqdm(range(2, 10)):
                clustering = cl.Clustering(data, n_cluster = n)
                
                clustering.clustering_kmeans()
            
                #print(f"Time Clustering KMeans: {time_k_means}")
                #clustering.clustering_hierarchical()
                #print(f"Clustering Hierarchical: {clustering.data['Cluster_HC'].unique()}")
                #clustering.clustering_dbscan()
                #print(f"Clustering DBSCAN: {clustering.data['Cluster_DBSCAN'].unique()}")
                #clustering.clustering_expectationMaximisation()


                '''
                time_em_start = time.time()
                clustering.clustering_expectationMaximisation()
                time_em_end = time.time()
                time_em = time_em_end - time_em_start
                print(f"Time Clustering Expectation Maximisation: {time_em}")
                '''
                data = clustering.data

                label_counts = data['Cluster_Kmeans'].value_counts()
                #label_counts = data['Cluster_HC'].value_counts()
                #label_counts = data['Cluster_DBSCAN'].value_counts()
                #label_counts = data['Cluster_EM'].value_counts()
                

                print("\nNumero di elementi per ogni cluster:")
                print(label_counts)

                label_counts = data['incremento_teleassistenze'].value_counts()
                print("\nNumero di elementi per ogni incremento:")
                print(label_counts)


                # valutazione: in questa fase si valuta il clustering ottenuto e si salvano i risultati ottenuti
                # vogliamo in ingresso un dataframe e restituisce un dizionario con i risultati del clustering
                evaluation = ev.ClusteringEvaluation(data, 'incremento_teleassistenze', 'Cluster_Kmeans')
                #evaluation = ev.ClusteringEvaluation(data, 'incremento_teleassistenze', 'Cluster_EM')
                results = evaluation.eval()
                results['features'] = feature
                results['n_cluster'] = n
                risultati.append(results)
                '''
                results = evaluation.evaluate()
                results['features'] = feature
                risultati.append(results[1])
                '''
            
        #salvo i dati in un csv
        pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_csv('test_results/test_results5.json')





