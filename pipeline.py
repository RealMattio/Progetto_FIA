# aggiungere tutti gli import necessari per eseguire opgni step della pipeline
import pandas as pd
import data_preprocessing as dp
import feature_extraction as fe
import clustering as cl
import evaluation as ev
import itertools
from tqdm import tqdm
import time
import prince
from sklearn.preprocessing import OneHotEncoder


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
    def __init__(self, path ='data/challenge_campus_biomedico_2024_sample.csv', clustering_type = 'kmeans', n_cluster = 4):
        self.path = path
        self.data = self.load_data()
        self.clustering_type = clustering_type
        self.n_cluster = n_cluster

    #funzione per legge il dataset
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
    
    def run(self):
        print(f"Running pipeline from data in {self.path}")
        print("Reading file")
        data_preprocessing = dp.DataPreprocessing(self.data)
        print("Data preprocessing")
        data = data_preprocessing.clean_data()
        #columns_with_nan = data.columns[data.isnull().any()].tolist()
        #print(columns_with_nan)

        data = data_preprocessing.transform_data()
        dati = data[['id_prenotazione', 'data_nascita', 'sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita', 'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario', 'tipologia_professionista_sanitario', 'data_erogazione', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_disdetta','eta', 'fascia_eta']]
        dati_dummy = data_preprocessing.reduce_data()
        lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'sesso', 'asl_erogazione', 'fascia_eta']
        features = all_combinations(lista_di_features)

        # Convertiamo le tuple in liste per una visualizzazione più chiara
        features = [list(comb) for comb in features]
        #rimuovo la sola feature 'sesso' perchè da vita a solo due cluster
        features.remove(['sesso'])

        print(f"Number of combinations: {len(features)}")

        feature_extractor = fe.FeatureExtraction(dati)
        dati = feature_extractor.extract()
        label_counts = dati['incremento_teleassistenze'].value_counts()
        #print(f"\nNumero di elementi per ogni incremento: {label_counts}")

        feature_extractor2 = fe.FeatureExtraction(dati_dummy)
        dati_dummy = feature_extractor2.extract()
        
        
        
        #escludo il primo quadrimestre del 2019 (Va fatto se uso l-incrmemento sequenziale)
        #dati = dati[~((dati['anno'] == 2019) & (dati['quadrimestre'] == 1))]
        #dati_dummy = dati_dummy[~((dati_dummy['anno'] == 2019) & (dati_dummy['quadrimestre'] == 1))]
        
        #escludo il 2019 (se calcolo incrememnto quadrimestre per quadrimestre)
        dati = dati[~(dati['anno'] == 2019)]
        dati_dummy = dati_dummy[~(dati_dummy['anno'] == 2019)]
        label_counts = dati['incremento_teleassistenze'].value_counts()
        print(f"\nNumero di elementi per ogni incremento dopo elminiazione 2019: {label_counts}")
        
        columns_with_nan = dati.columns[dati.isnull().any()].tolist()

        dati = dati.drop(columns=columns_with_nan)
        #print(dati_dummy)
        '''
        mca = prince.MCA(n_components=3)
        dati_da_fittare = dati.loc[:, lista_di_features]
        print(dati_da_fittare.columns)
        mca = mca.fit(dati_da_fittare)
        nuovi_dati = mca.transform(dati_da_fittare)
        print(nuovi_dati.head())
        '''
        risultati = []
        n = self.n_cluster
        feature=['tutte']
        features=[["codice_descrizione_attivita","incremento_teleassistenze"]]
        for feature in tqdm(features):
        #for n in tqdm(range(2,6)):
            # seleziono solo le colonne presenti in feature
            #feature.append('incremento_teleassistenze')
            data = dati[feature]
            print(data)
            
            clustering = cl.Clustering(dati_dummy, n_cluster = n, data_categorical = data)
            #clustering = cl.Clustering(nuovi_dati, n_cluster = n, data_categorical = data)

            if self.clustering_type == 'kmeans':
                clustering.clustering_kmeans()
                data_clustered = clustering.data
            elif self.clustering_type == 'hierarchical':
                clustering.clustering_hierarchical()
                data_clustered = clustering.data
            elif self.clustering_type == 'dbscan':
                clustering.clustering_dbscan()
                data_clustered = clustering.data
            elif self.clustering_type == 'expectationMaximisation':
                clustering.clustering_expectationMaximisation()
                data_clustered = clustering.data
            elif self.clustering_type == 'kmodes':
                clustering.clustering_kmodes()
                data_clustered = clustering.data_categorical
            
            # EDO LAVORA QUA
            # effettuare encoding con OneHot 
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(data_clustered)
            # poi passi a ClusteringEvaluation i dati binari
            print(encoded_data)

            self.clustering_type == 'PCA'
            clustering.clustering_PCA()
            data_clustered = clustering.data_categorical
            

            evaluation = ev.ClusteringEvaluation(data_clustered, 'incremento_teleassistenze', 'Cluster_Kmodes', encoded_data)
            #evaluation = ev.ClusteringEvaluation(data, 'incremento_teleassistenze', 'Cluster_EM')
            results = evaluation.eval2()
            results['features'] = feature
            results['n_cluster'] = n

            label_counts = data_clustered['Cluster_Kmodes'].value_counts()
            results['cluster counts'] = label_counts
            label_counts = data_clustered['incremento_teleassistenze'].value_counts()
            results['label counts'] = label_counts

            risultati.append(results)
        
        #salvo i dati in un csv
        pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_json('test_results/test_results_Kmodes_old_labels_Huang_scelta_features_trimestri.json')





    
    '''
    # Metodo che esegue la pipeline
    def run(self):
        print(f"Running pipeline from data in {self.path}")
        print("Reading file")
        data_preprocessing = dp.DataPreprocessing(self.data)
        print("Data preprocessing")
        dati = data_preprocessing.preprocessing_data()

        # ciclo per la selezione delle migliori features da utilizzare nel clustering
    
        #lista_di_features=['fascia_eta']
        #lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'fascia_eta']
        #lista_di_features=['asl_erogazione', 'fascia_eta']
        lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'sesso', 'asl_erogazione', 'fascia_eta']
        #lista_di_features=['regione_residenza', 'asl_residenza', 'codice_descrizione_attivita', 'sesso', 'tipologia_professionista_sanitario', 'regione_erogazione', 'asl_erogazione', 'fascia_eta']
        
        # lista di liste dove ogni lista interna è l'elenco delle features da utilizzare in quell'iterazione
        features = all_combinations(lista_di_features)
        #features = features.remove(('sesso',))
        # Convertiamo le tuple in liste per una visualizzazione più chiara
        features = [list(comb) for comb in features]
        print(f"Number of combinations: {len(features)}")
        #print(f"Features: {features}")
        feature_extractor = fe.FeatureExtraction(dati)
        dati = feature_extractor.extract()
        #print(dati.columns)
        #dati = dati[dati['anno'] != 2019]
        #escludo il primo quadrimestre del 2019
        dati = dati[~((dati['anno'] == 2019) & (dati['quadrimestre'] == 1))]


        risultati = []
        #features=[['asl_erogazione', 'fascia_eta']]
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
            #''
            feature_extractor = fe.FeatureExtraction(data)
            data = feature_extractor.extract()
            print(data.shape)
            data = data[data['anno'] != 2019]
            print(data.shape)
            ''
            # clustering: in questa fase si esegue il clustering utilizzando le features appena selezionate
            # vogliamo in ingresso un dataframe e restituisce lo stesso dataframe con N (numero di tipologie di cluster) colonne in più, una per ogni tipologia di cluster

            # ATTENZIONE : quando viiene eseguito il clustering bisogna eliminare dal dataframe le colonne che non sono numeriche e rendere numeriche le colonne booleane
            n = 4
            #for n in tqdm(range(2, 10)):
            clustering = cl.Clustering(data, n_cluster = n)
            
            clustering.clustering_kmeans()
        
            #print(f"Time Clustering KMeans: {time_k_means}")
            #clustering.clustering_hierarchical()
            #print(f"Clustering Hierarchical: {clustering.data['Cluster_HC'].unique()}")
            #clustering.clustering_dbscan()
            #print(f"Clustering DBSCAN: {clustering.data['Cluster_DBSCAN'].unique()}")
            #clustering.clustering_expectationMaximisation()


            ''
            time_em_start = time.time()
            clustering.clustering_expectationMaximisation()
            time_em_end = time.time()
            time_em = time_em_end - time_em_start
            print(f"Time Clustering Expectation Maximisation: {time_em}")
            ''
            data = clustering.data

            #label_counts = data['Cluster_HC'].value_counts()
            #label_counts = data['Cluster_DBSCAN'].value_counts()
            #label_counts = data['Cluster_EM'].value_counts()
            #print("\nNumero di elementi per ogni cluster:")
            #print(label_counts)

            #print("\nNumero di elementi per ogni incremento:")
            #print(label_counts)


            # valutazione: in questa fase si valuta il clustering ottenuto e si salvano i risultati ottenuti
            # vogliamo in ingresso un dataframe e restituisce un dizionario con i risultati del clustering
            evaluation = ev.ClusteringEvaluation(data, 'incremento_teleassistenze', 'Cluster_Kmeans')
            #evaluation = ev.ClusteringEvaluation(data, 'incremento_teleassistenze', 'Cluster_EM')
            results = evaluation.eval()
            results['features'] = feature
            results['n_cluster'] = n

            label_counts = data['Cluster_Kmeans'].value_counts()
            results['cluster counts'] = label_counts
            label_counts = data['incremento_teleassistenze'].value_counts()
            results['label counts'] = label_counts

            risultati.append(results)
            ''
            results = evaluation.evaluate()
            results['features'] = feature
            risultati.append(results[1])
            ''
        
        #salvo i dati in un csv
        pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_csv('test_results/test_results6.json')

    '''



