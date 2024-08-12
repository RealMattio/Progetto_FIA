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
from sklearn.decomposition import PCA
from datetime import datetime


# Funzione per ottenere tutte le combinazioni degli elementi di una lista
def all_combinations(input_list) -> list:
    result = []
    # Iteriamo su tutte le lunghezze possibili
    for r in range(1, len(input_list) + 1):
        # Otteniamo le combinazioni di lunghezza r
        combinations = itertools.combinations(input_list, r)
        result.extend([list(comb) for comb in combinations])
    for e in result:
        list(e).append('incremento_teleassistenze')
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

        # Fase 0: Lettura del file
        print("Reading file")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()

        # Fase 3: Clustering
        print("Clustering")
        # Lista che contiene tutte le colonne con cui vorrò fare il clustering
        lista_di_features=['asl_residenza', 'codice_descrizione_attivita', 'sesso', 'asl_erogazione', 'fascia_eta']
        # Lista di tutte le combinazioni possibili delle features
        features = all_combinations(lista_di_features)
        risultati = []
        n_cluster = self.n_cluster
        df_cluster = data[lista_di_features]
        clustering = cl.Clustering(df_cluster, n_cluster, clustering_model = self.clustering_type)
        cluster_assigned = clustering.execute()
        pd.concat([data, cluster_assigned], axis=1)

        # Fase 4: Evaluation
        print("Evaluation")
        evaluation = ev.ClusteringEvaluation(df_cluster, data[['incremento_teleassistenze']], cluster_assigned, self.clustering_type)
        data['Silhouette'] = evaluation.calculate_silhouette()
        results = evaluation.evaluate()
        results['features'] = lista_di_features
        risultati.append(results)
        
        # Fase 5: Salvataggio dei risultati
        print("Saving results")
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_file = f'test_results/test_results_{self.clustering_type}_{self.n_cluster}_{now}.csv'
        pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_csv(name_file)

    
    def run2(self):
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
            #print(data)
            
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
            '''
            # effettuare encoding con OneHot 
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(data_clustered)
            # poi passi a ClusteringEvaluation i dati binari
            #print(encoded_data)
            '''
            print(data_clustered.columns)
            # stampo i valori unici per ogni colonna
            print(data_clustered.nunique())

            encoded_data = pd.get_dummies(data_clustered).astype(float)

            print(encoded_data.columns)
            # Fase 3: PCA
            n_components = 3
            pca = PCA(n_components = n_components)
            pca_data = pca.fit_transform(encoded_data)
            
            
            #in evaluation invece di data_clustered, gli va passato pca_data
            evaluation = ev.ClusteringEvaluation(pca_data, 'incremento_teleassistenze', 'Cluster_Kmodes', pca_data)
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





   