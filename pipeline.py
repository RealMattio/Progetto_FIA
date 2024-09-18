# aggiungere tutti gli import necessari per eseguire opgni step della pipeline
import pandas as pd
import data_preprocessing as dp
import feature_extraction as fe
import clustering as cl
import evaluation as ev
import itertools
from tqdm import tqdm
import json
import os
import pickle
from datetime import datetime
import ast
from grafici import grafici


# Funzione per ottenere tutte le combinazioni degli elementi di una lista
def all_combinations(input_list) -> list:
    result = []
    # Iteriamo su tutte le lunghezze possibili
    for r in range(1, len(input_list) + 1):
        # Otteniamo le combinazioni di lunghezza r
        combinations = itertools.combinations(input_list, r)
        result.extend([list(comb) for comb in combinations])
        
    result.remove(['incremento_teleassistenze'])
    return result

def count_iter_folders(directory = './all_clustering_results/') -> int:
    # Se la cartella non esiste, la creo
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Lista delle cartelle che iniziano con 'iter'
    iter_folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name.startswith('results_iter')]
    
    return len(iter_folders)

class Pipeline:

    # Costruttore della classe che si occuperà di eseguire tutti i passaggi del pipeline
    def __init__(self, path ='data/challenge_campus_biomedico_2024_sample.csv', clustering_type = 'kmeans', n_cluster = 4):
        self.path = path
        self.data = self.load_data()
        self.clustering_type = clustering_type
        self.n_cluster = n_cluster

    #funzione per leggere il dataset viene utilizzata all'interno del costruttore
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
    
    # Fase 1: seleziono il sottoinsieme di features migliore provando tutte le combinazioni possibili
    def fase1_clustering_evaluation_combinazioni_feature(self):
        print(f"Running pipeline from data in {self.path}")

        # Fase 0: Lettura del file
        print("Reading file in fase 1")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing in fase 1")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction in fase 1")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()
        
        # Elimino i dati del 2019 perchè non hanno incremento
        data = data[data['anno'] != 2019]
        # Fase 3: Clustering
        print("Clustering and Evaluation in fase 1")
        with open('lista_possibili_features.pkl', 'rb') as file:
            features = pickle.load(file)
        iter = count_iter_folders()
        features = features[iter:]
        print(f"Starting performing {len(features)} features combinations from iteration {iter}")
        for feature in features:
            cluster_assigned = pd.DataFrame()
            risultati = []
            print(f"Features: {feature}")
            df_cluster = data[feature]
            for n in tqdm(range(3, self.n_cluster + 3)):
                clustering = cl.Clustering(df_cluster, n, clustering_model = self.clustering_type)
                column_name = f'{self.clustering_type}_{n}_clusters_iter{iter}'
                cluster_assigned[column_name] = clustering.execute()

                # Fase 4: Evaluation
                evaluation = ev.ClusteringEvaluation(df_cluster, data[['incremento_teleassistenze']], cluster_assigned[column_name], self.clustering_type)

                results = evaluation.eval()
                results['features'] = feature
                results['n_cluster'] = n
                results['iter'] = iter
                risultati.append(results)
            # Salvo i risultati ad ogni iterazione
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory = f'all_clustering_results/results_iter{iter}'
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Cartella '{directory}' creata.")
            name_file = f'all_clustering_results/results_iter{iter}/performance_{self.clustering_type}_{feature}_{now}.csv'
            pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_csv(name_file)
            name_cluster_file = f'all_clustering_results/results_iter{iter}/cluster_assigned_{self.clustering_type}_{feature}_{now}.csv'
            cluster_assigned.to_csv(name_cluster_file)
            iter += 1

    def fase1_clustering_evaluation_tutte_features(self):
        # Fase 0: Lettura del file
        print("Reading file in fase 1.1")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing in fase 1.1")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction in fase 1.1")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()
        
        # Elimino i dati del 2019 perchè non hanno incremento
        data = data[data['anno'] != 2019]
        # Fase 3: Clustering
        print("Clustering and Evaluation in fase 1.1")
        # Lista che contiene tutte le colonne con cui vorrò fare il clustering
        features = ['sesso', 'asl_residenza', 'codice_descrizione_attivita', 'asl_erogazione', 'tipologia_struttura_erogazione', 'tipologia_professionista_sanitario',
            'fascia_eta', 'incremento_teleassistenze']       
        # Lista di tutte le combinazioni possibili delle features
        if not os.path.exists('all_clustering_results/results_all_features'):
            print(f"Starting performing clustering on all features")

            cluster_assigned = pd.DataFrame()
            risultati = []
            df_cluster = data[features]
            for n in tqdm(range(3, self.n_cluster + 3)):
                clustering = cl.Clustering(df_cluster, n, clustering_model = self.clustering_type)
                column_name = f'{self.clustering_type}_{n}_clusters'
                cluster_assigned[column_name] = clustering.execute()

                # Fase 4: Evaluation
                evaluation = ev.ClusteringEvaluation(df_cluster, data[['incremento_teleassistenze']], cluster_assigned[column_name], self.clustering_type)
                results = evaluation.eval()
                results['features'] = features
                results['n_cluster'] = n
                risultati.append(results)
            # Salvo i risultati ad ogni iterazione
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory = f'all_clustering_results/results_all_features'
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Cartella '{directory}' creata.")
            name_file = f'{directory}/performance_{self.clustering_type}_all_features_{now}.csv'
            pd.DataFrame(risultati).sort_values(by='purity', ascending=False).to_csv(name_file)
            name_cluster_file = f'{directory}/cluster_assigned_{self.clustering_type}_all_features_{now}.csv'
            cluster_assigned.to_csv(name_cluster_file)

    def fase1_estrazione_dei_migliori_risultati(self):
        if not os.path.exists('best_results/best_performance.csv') and not os.path.exists('best_results/best_cluster_assigned.parquet'):
            # Fase 1: lettura dei risultati ottenuti
            dir = 'all_clustering_results'
            folders = [dir+'/'+name for name in os.listdir(dir) if os.path.isdir(dir+'/'+name) and name.startswith('results_')]

            # se il numero di elementi presenti nella folder è maggiore di 2 allora leggo l'elemento che inizia con 'performance' e leggo le purity
            cluster_to_save = None
            performance_to_save = None
            for folder in tqdm(folders):
                # Conto il numero di file nella cartella
                file_in_folder = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            
                # Variabile per il nome del file
                nome_file_performance = None

                # Controlla se ci sono più di due file
                if len(file_in_folder) >= 2:
                    # Cerca un file il cui nome inizia con 'performance'
                    for file in file_in_folder:
                        if file.startswith('performance'):
                            # Salva il nome del file
                            nome_file_performance = file
                            # Apri e leggi il contenuto del file
                            with open(os.path.join(folder, file), 'r') as f:
                                data_performance = pd.read_csv(f)  

                        elif file.startswith('cluster_assigned'):
                            # Apri e leggi il contenuto del file
                            with open(os.path.join(folder, file), 'r') as f:
                                data_cluster = pd.read_csv(f)
                        
                        # Se entrambi i file sono stati letti, esco dal ciclo
                        if nome_file_performance and data_cluster is not None:
                            break
                    # seleziono le righe con putiry maggiore di 0.7
                    data_performance = data_performance[data_performance['purity'] > 0.7]
                    # salvo i valori unici contenuti nella colonna 'n_cluster'
                    n_cluster = data_performance['n_cluster'].unique()
                    # seleziono le colonne di 'data_cluster' che iniziano con 'kmodes'
                    for n in n_cluster:
                        column_name = f'kmodes_{n}'
                        cluster_to_save = pd.concat([cluster_to_save, data_cluster.loc[:, data_cluster.columns.str.startswith(column_name)]], axis=1)
                    performance_to_save = pd.concat([performance_to_save, data_performance])
            
            # Salvo i risultati
            directory = 'best_results'
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Cartella '{directory}' creata.")
            performance_to_save.sort_values(by='purity', ascending=False).to_csv(f'{directory}/best_performance.csv')
            cluster_to_save.to_parquet(f'{directory}/best_cluster_assigned.parquet')

    # Fase 2: valutazione dei risultati ottenuti, i migliori ci diranno quali sono le features piu' significative
    def fase2_preliminary_results_evaluation(self):
        # Eseguo nuovamente la fase di lettura e preprocessing del dataset
        print("Reading file in fase 2")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing in fase 2")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction in fase 2")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()
        
        # Elimino i dati del 2019 perchè non hanno incremento
        data = data[data['anno'] != 2019]
        # Fase 3: evaluation
        print("Evaluation in fase 2")
        
        features = ['sesso', 'asl_residenza', 'codice_descrizione_attivita', 'asl_erogazione', 'tipologia_struttura_erogazione', 'tipologia_professionista_sanitario',
            'fascia_eta', 'incremento_teleassistenze']       
        
        print(f"Starting performing evaluation on best features combinations")
        
        # leggo le performance migliori già filtrate nelle fasi precedenti: con purity maggiore di 0.9 e con un numero di features maggiore o uguale a 4
        performances = pd.read_csv('best_results/best_performance_filtered.csv')
        assegnazioni = pd.read_parquet('best_results/best_cluster_assigned.parquet')
        silhouettes = pd.DataFrame()
        risultati = []
        # poiche' in questa fase il calcolo della silhouette richiede molto tempo, si utilizza l'indice per riprendere il calcolo da dove si era interrotto
        # se l'indice non esiste, si parte da 0, altrimenti si parte dall'indice salvato
        if os.path.exists('best_results/index.json'):
            with open('best_results/index.json', 'r') as f:
                l = json.load(f)
                index = l['index']
        else:
            index = 0 

        # itero su tutte le performance filtrate
        for ind, performace in tqdm(performances.iterrows(), total = performances.shape[0]):
            # se l'indice è minore di quello salvato, salto l'iterazione perche' il calcolo è già stato fatto
            if ind < index:
                continue
            # all'interno di performance ho salvato le features come una stringa. La converto in lista
            features = ast.literal_eval(performace['features'])
            n_cluster = performace['n_cluster']
            iter = performace['iter']
            
            # seleziono solo le colonne che sono state utilizzate per il clustering relativamente alla performance corrente
            data_to_eval = data[features]
            # seleziono solo le assegnazioni relative alla performance corrente, che sono poi le predizioni fatte con quel clustering
            cluster_assigned = assegnazioni[[f'kmodes_{n_cluster}_clusters_iter{iter}']]
            
            # Fase 4: Evaluation
            evaluation = ev.ClusteringEvaluation(data_to_eval, data[['incremento_teleassistenze']], cluster_assigned, self.clustering_type, True, 10)
            silhouette = evaluation.calculate_silhouette()
            silhouette.rename(columns={'silhouette': f'silhouette_kmodes_{n_cluster}_clusters_iter{iter}'}, inplace=True)
            
            # Salvo la silhouette di tutti i punti così da poterla utilizzare per i grafici
            if os.path.exists('best_results/silhouettes.csv'):
                silhouettes = pd.read_csv('best_results/silhouettes.csv')
                silhouettes = pd.concat([silhouettes, silhouette])
                silhouettes.to_csv('best_results/silhouettes.csv')
            else:
                silhouette.to_csv('best_results/silhouettes.csv')
            
            # Salvo i risultati
            results = evaluation.evaluate()
            results['features'] = str(features)
            results['n_cluster'] = n_cluster
            results['iter'] = iter
            results = pd.DataFrame(results, index=[0])
            if os.path.exists('best_results/risultati.csv'):
                risultati = pd.read_csv('best_results/risultati.csv')
                risultati = pd.concat([risultati, results])
                risultati.to_csv('best_results/risultati.csv', index=False)
                
            else:
                pd.DataFrame(results).to_csv('best_results/risultati.csv', index=False)
                
            # aggiorno e salvo l'indice
            index += 1
            if os.path.exists('best_results/index.json'):
                with open('best_results/index.json', 'r') as f:
                    indici = json.load(f)
                indici['index'] = index
                with open('best_results/index.json', 'w') as f:
                    json.dump(indici, f)
            else:
                with open('best_results/index.json', 'w') as f:
                    json.dump({'index': index}, f)
    
    # Fase 3: Hyperparameters tuning - dopo la scelta delle features, su di esse calcoliamo i migliori iperparametri, usando una ricerca a griglia
    def fase3_hyperparameter_tuning(self):
        print("Reading file in fase 3")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing in fase 3")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction in fase 3")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()
        
        # Elimino i dati del 2019 perchè non hanno incremento
        data = data[data['anno'] != 2019]
        # Fase 3: Hyperparameters tuning
        print("Hyperparameters tuning in fase 3")

        # leggo il file con i risultati per ottenere le features migliori precedentemente individuate
        performances = pd.read_csv('best_results/risultati.csv')
        # ordino le performances in base alla metrica finale e resetto gli indici per poter iterare su di essi
        performances = performances.sort_values(by='final_metric', ascending=False).reset_index(drop=True)


        if os.path.exists('best_results/index.json'):
            with open('best_results/index.json', 'r') as f:
                l = json.load(f)
                index_hp = l['index_hp']
        else:
            index_hp = 0 

        # per ogni performance rieseguo il clustering andando a modificare gli iperparametri
        for ind, performace in performances.iterrows():
            if ind < index_hp:
                continue
            print(f'Iteration {ind}')
            cluster_assigned = pd.DataFrame()
            #risultati = []
            features = ast.literal_eval(performace['features'])
            data_to_cluster = data[features]
            n = performace['n_cluster']
            n_init = range(1, 11)
            l = ['Huang', 'Cao']

            for d in tqdm(list(itertools.product(n_init, l))):
                clustering = cl.Clustering(data_to_cluster, n, clustering_model = self.clustering_type, kmodes_init=d[1], kmodes_n_init=d[0])
                column_name = f'{self.clustering_type}_{n}_clusters_features_at_{ind}_{d[1]}_neighbors_init_{d[0]}'
                cluster_assigned[column_name] = clustering.execute()

                evaluation = ev.ClusteringEvaluation(data_to_cluster, data[['incremento_teleassistenze']], cluster_assigned[column_name], self.clustering_type)
                
                results = evaluation.eval()
                results['features'] = str(features)
                results['n_cluster'] = n
                results['iter_on_best_performances'] = ind
                results['kmodes_init_type'] = d[1]
                results['kmodes_n_init'] = d[0]

                results = pd.DataFrame(results, index=[0])
                # Salvo i risultati ad ogni iterazione
                dir_results = f'best_results/hp_tuning_results.csv'
                dir_cluster = f'best_results/hp_tuning_cluster_assigned.parquet'

                if os.path.exists(dir_results):
                    risultati = pd.read_csv(dir_results)
                    risultati = pd.concat([risultati, results])
                    risultati.to_csv(dir_results, index=False)
                else:
                    pd.DataFrame(results).to_csv(dir_results, index=False)
                
                if os.path.exists(dir_cluster):
                    clusters = pd.read_parquet(dir_cluster)
                    clusters = pd.concat([clusters, cluster_assigned[column_name]], axis=1)
                    clusters.to_parquet(dir_cluster)
                else:
                    cluster_assigned.to_parquet(dir_cluster)
            # aggiorno il contatore per l'indice
            index_hp += 1
            if os.path.exists('best_results/index.json'):
                with open('best_results/index.json', 'r') as f:
                    indici = json.load(f)
                indici['index_hp'] = index_hp
                with open('best_results/index.json', 'w') as f:
                    json.dump(indici, f)
            else:
                with open('best_results/index.json', 'w') as f:
                    json.dump({'index_hp': index_hp}, f)

    # Fase 4: Hyperparameters tuning evaluation - si valutano i risultati ottenuti. I migliori forniranno il modello definitivo
    def fase4_hyperparameter_tuning_evaluation(self):
        print("Reading file in fase 4")
        data_preprocessing = dp.DataPreprocessing(self.data)
        
        # Fase 1: Data Preprocessing
        print("Data preprocessing in fase 4")
        data = data_preprocessing.preprocessing_data()
        
        # Fase 2: Feature Extraction
        print("Feature extraction in fase 4")
        feature_extractor = fe.FeatureExtraction(data)
        data = feature_extractor.extract()
        
        # Elimino i dati del 2019 perchè non hanno incremento
        data = data[data['anno'] != 2019]
        # Fase 3: evaluation
        print("Evaluation in fase 4")
        
        # devo leggere i risultati del tuning degli iperparametri
        # poi devo ordinarli in base alla purezza e prendere solo quelli la cui purezza e' maggiore di 0.9
        ris = pd.read_csv('best_results/hp_tuning_results.csv')
        ris = ris[ris['purity'] >= 0.9]
        # seleziono tutte le colonne tranne 'kmodes_n_init', perche' e' l'unica che potrebbe creare dei duplicati. In altre parole se si ottiene
        # la stessa purity indipendentemente dal numero dei vicini, allora non e' un iperparametro interessante. Pertanto lo escludo in fase di filtraggio.
        df_ = ris.loc[:, ris.columns != 'kmodes_n_init']
        df_ = df_.drop_duplicates()
        ris = ris.loc[df_.index]
        ris.to_csv('best_results/hp_tuning_results_filtered.csv', index=False)

        # poi devo leggere le assegnazioni ai cluster dei punti
        assegnazioni = pd.read_parquet('best_results/hp_tuning_cluster_assigned.parquet')
        silhouettes = pd.DataFrame()
        risultati = []
        if os.path.exists('best_results/index.json'):
            with open('best_results/index.json', 'r') as f:
                l = json.load(f)
                index = l['index_hp_eval']
        else:
            index = 0 
        ris.reset_index(drop=True, inplace=True)
        can_save = False
        for ind, performace in tqdm(ris.iterrows(), total = ris.shape[0]):
            if ind < index:
                continue
            # la variabile can save serve fuori dal for: se non e' mai stata fatta nessuna iterazione allora non c'e' nulla da salvare
            can_save = True
            features = ast.literal_eval(performace['features'])
            n_cluster = performace['n_cluster']
            iter = performace['iter_on_best_performances']
            kmodes_init = performace['kmodes_init_type']
            kmodes_n_init = performace['kmodes_n_init']
            
            data_to_eval = data[features]
            cluster_assigned = assegnazioni[[f'{self.clustering_type}_{n_cluster}_clusters_features_at_{iter}_{kmodes_init}_neighbors_init_{kmodes_n_init}']]
        # poi devo calcolare la silhouette per ogni assegnazione
            
            evaluation = ev.ClusteringEvaluation(data_to_eval, data[['incremento_teleassistenze']], cluster_assigned, self.clustering_type, True, 10)
            silhouette = evaluation.calculate_silhouette()
            silhouette.rename(columns={'silhouette': f'silhouette_kmodes_{n_cluster}_clusters_features_at_{iter}_{kmodes_init}_neighbors_init_{kmodes_n_init}'}, inplace=True)
        # poi devo salvare la silhouette 

            if os.path.exists('best_results/silhouettes_hp_tuning_finale.csv'):
                silhouettes = pd.read_csv('best_results/silhouettes_hp_tuning_finale.csv')
                silhouettes = pd.concat([silhouettes, silhouette], axis=1)
                silhouettes.to_csv('best_results/silhouettes_hp_tuning_finale.csv', index=False)
            else:
                silhouette.to_csv('best_results/silhouettes_hp_tuning_finale.csv', index=False)
            
        # poi calcolo la metrica finale e i vari risultati
            results = evaluation.evaluate()
            results['features'] = str(features)
            results['n_cluster'] = n_cluster
            results['iter_on_best_performances'] = iter
            results['kmodes_init_type'] = kmodes_init
            results['kmodes_n_init'] = kmodes_n_init
            results = pd.DataFrame(results, index=[0])
            if os.path.exists('best_results/risultati_hp_tuning_finale.csv'):
                risultati = pd.read_csv('best_results/risultati_hp_tuning_finale.csv')
                risultati = pd.concat([risultati, results])
                risultati.to_csv('best_results/risultati_hp_tuning_finale.csv', index=False)
            else:
                pd.DataFrame(results).to_csv('best_results/risultati_hp_tuning_finale.csv', index=False)
            
            # aggiorno il contatore per l'indice
            index += 1
            if os.path.exists('best_results/index.json'):
                with open('best_results/index.json', 'r') as f:
                    indici = json.load(f)
                indici['index_hp_eval'] = index
                with open('best_results/index.json', 'w') as f:
                    json.dump(indici, f)
            else:
                with open('best_results/index.json', 'w') as f:
                    json.dump({'index_hp_eval': index}, f)

        # al termine dei cicli ordino i risultati in base alla metrica finale e salvo i risultati
        if can_save:
            risultati_finali = pd.read_csv('best_results/risultati_hp_tuning_finale.csv')
            risultati_finali = risultati_finali.sort_values(by='final_metric', ascending=False)
            risultati_finali.to_csv('best_results/risultati_hp_tuning_finale.csv', index=False)
    
    # Funzione per eseguire tutti gli step della pipeline
    def run(self):
        if os.path.exists('step.json'):
            with open('step.json', 'r') as f:
                l = json.load(f)
                step = l['step']
        else:
            step = 0
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
        # con il seguente costrutto if controllo quale fase eseguire se precedentemente era stata interrotta
        if step == 0:
            print('\n\n\n---Fase 1---\n\n\n')
            self.fase1_clustering_evaluation_combinazioni_feature()
            self.fase1_clustering_evaluation_tutte_features()
            self.fase1_estrazione_dei_migliori_risultati()
            step += 1
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
        if step == 1:
            print('\n\n\n---Fase 2---\n\n\n')
            self.fase2_preliminary_results_evaluation()
            step += 1
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
        if step == 2:
            print('\n\n\n---Fase 3---\n\n\n')
            self.fase3_hyperparameter_tuning()
            step += 1
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
        if step == 3:
            print('\n\n\n---Fase 4---\n\n\n')
            self.fase4_hyperparameter_tuning_evaluation()
            step += 1
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
        if step == 4:
            print('\n\n\n---Fase 5---\n\n\n')
            grafici()
            step += 1
            d = {'step': step}
            with open('step.json', 'w') as f:
                json.dump(d, f)
