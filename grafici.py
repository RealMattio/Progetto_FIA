import pandas as pd
import chart
import os 
import ast
from tqdm import tqdm
import data_preprocessing as dp
import feature_extraction as fe

def grafici():
    directory = 'plots/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    results = pd.read_csv('best_results/risultati_hp_tuning_finale.csv')
    silhouettes = pd.read_csv('best_results/silhouettes_hp_tuning_finale.csv')
    clusters = pd.read_csv('best_results/hp_tuning_cluster_assigned_finale.csv')
    data_raw = pd.read_parquet('data/challenge_campus_biomedico_2024.parquet')

    data_preprocessing = dp.DataPreprocessing(data_raw)
    data = data_preprocessing.preprocessing_data()

    feature_extractor = fe.FeatureExtraction(data)
    data = feature_extractor.extract()
    data = data[data['anno'] != 2019]
    data_raw = data_raw.iloc[data.index]
    print('\n\n\n\n\n\n')

    # per ogni riga presente nel file risultati_hp_tuning_finale.csv creo una cartella nel quale inserisco i plot di ogni feature
    for index, row in tqdm(results.iterrows(), total=results.shape[0], desc='Creazione grafici'):
        n_clusters = row['n_cluster']
        it = row['iter_on_best_performances']
        kmodes_init_type = row['kmodes_init_type']
        kmodes_n_init = row['kmodes_n_init']
        features = ast.literal_eval(row['features'])

        # kmodes_4_clusters_features_at_0_Huang_neighbors_init_4
        col = f'kmodes_{n_clusters}_clusters_features_at_{it}_{kmodes_init_type}_neighbors_init_{kmodes_n_init}'
        clus = clusters[col]
        clusters_name = clus.unique()

        # silhouette_kmodes_5_clusters_features_at_3_Huang_neighbors_init_9
        sil = silhouettes['silhouette_'+col]

        for c in clusters_name:
            #sil_c = sil.loc[clus == c]
            #chart.plot_silhouette(sil_c, cluster_label=c, save=True, show=False, directory=f'{directory}{index}_metrica_finale_{round(row['final_metric'],2)}/')
            #chart.plot_silhouette_colors(sil_c, cluster_label=c, save=True, show=False, directory=f'{directory}{index}_metrica_finale_{round(row['final_metric'],2)}/')
            for feature in features[:-1]:
                titolo = feature.replace('_', ' ')
                # se la feature è un codice devo usare il valore corrispondente nel dataset di partenza quindi cambio il nome della feature
                if feature.startswith('codice'):
                    feature = feature.replace('codice_', '')
                    data_to_graph = data_raw.iloc[clus[clus == c].index]
                else:
                    data_to_graph = data.iloc[clus[clus == c].index]
                chart.plot_bar(data_to_graph, feature, titolo=titolo, cluster_label=c, show = False, save = True, directory=f'{directory}{index}_metrica_finale_{round(row['final_metric'],2)}/{feature}/')
                chart.plot_pie(data_to_graph, feature, titolo=titolo, cluster_label=c, show = False, save = True, directory=f'{directory}{index}_metrica_finale_{round(row['final_metric'],2)}/{feature}/')
                chart.feature_values(data_to_graph, feature, save = True, directory=f'{directory}{index}_metrica_finale_{round(row['final_metric'],2)}/{feature}/')