# classe per la creazione dei grafici e per il calcolo delle statistiche
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#funzione per il plot della silhouette: essa prende in ingresso tutti i valori della silhouette e li plotta
#in un grafico a barre orizzontali
def plot_silhouette(silhouette_vals:pd.DataFrame, cluster_label = 0):
    # Plot dei valori di silhouette
    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    y_lower = 10
    silhouette_vals.sort_values(ascending=True, inplace=True)

    size_cluster = silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    cmap = plt.cm.get_cmap('RdYlGn')
    colors = cmap(np.linspace(0, 1, len(silhouette_vals)))

    #color = 'b' # colore blu
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_vals, facecolor=colors, edgecolor=colors, alpha=0.7)

    #plt.text(-0.05, y_lower + 0.5 * size_cluster, f'Cluster: {cluster_label}')
    y_lower = y_upper + 10  # 10 per separare i plot dei cluster
    mean_silhouette = silhouette_vals.mean()
    ax.set_xlim([-1, 1])
    plt.axvline(x=mean_silhouette, color="red", linestyle="--")
    plt.title(f"Plot dei coefficienti di silhouette per il cluster {cluster_label}")
    plt.xlabel("Valore del coefficiente di silhouette")
    plt.ylabel("Cluster")
    plt.show()

# funzione per il plot della silhouette ma stavolta le barre sono colorate in funzione del valore del coefficiente
def plot_silhouette_colors(silhouette_vals:pd.DataFrame, cluster_label = 0):
    # Plot dei valori di silhouette
    # Numero di cluster (lunghezza dei valori della silhouette)
    n_points = len(silhouette_vals)
    plt.figure(figsize=(10, 6))
    silhouette_vals.sort_values(ascending=True, inplace=True)
    ax = plt.axes()
    # Creazione del colormap
    cmap = plt.cm.get_cmap('RdYlGn')

    # Normalizzazione dei valori per ottenere i colori corrispondenti
    norm = plt.Normalize(vmin=min(silhouette_vals), vmax=max(silhouette_vals))
    colors = cmap(norm(silhouette_vals))

    # Plot delle silhouette in un grafico a barre orizzontali
    plt.barh(range(n_points), silhouette_vals, color=colors)


    mean_silhouette = silhouette_vals.mean()
    # Titolo e label degli assi
    plt.axvline(x=mean_silhouette, color="red", linestyle="--")
    ax.set_xlim([-1, 1])
    plt.title(f"Plot dei coefficienti di silhouette per il cluster {cluster_label}")
    plt.xlabel("Valore del coefficiente di silhouette")
    plt.ylabel("Cluster")
    plt.show()

# funzione per il calcolo dei valori presenti in una features, delle occorrenze e delle percentuali sul totale
def feature_values(data:pd.DataFrame, feature:str):
    # Calcola le occorrenze di ciascuna categoria
    counts = data[feature].value_counts()
    # Calcola la percentuale di occorrenza
    percentages = (counts / counts.sum()) * 100
    # Crea un DataFrame con le colonne richieste
    result_df = pd.DataFrame({
        feature: counts.index,
        'Percentage': percentages.values,
        'Count': counts.values
    })
    
    # Salva il DataFrame in un file CSV
    csv_feature_analysis = f"{feature}_analysis.csv"
    result_df.to_csv(csv_feature_analysis, index=False)

# funzione per il plot di un grafico a torta
def plot_pie(data:pd.DataFrame, feature:str, titolo:str = 'Nome Feature'):
    counts = data[feature].value_counts()
    cmap = plt.cm.get_cmap('rainbow')
    # Genera una lista di colori usando linspace per ottenere esattamente il numero di colori che servono
    colors = cmap(np.linspace(0, 1, len(counts)))

    # Plot del grafico a torta
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title(f"Distribuzione della feature {titolo}")
    plt.show()

# funzione per il plot di un grafico a barre
def plot_bar(data:pd.DataFrame, feature:str, titolo:str = 'Nome Feature'):
    counts = data[feature].value_counts()
    cmap = plt.cm.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(counts)))
    # Crea il grafico a barre
    plt.figure(figsize=(8, 6))
    counts.plot(kind='bar', color=colors)
    plt.title('Occorrenze della feature ' + titolo)
    plt.xlabel('Valore')
    plt.ylabel('Occorrenze')
    plt.xticks(rotation=75)  # Mantiene le etichette orizzontali
    plt.show()