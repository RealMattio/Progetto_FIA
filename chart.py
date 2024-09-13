# classe per la creazione dei grafici e per il calcolo delle statistiche
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def pie_perc(pct):
    return f'{pct:.1f}%' if pct > 2.5 else ''
def label_filter(pct, label):
    return label if pct > 2.5 else ''

#funzione per il plot della silhouette: essa prende in ingresso tutti i valori della silhouette e li plotta
#in un grafico a barre orizzontali
def plot_silhouette(silhouette_vals:pd.DataFrame, cluster_label = 0, show = True, save = False, directory:str = ''):
    # Plot dei valori di silhouette
    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    y_lower = 10
    silhouette_vals.sort_values(ascending=True, inplace=True)

    size_cluster = silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    color = 'b' # colore blu
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    #plt.text(-0.05, y_lower + 0.5 * size_cluster, f'Cluster: {cluster_label}')
    y_lower = y_upper + 10  # 10 per separare i plot dei cluster
    mean_silhouette = silhouette_vals.mean()
    ax.set_xlim([-1, 1])
    plt.axvline(x=mean_silhouette, color="red", linestyle="--")
    plt.title(f"Plot dei coefficienti di silhouette per il cluster {cluster_label}")
    plt.xlabel("Valore del coefficiente di silhouette")
    plt.ylabel("Cluster")
    if save:
        if os.path.exists(directory):
            plt.savefig(f'{directory}silhouette_cluster_{cluster_label}.png', dpi = 500)
        else:
            os.mkdir(directory)
            plt.savefig(f'{directory}silhouette_cluster_{cluster_label}.png', dpi = 500)
    if show:
        plt.show()
    plt.close()

# funzione per il plot della silhouette ma stavolta le barre sono colorate in funzione del valore del coefficiente
def plot_silhouette_colors(silhouette_vals:pd.DataFrame, cluster_label = 0, show = True, save = False, directory:str = ''):
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
    if save:
        if os.path.exists(directory):
            plt.savefig(f'{directory}silhouette_color_cluster_{cluster_label}.png', dpi = 500)
        else:
            os.mkdir(directory)
            plt.savefig(f'{directory}silhouette_color_cluster_{cluster_label}.png', dpi = 500)
    if show:
        plt.show()
    plt.close()

# funzione per il calcolo dei valori presenti in una features, delle occorrenze e delle percentuali sul totale
def feature_values(data:pd.DataFrame, feature:str, save = True, directory:str = '', cluster_label = 0):
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
    if save:
        if os.path.exists(directory):
            csv_feature_analysis = f"{directory}{feature}_analysis_cluster_{cluster_label}.csv"
            result_df.to_csv(csv_feature_analysis, index=False)
        else:
            os.mkdir(directory)
            csv_feature_analysis = f"{directory}{feature}_analysis_cluster_{cluster_label}.csv"
            result_df.to_csv(csv_feature_analysis, index=False)

# funzione per il plot di un grafico a torta
def plot_pie(data:pd.DataFrame, feature:str, titolo:str = 'Nome Feature', show = True, save = False, directory:str = '', cluster_label = 0):
    counts = data[feature].value_counts()
    cmap = plt.cm.get_cmap('rainbow')
    # Genera una lista di colori usando linspace per ottenere esattamente il numero di colori che servono
    colors = cmap(np.linspace(0, 1, len(counts)))

    # Plot del grafico a torta
    plt.figure(figsize=(13, 8))
    font_size = max(10, min(12, 30 - len(counts)))

    percentages = [count / sum(counts) * 100 for count in counts]

    # Applica le etichette solo per percentuali > 2.5
    filtered_labels = [label_filter(pct, label) for pct, label in zip(percentages, counts.index)]

    plt.pie(counts, labels=filtered_labels, autopct=lambda pct:pie_perc(pct), startangle=90, colors=colors, textprops={'fontsize': font_size})
    plt.title(f"Distribuzione della feature {titolo} nel cluster {cluster_label}")
    if save:
        if os.path.exists(directory):
            plt.savefig(f'{directory}pie_plot_{feature}_cluster_{cluster_label}.png', dpi = 1000)
        else:
            os.mkdir(directory)
            plt.savefig(f'{directory}pie_plot_{feature}_cluster_{cluster_label}.png', dpi = 1000)
    if show:
        plt.show()
    plt.close()

# funzione per il plot di un grafico a barre
def plot_bar(data:pd.DataFrame, feature:str, titolo:str = 'Nome Feature', cluster_label = 0, show = True, save = False, directory:str = ''):
    counts = data[feature].value_counts()
    cmap = plt.cm.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(counts)))
    # Crea il grafico a barre
    plt.figure(figsize=(10, 8))
    counts.plot(kind='bar', color=colors)
    plt.title('Occorrenze della feature ' + titolo +' nel cluster ' + str(cluster_label))
    plt.xlabel('Valore')
    plt.ylabel('Occorrenze')
    font_size = max(5, min(10, 30 - len(counts)))
    plt.xticks(rotation=90, fontsize = font_size)  # Mantiene le etichette orizzontali e imposta la dimensione del font
    plt.subplots_adjust(bottom=0.35)
    if save:
        if os.path.exists(directory):
            plt.savefig(f'{directory}bar_plot_{feature}_cluster{cluster_label}.png', dpi = 1000)
        else:
            os.mkdir(directory)
            plt.savefig(f'{directory}bar_plot_{feature}_cluster{cluster_label}.png', dpi = 1000)
    if show:
        plt.show()
    plt.close()
    