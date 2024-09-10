from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('TkAgg')  # Usa il backend 'Agg' per la generazione di grafici non interattivi
import matplotlib.pyplot as plt


# Verifica se le stopwords italiane sono già installate
try:
    stop_words = stopwords.words('italian')
except LookupError:
    print("Stopwords italiane non trovate. Le sto scaricando...")
    nltk.download('stopwords')
    stop_words = stopwords.words('italian')


def calcolaCluster(dataSet, n_clusters):
    """
    Esegue il K-means clustering sui dati forniti e restituisce le etichette dei cluster e i centroidi.

    Args:
        dataSet (array-like): Dati numerici per il clustering.
        n_clusters (int): Numero di cluster desiderati.

    Returns:
        tuple: Una coppia contenente (etichette dei cluster, centroidi dei cluster).
    """
    # Inizializzazione di KMeans con il metodo 'k-means++' per una migliore partenza
    km = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42)

    # Addestramento del modello KMeans sui dati
    km.fit(dataSet)

    # Recupero delle etichette dei cluster e dei centroidi
    etichette = km.labels_
    centroidi = km.cluster_centers_

    return etichette, centroidi


def regolaGomito(dataSet, show_graph=True, save_path=None):
    """
    Calcola il numero di cluster ottimale per il dataset mediante il metodo del gomito.
    Args:
        dataSet: Dati per il clustering.
        show_graph (bool): Se True, mostra il grafico.
        save_path (str): Percorso per salvare il grafico. Se None, il grafico non viene salvato.
    """
    inertia = []
    maxK = 30  # Numero massimo di cluster da testare
    for i in range(1, maxK):
        kmeans = KMeans(n_clusters=i, n_init=10, init='k-means++', random_state=42)
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)

    kl = KneeLocator(range(1, maxK), inertia, curve="convex", direction="decreasing")

    if show_graph or save_path:
        plt.plot(range(1, maxK), inertia, 'bx-')
        plt.scatter(kl.elbow, inertia[kl.elbow - 1], c='red', label=f'Miglior k: {kl.elbow}')
        plt.xlabel('Numero di Cluster (k)')
        plt.ylabel('Inertia')
        plt.title('Metodo del gomito per trovare il k ottimale')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            print(f"Graph of elbow rules saved as '{save_path}'")
        if show_graph:
            plt.show()
        plt.close()

    return kl.elbow

def visualizza_distribuzione_cluster(etichette, show_graph=True, save_path=None):
    """
    Visualizza il grafico a torta del rapporto degli esempi per ogni cluster.
    Args:
        etichette: Etichette dei cluster.
        show_graph (bool): Se True, mostra il grafico.
        save_path (str): Percorso per salvare il grafico. Se None, il grafico non viene salvato.
    """
    unique, counts = np.unique(etichette, return_counts=True)
    if show_graph or save_path:
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=[f'Cluster {i}' for i in unique], autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        plt.title('Distribuzione degli esempi per ogni cluster')
        if save_path:
            plt.savefig(save_path)
            print(f"Graph of cluster distribution saved as '{save_path}'")
        if show_graph:
            plt.show()
        plt.close()

def raggruppa_e_bilancia_target(data, show_graphs=True, save_graphs_path=None):
    """
    Raggruppa le etichette target e bilancia i cluster.
    Args:
        data: DataFrame con i dati.
        show_graphs (bool): Se True, mostra i grafici.
        save_graphs_path (str): Directory base dove salvare i grafici.
    """
    if save_graphs_path and not os.path.exists(save_graphs_path):
        os.makedirs(save_graphs_path)

    targets = data['target']
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words=stop_words)
    X_tfidf = vectorizer.fit_transform(targets)

    elbow_graph_path = os.path.join(save_graphs_path, "elbow_graph.png") if save_graphs_path else None
    n_clusters = regolaGomito(X_tfidf, show_graph=show_graphs, save_path=elbow_graph_path)

    etichette, _ = calcolaCluster(X_tfidf, n_clusters)

    distribution_graph_path = os.path.join(save_graphs_path, "cluster_distribution.png") if save_graphs_path else None
    visualizza_distribuzione_cluster(etichette, show_graph=show_graphs, save_path=distribution_graph_path)

    data['target'] = etichette
    return data



