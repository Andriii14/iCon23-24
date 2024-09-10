import pandas as pd
import os
from utility import clean_data, preprocess_data, installPackages
from supervised import train_models, classify_audiobook, recommend_audiobooks
from prolog import csv_to_prolog, gestisci_utente, trasferisci_preferiti
from unsupervised import raggruppa_e_bilancia_target

# installPackages() # decommentare per eseguire l'istallazione dei pacchetti necessari

if __name__ == "__main__":
    # Esegue la pulizia del dataset grezzo estratto mediante audible_scraper
    clean_data()

    filename = "audible.pl"
    # Controlla se il file esiste
    if not os.path.exists(filename):
        # Se non esiste, crealo vuoto
        open(filename, 'w').close()

    # Il dataset viene converito in fatti e regole mediante prolog per permettere un ragionamento logico
    csv_to_prolog(csv_file = "audible_italiano_cleaned.csv", prolog_file = "audible.pl")
    print(f"Converted CSV file to Prolog facts in: audible.pl")
    gestisci_utente()

    # Raggruppa le etichette target, bilancia i cluster e analizza i risultati
    data_clustered = raggruppa_e_bilancia_target(pd.read_csv('audible_italiano_cleaned.csv'), show_graphs=True) #,  save_graphs_path='grafici'

    # Salva il dataset con i target raggruppati
    data_clustered.to_csv('audible_italiano_clustered.csv', index=False)
    print("Clustering dei target completato e salvato nel file 'audible_italiano_clustered.csv'.")

    # Elimina i preferiti dalla lista e crea un file separato preferiti.csv dove inserirli
    trasferisci_preferiti(input_file='audible_italiano_clustered.csv')

    # Apprendimento dei modelli e valutazione degli iperparametri (può richiedere abbastanza tempo per l'esecuzione)
    # train_models(input_file='audible_italiano_clustered.csv', save_model=True, models_to_train=['Naive_Bayes'],  show_graphs=True, text_columns=['summary', 'tags', 'title']) #,  save_graphs_path='grafici'

    # Carica il DataFrame dei preferiti dal file CSV
    df = preprocess_data(input_file='preferiti.csv', save_output=False)
    # Lista per salvare le predizioni
    predictions = []
    # Itera su ogni riga del DataFrame
    for index, row in df.iterrows():
        # Applica la funzione di classificazione alla riga corrente
        prediction = classify_audiobook(df, index, model_name='SVM', text_columns=['summary', 'tags', 'title'])
        predictions.append(prediction)

    # Aggiungi le predizioni come una nuova colonna nel DataFrame
    df['prediction'] = predictions
    # Confronta le predizioni con i valori reali della colonna target
    correct_predictions = df['prediction'] == df['target']
    # Calcola il numero di predizioni corrette
    num_correct = correct_predictions.sum()
    total = len(df)
    # Calcola l'accuratezza
    accuracy = num_correct / total
    print(f"Numero di predizioni corrette: {num_correct} su {total}")
    print(f"Accuratezza del modello: {accuracy * 100:.2f}%")

    # Chiamata alla funzione per ottenere le raccomandazioni
    raccomandazioni = recommend_audiobooks(favorites_file = 'preferiti.csv', all_audiobooks_file = 'audible_italiano_clustered.csv', text_columns=['summary', 'tags', 'title'])

    # Mostra i primi 10 audiolibri consigliati
    print(raccomandazioni.head(10))
