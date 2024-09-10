import re
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import subprocess
import sys
from tqdm import tqdm
import importlib.metadata

# Verifica e carica il modello italiano di spaCy
try:
    nlp = spacy.load('it_core_news_sm')
except OSError:
    print("Modello 'it_core_news_sm' non trovato. Sto scaricando il modello...")
    spacy.cli.download('it_core_news_sm')
    nlp = spacy.load('it_core_news_sm')

# Verifica la presenza delle stopwords italiane in NLTK e carica lo stemmer italiano
try:
    stopwords.words('italian')
except LookupError:
    print("Stopwords italiane non trovate. Le sto scaricando...")
    import nltk
    nltk.download('stopwords')

# Carica il modello italiano di spaCy
nlp = spacy.load('it_core_news_sm')
# Carica il modello italiano di SnowballStemmer
stemmer = SnowballStemmer("italian")


# Funzione compatta per installare le librerie necessarie
def installPackages():
    try:
        with open("requirements.txt", "r") as file:
            packages = [pkg.strip() for pkg in file.readlines()]
    except FileNotFoundError:
        print("Il file requirements.txt non è stato trovato.")
        return

    for package in tqdm(packages, desc="Installazione pacchetti"):
        try:
            importlib.metadata.version(package)
            print(f"Il pacchetto {package} è già installato.")
        except importlib.metadata.PackageNotFoundError:
            try:
                print(f"Installazione del pacchetto: {package}")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except subprocess.CalledProcessError:
                print(f"Errore durante l'installazione di {package}.")

    print("Verifica dei pacchetti installati...")
    for package in tqdm(packages, desc="Verifica pacchetti"):
        try:
            importlib.metadata.version(package)
            print(f"Il pacchetto {package} è installato correttamente.")
        except importlib.metadata.PackageNotFoundError:
            print(f"Errore: Il pacchetto {package} non è stato installato correttamente.")

def clean_data(input_file='audible_italiano_uncleaned.csv', output_file='audible_italiano_cleaned.csv'):
    """
    Funzione che esegue il cleaning e la predisposizione dei dati.

    Args:
        input_file (str): Nome del file di input. Default: 'audible_italiano_uncleaned.csv'
        output_file (str): Nome del file di output. Default: 'audible_italiano_cleaned.csv'
    """

    data = pd.read_csv(input_file)  # Leggi i dati CSV in un DataFrame
    original_rows = data.shape[0]  # Calcola il numero di righe nel DataFrame originale

    # Riempi valori mancanti e calcola durata
    for col in ['review_count', 'duration', 'rating']: data[col] = data[col].fillna(0)
    data['duration'] = data['duration'].str.extract(r'([0-9]+)(?: ore| ora)').fillna(0).astype(int) * 60 \
                       + data['duration'].str.extract('([0-9]+) min').fillna(0).astype(int)

    # Processa colonne testuali e rating
    data['rating'] = data['rating'].apply(lambda x: float(x.replace(',', '.')) if x else None)
    for col in ['subtitle', 'series', 'narrator', 'publisher']: data[col] = data[col].fillna(' ')

    # Elimina righe con valori mancanti e duplicati, elabora 'summary' e 'tags'
    data.dropna(subset=['title', 'author', 'summary', 'category', 'subcategory', 'tags'], inplace=True)
    data['summary'] = data['summary'].astype(str).str.split('©').str[
        0]  # elimina il testo dopo e compreso il carattere speciale ©
    data = data.drop_duplicates(subset=['title'])  # elimina duplicati su tutto il dataset
    data['tags'] = data['tags'].str.split(', ').apply(lambda x: ', '.join(sorted(x)))  # ordina alfabeticamente i tags

    # Definisci la colonna target
    data['target'] = data[['category', 'subcategory']].apply(lambda row: ' -> '.join(row.astype(str)), axis=1)

    # Rimuove target con pochi esempi
    threshold = 50  # Soglia per il numero minimo di esempi per target
    target_counts = data['target'].value_counts()
    targets_to_remove = target_counts[target_counts < threshold].index
    data = data[~data['target'].isin(targets_to_remove)]

    # Salva e stampa risultati
    duplicate_rows_removed = original_rows - data.shape[0]
    data.to_csv(output_file, index=False)
    print(
        f"\nIl cleaning ha rimosso {duplicate_rows_removed} righe, lasciando un totale di {data.shape[0]} audiolibri unici nel dataset.")

def preprocess_data(input_file='audible_italiano_cleaned.csv',
                    output_file='preprocessed_data.csv',
                    columns_to_preprocess=['summary'],
                    columns_to_exclude=['link', 'review_1', 'review_2', 'review_3', 'review_4', 'review_5', 'review_6', 'review_7', 'review_8', 'review_9', 'review_10', 'target'],
                    save_output=False):
    """
    Funzione che esegue il preprocessing delle colonne del dataset.

    Args:
        input_file (str): Nome del file di input (output di clean_data). Default: 'audible_italiano_cleaned.csv'
        output_file (str): Nome del file di output. Default: 'preprocessed_data.csv'
        columns_to_preprocess (list): Lista delle colonne a cui applicare rimozione stop words e stemming.
        columns_to_exclude (list): Lista delle colonne da escludere da qualsiasi preprocessing.
        save_output (bool): Se True, salva il file preprocessed_data.csv. Default: False.
    """

    data = pd.read_csv(input_file)

    def normalize_text(text):
        """
        Normalizza il testo rimuovendo la punteggiatura, gli spazi bianchi extra e convertendo in minuscolo.
        """
        text = re.sub(r'[^\w\s]', '', text)  # Rimuovi la punteggiatura
        text = re.sub(r'\s+', ' ', text)  # Rimuovi gli spazi bianchi extra
        text = text.lower()  # Converti in minuscolo
        return text

    def remove_stop_words(text):
        """
        Rimuove le stop words (parole comuni e poco informative) dal testo.
        """
        stop_words = set(stopwords.words('italian'))
        return " ".join([word for word in text.split() if word not in stop_words])

    def stem_text(text):
        """
        Applica lo stemming al testo.
        """
        return " ".join([stemmer.stem(word) for word in text.split()])

    # Identifica le colonne testuali
    text_columns = data.select_dtypes(include='object').columns.tolist()

    # Normalizzazione per tutte le colonne testuali, escludendo quelle specificate
    for col in text_columns:
        if col not in columns_to_exclude:
            data[col] = data[col].astype(str).apply(normalize_text)

    # Preprocessing (remove_stop_words e stem_text) per colonne specifiche, escludendo quelle in columns_to_exclude
    for col in columns_to_preprocess:
        if col not in columns_to_exclude:
            data[col] = data[col].apply(remove_stop_words) \
                                  .apply(stem_text)

    # Salva il file solo se save_output è True
    if save_output:
        data.to_csv(output_file, index=False)
        print(f"Il preprocessing è completo. I dati sono stati salvati in {output_file}")
    else:
        print("Il preprocessing è completo. I dati non sono stati salvati.")

    return data



