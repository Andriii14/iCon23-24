import pandas as pd
import numpy as np
import os
import logging
from joblib import dump, load
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import uniform, randint
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline  # Usa il pipeline di imbalanced-learn
import matplotlib
matplotlib.use('TkAgg')  # Usa il backend 'Agg' per la generazione di grafici non interattivi
import matplotlib.pyplot as plt
from utility import preprocess_data  # Importa la funzione di preprocessing


# ---------------------------------------------------------------------
# Funzioni di utilità
# ---------------------------------------------------------------------

# Configura il logging
def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Mantiene anche l'output sulla console
        ]
    )

def evaluate_with_cross_validation(model, X, y, param_distributions, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3), n_iter=50):
    """Esegue la valutazione incrociata e l'ottimizzazione degli iperparametri."""
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv, scoring='accuracy', n_jobs=-1)
    random_search.fit(X, y)
    logging.info("Migliori iperparametri: %s", random_search.best_params_)
    logging.info("Miglior punteggio (accuratezza media): %s", random_search.best_score_)
    return random_search.best_estimator_


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3), show_graph=True, save_path=None):
    """
    Addestra il modello e valuta le prestazioni sia con cross-validation che sul set di test.
    Gestisce la visualizzazione e il salvataggio delle curve di apprendimento e delle matrici di confusione.

    Args:
        model: Il modello da addestrare.
        X_train (pd.DataFrame): Dati di addestramento.
        y_train (pd.Series): Target di addestramento.
        X_test (pd.DataFrame): Dati di test.
        y_test (pd.Series): Target di test.
        cv (StratifiedKFold): Strategia di cross-validation da utilizzare.
        show_graph (bool): Se True, mostra i grafici.
        save_path (str): Percorso di base per salvare i grafici. Se None, i grafici non vengono salvati.
    """
    # Valutazione incrociata
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    logging.info("Mean cross-validation accuracy: %s", cv_scores.mean())

    # Addestramento del modello
    model.fit(X_train, y_train)

    # Predizione sul set di test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    model_name = model.named_steps['clf'].__class__.__name__  # Ottieni il nome del classificatore
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred, zero_division=1))
    logging.info("%s Accuracy on test set: %s", model_name, accuracy)

    metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }

    # Percorsi specifici per i grafici
    learning_curve_path = os.path.join(save_path, f"{model_name}_learning_curve.png") if save_path else None
    confusion_matrix_path = os.path.join(save_path, f"{model_name}_confusion_matrix.png") if save_path else None

    # Traccia la curva di apprendimento e salva il grafico
    plot_learning_curve(model, X_train, y_train, title=f"Learning Curve - {model_name}", cv=cv, show_graph=show_graph, save_path=learning_curve_path)

    # Visualizza e salva la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    if save_path:
        plt.savefig(confusion_matrix_path)
        print(f"Confusion matrix saved as '{confusion_matrix_path}'")
    if show_graph:
        plt.show()
    else:
        plt.close()

    return metrics


def visualize_metrics_graphs(metrics_results, show_graph=True, save_path=None):
    """
    Visualizza graficamente le metriche di vari modelli.

    Args:
        metrics_results (dict): Dizionario contenente le metriche per ogni modello.
        show_graph (bool): Se True, mostra il grafico.
        save_path (str): Percorso per salvare il grafico. Se None, il grafico non viene salvato.
    """
    models = list(metrics_results.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([metrics_results[clf]['accuracy'] for clf in models])
    precision = np.array([metrics_results[clf]['precision'] for clf in models])
    recall = np.array([metrics_results[clf]['recall'] for clf in models])
    f1 = np.array([metrics_results[clf]['f1'] for clf in models])

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    plt.bar(index, accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, f1, bar_width, label='F1')

    # Aggiunta di etichette e legenda
    plt.xlabel('Modelli')
    plt.ylabel('Punteggi medi')
    plt.title('Punteggio medio per ogni modello')
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    models_comparison_path = os.path.join(save_path, f"models_comparison.png") if save_path else None

    if save_path:
        plt.savefig(models_comparison_path)
        print(f"Graph of models comparison saved as '{models_comparison_path}'")

    if show_graph:
        plt.show()
    else:
        plt.close()


def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), show_graph=True, save_path=None):
    """
    Genera la curva di apprendimento per un modello, mostrando l'errore di training e l'errore di validazione.

    Args:
        estimator: Modello da valutare.
        X (array-like): Feature di input.
        y (array-like): Target.
        title (str): Titolo del grafico.
        cv (int or cross-validation generator): Numero di fold per la cross-validation.
        n_jobs (int): Numero di processi paralleli per la cross-validation.
        train_sizes (array-like): Quote dei dati di addestramento da utilizzare.
        show_graph (bool): Se True, mostra il grafico.
        save_path (str): Percorso per salvare il grafico. Se None, il grafico non viene salvato.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Error")

    # Calcola le curve di apprendimento
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )

    # Converti i punteggi in errori
    train_errors = 1 - np.mean(train_scores, axis=1)
    validation_errors = 1 - np.mean(valid_scores, axis=1)

    # Calcola le deviazioni standard degli errori
    train_errors_std = np.std(1 - train_scores, axis=1)
    validation_errors_std = np.std(1 - valid_scores, axis=1)

    plt.grid()

    # Visualizza gli errori con le aree di confidenza
    plt.fill_between(train_sizes, train_errors - train_errors_std,
                     train_errors + train_errors_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, validation_errors - validation_errors_std,
                     validation_errors + validation_errors_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_errors, 'o-', color="g", label="Training error")
    plt.plot(train_sizes, validation_errors, 'o-', color="r", label="Test error")

    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved as '{save_path}'")

    if show_graph:
        plt.show()
    else:
        plt.close()


def calculate_normalized_weights(text_columns):
    # Se c'è solo una colonna, assegna il peso intero a quella colonna
    if len(text_columns) == 1:
        return {text_columns[0]: 1.0}

    # Calcolo dei pesi decrescenti
    total_columns = len(text_columns)
    base_weights = {col: 2**(total_columns - idx - 1) for idx, col in enumerate(text_columns)}

    # Calcola la somma totale dei pesi
    total_sum = sum(base_weights.values())

    # Normalizza i pesi in modo che sommino a 1
    normalized_weights = {col: weight / total_sum for col, weight in base_weights.items()}

    return normalized_weights


# ---------------------------------------------------------------------
# Custom Transformer
# ---------------------------------------------------------------------


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Seleziona una colonna specifica dal DataFrame."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column_name].astype(str).values
        else:
            raise ValueError("Il trasformatore ColumnSelector si aspetta un DataFrame come input.")


class WeightedTransformer(BaseEstimator, TransformerMixin):
    """Custom Transformer per applicare un peso al risultato della trasformazione TF-IDF."""
    def __init__(self, transformer, weight=1.0):
        self.transformer = transformer
        self.weight = weight

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X) * self.weight


# ---------------------------------------------------------------------
# Funzioni principali (possono essere chiamate da un altro file)
# ---------------------------------------------------------------------


def train_models(input_file='audible_italiano_cleaned.csv', save_model=False, models_to_train=None, show_graphs=True, save_graphs_path=None, text_columns=None):
    """
    Funzione principale per caricare i dati, addestrare e valutare i modelli con l'aggiunta di RandomOverSampler
    per bilanciare il dataset e prevenire l'overfitting. Gestisce anche la visualizzazione e il salvataggio dei grafici.

    Args:
        input_file (str): Nome del file di input contenente i dati pre-elaborati.
        save_model (bool): Se True, salva i modelli addestrati e i dati associati in un file.
        models_to_train (list): Lista di modelli da addestrare. Default a ['SVM', 'Random_Forest', 'Logistic_Regression'].
        show_graphs (bool): Se True, mostra i grafici.
        save_graphs_path (str): Percorso base dove salvare i grafici. Se None, i grafici non vengono salvati.
    """

    # Carica i dati pre-elaborati
    data = preprocess_data(input_file, save_output=True)  # Esegue il preprocessing dei dati per migliorare le prestazioni dei modelli

    log_file = "training_evaluation.log"
    setup_logging(log_file)

    # Definisci le colonne con feature testuali e i loro pesi
    if text_columns is None:
        text_columns = ['summary', 'tags', 'title', 'author', 'subtitle', 'series', 'narrator', 'publisher']


    # Calcolo dei pesi decrescenti e normalizzazione
    normalized_weights = calculate_normalized_weights(text_columns)
    print("Normalized Column Weights:", normalized_weights)

    X = data[text_columns]
    y = data['target']

    # Divisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea una pipeline per ogni colonna testuale con il peso applicato
    pipelines = [(col, Pipeline([
        ('selector', ColumnSelector(col)),
        ('tfidf', WeightedTransformer(TfidfVectorizer(), weight=normalized_weights[col]))
    ])) for col in text_columns]

    # Combina le pipeline in una FeatureUnion
    feature_union = FeatureUnion(pipelines)

    # --- Dizionari di distribuzioni di iperparametri per RandomizedSearchCV ---
    # Dizionario base per i parametri TF-IDF comuni a tutte le colonne
    tfidf_params = {
        f'features__{col}__tfidf__transformer__max_features': [1000, 2000, 3000, None] for col in text_columns
    }

    # Dizionari di iperparametri per i modelli, includendo i parametri TF-IDF
    naive_bayes_param_distributions = {**tfidf_params, 'clf__alpha': uniform(0.1, 2.0)}
    svm_param_distributions = {**tfidf_params,
                               'clf__C': uniform(0.1, 100.0),
                               'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                               'clf__gamma': ['scale', 'auto'] + list(np.linspace(0.001, 1.0, 3))}
    random_forest_param_distributions = {**tfidf_params,
                                         'clf__n_estimators': randint(50, 301),
                                         'clf__max_depth': [None, 5, 10, 20],
                                         'clf__min_samples_split': randint(2, 11),
                                         'clf__min_samples_leaf': randint(1, 5),
                                         'clf__criterion': ['gini', 'entropy', 'log_loss']}
    decision_tree_param_distributions = {**tfidf_params,
                                         'clf__criterion': ['gini', 'entropy', 'log_loss'],
                                         'clf__max_depth': [None, 5, 10],
                                         'clf__min_samples_split': randint(2, 21),
                                         'clf__min_samples_leaf': randint(1, 21),
                                         'clf__splitter': ['best', 'random']}
    logistic_regression_param_distributions = {**tfidf_params,
                                               'clf__C': uniform(0.001, 100.0),
                                               'clf__penalty': ['l2'],
                                               'clf__solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
                                               'clf__max_iter': randint(100, 1001)}

    # --- Modelli disponibili ---
    available_models = {
        'Naive_Bayes': (MultinomialNB(), naive_bayes_param_distributions),
        'SVM': (SVC(kernel='linear'), svm_param_distributions),
        'Random_Forest': (RandomForestClassifier(), random_forest_param_distributions),
        'Decision_Tree': (DecisionTreeClassifier(), decision_tree_param_distributions),
        'Logistic_Regression': (LogisticRegression(), logistic_regression_param_distributions)
    }

    # Se non vengono specificati modelli, utilizza quelli predefiniti
    if models_to_train is None:
        models_to_train = ['Naive_Bayes', 'SVM', 'Random_Forest', 'Decision_Tree', 'Logistic_Regression']

    # Filtra i modelli da addestrare
    models_and_params = {name: available_models[name] for name in models_to_train if name in available_models}

    if save_graphs_path and not os.path.exists(save_graphs_path):
        os.makedirs(save_graphs_path)

    best_models = {}
    metrics_results = {}
    for name, (model, params) in models_and_params.items():
        logging.info(f"\n--- {name} ---")

        if name == 'SVM':
            # Per SVM, calcoliamo i pesi delle classi ma non facciamo oversampling
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            model.set_params(class_weight=class_weight_dict)
            pipeline = Pipeline([
                ('features', feature_union),
                ('clf', model)
            ])
        else:
            # Per gli altri modelli, usiamo RandomOverSampler per bilanciare le classi
            pipeline = ImbPipeline([
                ('features', feature_union),
                ('oversample', RandomOverSampler(random_state=42)),
                ('clf', model)
            ])

        # Valutazione incrociata e ottimizzazione degli iperparametri
        best_models[name] = evaluate_with_cross_validation(pipeline, X_train, y_train, params)

        # Addestramento e valutazione sul set di test completo
        metrics_results[name] = train_and_evaluate_model(best_models[name], X_train, y_train, X_test, y_test, show_graph = show_graphs, save_path = save_graphs_path)


    # Visualizzazione delle metriche
    visualize_metrics_graphs(metrics_results, show_graph = show_graphs, save_path = save_graphs_path)

    # Salvataggio di tutti i modelli migliori con i loro iperparametri, i dati di test e i tf-idf fittati
    if save_model:
        dump({
            'best_models': best_models
        }, 'best_models.joblib')
        print("Modelli salvati con successo.")
    else:
        print("Salvataggio dei modelli non eseguito.")


def classify_audiobook(df, index, model_name='Naive_Bayes', text_columns=None):
    """
    Funzione per classificare un audiolibro selezionato da un DataFrame utilizzando il modello salvato.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati degli audiolibri.
        index (int): Indice dell'audiolibro da classificare.
        model_name (str): Nome del modello da usare per la classificazione. Default è 'Naive_Bayes'.
        text_columns (list): Lista delle colonne di testo da considerare per la classificazione.

    Returns:
        predicted_label: Etichetta predetta per l'audiolibro.
    """
    # print("Inizio classificazione del nuovo esempio...")

    # Carica i modelli e i dati salvati
    data = load('best_models.joblib')
    best_models = data['best_models']

    # Verifica che l'indice sia valido
    if index not in df.index:
        raise ValueError(f"Indice '{index}' non trovato nel DataFrame.")

    if text_columns is None:
        text_columns = ['summary', 'tags', 'title', 'author', 'subtitle', 'series', 'narrator', 'publisher']

    # Estrai i dati dell'audiolibro per le colonne specificate
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    new_example = {col: df.loc[index, col] for col in text_columns if col in df.columns}

    # Verifica che il modello richiesto esista
    if model_name not in best_models:
        raise ValueError(f"Modello '{model_name}' non trovato. Disponibili: {list(best_models.keys())}")

    # Estrai il modello migliore
    model = best_models[model_name]

    # Trasforma il nuovo esempio usando il TF-IDF fittato
    feature_union = model.named_steps['features']

    # Crea un DataFrame con il nuovo esempio
    new_example_df = pd.DataFrame([new_example])

    # Trasforma il nuovo esempio
    X_new_transformed = feature_union.transform(new_example_df)

    # Effettua la previsione
    prediction = model.named_steps['clf'].predict(X_new_transformed)
    predicted_label = prediction[0]  # Assumiamo che `prediction` sia un array con un solo valore

    # print(f"Nuovo esempio classificato come: {predicted_label}")
    # print("Classificazione completata.")

    return predicted_label


def recommend_audiobooks(favorites_file, all_audiobooks_file, model_name='Naive_Bayes', text_columns=None):
    """
    Funzione per consigliare nuovi audiolibri basata sulla similarità con gli audiolibri preferiti.

    Args:
        favorites_file (str): File che contiene gli audiolibri preferiti dall'utente.
        all_audiobooks_file (str): File che contiene la lista completa di audiolibri.
        model_name (str): Nome del modello da usare per la raccomandazione. Default è 'Naive_Bayes'.
        text_columns (list): Colonne di testo da usare per il confronto e la raccomandazione.

    Returns:
        pd.DataFrame: Lista di audiolibri consigliati ordinata per similarità.
    """
    # Carica i modelli salvati
    data = load('best_models.joblib')
    best_models = data['best_models']

    # Carica i dataset
    favorites_df = pd.read_csv(favorites_file)
    all_audiobooks_df = pd.read_csv(all_audiobooks_file)

    # Verifica che il modello richiesto esista
    if model_name not in best_models:
        raise ValueError(f"Modello '{model_name}' non trovato. Disponibili: {list(best_models.keys())}")

    if text_columns is None:
        text_columns = ['summary', 'tags', 'title', 'author', 'subtitle', 'series', 'narrator', 'publisher']

    # Estrai il modello migliore
    model = best_models[model_name]

    # Trasforma i dati degli audiolibri preferiti e di tutti gli audiolibri usando il TF-IDF salvato
    feature_union = model.named_steps['features']
    favorites_features = feature_union.transform(favorites_df[text_columns])
    all_features = feature_union.transform(all_audiobooks_df[text_columns])

    # Calcola la similarità di coseno tra ogni audiolibro preferito e tutti gli altri
    similarity_matrix = cosine_similarity(favorites_features, all_features)

    # Calcola la media delle similarità per ogni audiolibro
    mean_similarities = similarity_matrix.mean(axis=0)

    # Aggiungi i punteggi di similarità al DataFrame degli audiolibri
    all_audiobooks_df['similarity_score'] = mean_similarities

    # Escludi gli audiolibri già presenti nella lista dei preferiti
    recommended_audiobooks = all_audiobooks_df[~all_audiobooks_df['title'].isin(favorites_df['title'])]

    # Ordina per punteggio di similarità
    recommended_audiobooks = recommended_audiobooks.sort_values(by='similarity_score', ascending=False)

    return recommended_audiobooks[['title', 'similarity_score']] # 'author',


