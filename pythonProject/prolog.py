from pyswip import Prolog
import csv
import re
import os

# Inizializza il motore Prolog
prolog = Prolog()

def csv_to_prolog(csv_file, prolog_file, csv_encoding="utf-8", prolog_encoding="utf-8"):
    """Converte un file CSV in un file Prolog, ordinando gli audiolibri per rating * review_count."""

    audiobooks = []

    with open(csv_file, 'r', encoding=csv_encoding) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Salta l'intestazione

        for row in reader:
            title = row[0].replace("'", "\\'")
            subtitle = row[1].replace("'", "\\'") if len(row) > 1 else ''
            series = row[2].replace("'", "\\'") if len(row) > 2 else ''
            author = row[3].replace("'", "\\'")
            narrator = row[4].replace("'", "\\'")
            duration = int(row[5])  # Assume duration is an integer
            rating = float(row[6]) if row[6] else 0  # Handle optional rating (convert to float)
            review_count = int(float(row[7])) if row[7] else 0  # Handle optional review count (convert to int)
            summary = row[8].replace("'", "\\'")  # Escape single quotes in summary
            category = row[9].replace("'", "\\'")
            subcategory = row[10].replace("'", "\\'")
            tags = row[11].replace("'", "\\'")  # Gestione del campo tags
            link = row[12]
            podcast_type = row[13].strip().lower()
            publisher = row[14].replace("'", "\\'")

            # Calcola il punteggio ponderato e aggiungi l'audiolibro alla lista
            score = rating * review_count
            audiobooks.append((
                score, title, subtitle, series, author, narrator, duration, rating, review_count, summary,
                category, subcategory, tags, link, podcast_type, publisher))

    # Ordina gli audiolibri per punteggio decrescente
    audiobooks.sort(reverse=True)

    with open(prolog_file, 'w', encoding=prolog_encoding) as prolog_file:
        for _, title, subtitle, series, author, narrator, duration, rating, review_count, summary, category, subcategory, tags, link, podcast_type, publisher in audiobooks:
            # Costruisci la stringa Prolog con l'ordine specificato
            fact_string = f"audiobook(\n  '{title}',\n  "
            fact_string += f"'{subtitle}',\n  "
            fact_string += f"'{series}',\n  "
            fact_string += f"'{author}',\n  "
            fact_string += f"'{narrator}',\n  "
            fact_string += f"{duration},\n  "
            fact_string += f"{rating},\n  "
            fact_string += f"{review_count},\n  "
            fact_string += f"'{summary}',\n  "
            fact_string += f"'{category}',\n  "
            fact_string += f"'{subcategory}',\n  "
            tag_string = tags.replace("['", "").replace("']", "").replace("'", "\\'").replace("[\"", "").replace("\"]",
                                                                                                                 "").split(
                ', ')
            fact_string += f"{tag_string},\n  "
            fact_string += f"'{link}',\n  "  # Gestione del campo link
            fact_string += f"'{podcast_type}',\n  "
            fact_string += f"'{publisher}'\n).\n"

            # Scrivi la stringa Prolog nel file
            prolog_file.write(fact_string)

        # Aggiungi le nuove regole Prolog
        prolog_file.write(
            r"""

% Relazione tra audiolibro e autore
written_by(Title, Author) :-
    audiobook(Title, _, _, Author, _, _, _, _, _, _, _, _, _, _, _).

% Relazione tra audiolibro e narratore
narrated_by(Title, Narrator) :-
    audiobook(Title, _, _, _, Narrator, _, _, _, _, _, _, _, _, _, _).

% Relazione tra audiolibro e categoria
belongs_to_category(Title, Category) :-
    audiobook(Title, _, _, _, _, _, _, _, _, Category, _, _, _, _, _).

% Relazione tra audiolibro e serie
belongs_to_series(Title, Series) :-
    audiobook(Title, _, Series, _, _, _, _, _, _, _, _, _, _, _, _).

% Relazione tra audiolibro e subcategoria
belongs_to_subcategory(Title, Subcategory) :-
    audiobook(Title, _, _, _, _, _, _, _, _, _, Subcategory, _, _, _, _).

% Relazione tra audiolibro e tipo di podcast
has_podcast_type(Title, PodcastType) :-
    audiobook(Title, _, _, _, _, _, _, _, _, _, _, _, _, PodcastType, _).

% Relazione tra audiolibro e editore
published_by(Title, Publisher) :-
    audiobook(Title, _, _, _, _, _, _, _, _, _, _, _, _, _, Publisher).

% Trova audiolibri per tag
audiobook_by_tag(Tag, Title) :-
    audiobook(Title, _, _, _, _, _, _, _, _, _, _, Tags, _, _, _),
    memberchk(Tag, Tags).

% Trova audiolibri con durata maggiore o uguale a un valore specificato
has_minimum_duration(Title, MinDuration) :-
    audiobook(Title, _, _, _, _, Duration, _, _, _, _, _, _, _, _, _),
    Duration >= MinDuration.

% Trova audiolibri con valutazione maggiore o uguale a un valore specificato
has_minimum_rating(Title, MinRating) :-
    audiobook(Title, _, _, _, _, _, Rating, _, _, _, _, _, _, _, _),
    Rating >= MinRating.

% Trova audiolibri con numero di recensioni maggiore o uguale a un valore specificato
has_minimum_review_count(Title, MinReviewCount) :-
    audiobook(Title, _, _, _, _, _, _, ReviewCount, _, _, _, _, _, _, _),
    ReviewCount >= MinReviewCount.

%% Regola per trovare audiolibri simili basata su più criteri
%% (modificare i criteri in base alle preferenze)
audiolibri_simili(Title, SimilarTitle) :-
    % Recupera i dettagli del titolo di riferimento
    written_by(Title, Author),
    belongs_to_category(Title, Category),
    belongs_to_subcategory(Title, Subcategory),
    %% narrated_by(Title, Narrator),
    published_by(Title, Publisher),
    has_podcast_type(Title, PodcastType),
    audiobook(Title, _, _, _, _, _, _, _, _, _, _, Tags, _, _, _),

    % Trova i dettagli per il titolo simile
    written_by(SimilarTitle, Author),
    belongs_to_category(SimilarTitle, Category),
    belongs_to_subcategory(SimilarTitle, Subcategory),
    %% narrated_by(SimilarTitle, Narrator),
    published_by(SimilarTitle, Publisher),
    has_podcast_type(SimilarTitle, PodcastType),
    audiobook_by_tag(SharedTag, SimilarTitle),
    memberchk(SharedTag, Tags),

    % Assicurati che non sia lo stesso titolo
    SimilarTitle \= Title,

    % Mostra i risultati
    nl, write('Trovato un audiolibro simile: '), writeln(SimilarTitle),
    write('Condivisi i seguenti attributi con il titolo di riferimento:'), nl,
    write('Autore: '), writeln(Author),
    write('Categoria: '), writeln(Category),
    write('Subcategoria: '), writeln(Subcategory),
    %% write('Narratore: '), writeln(Narrator),
    write('Editore: '), writeln(Publisher),
    write('Tipo di podcast: '), writeln(PodcastType),
    write('Tag condiviso: '), writeln(SharedTag)
.

:- use_module(library(random)).

% Conta il numero totale di audiolibri nel database
numero_audiolibri(Count) :-
    findall(Title, audiobook(Title, _, _, _, _, _, _, _, _, _, _, _, _, _, _), Titles),
    length(Titles, Count).

% Seleziona un titolo basato sull'indice
titolo_da_indice(Indice, Title) :-
    findall(Title, audiobook(Title, _, _, _, _, _, _, _, _, _, _, _, _, _, _), Titles),
    nth1(Indice, Titles, Title).

% Regola per ottenere N titoli casuali
titoli_casuali(N, TitoliCasuali) :-
    numero_audiolibri(Count),
    get_titoli_casuali(N, Count, [], TitoliCasuali).

% Funzione ricorsiva per selezionare titoli casuali
get_titoli_casuali(0, _, TitoliCasuali, TitoliCasuali).
get_titoli_casuali(N, Count, Accum, TitoliCasuali) :-
    N > 0,
    random_between(1, Count, IndiceCasuale),
    titolo_da_indice(IndiceCasuale, Titolo),
    \+ member(Titolo, Accum), % Assicurati che non ci siano duplicati
    N1 is N - 1,
    get_titoli_casuali(N1, Count, [Titolo|Accum], TitoliCasuali).

get_titoli_casuali(N, Count, Accum, TitoliCasuali) :-
    N > 0,
    get_titoli_casuali(N, Count, Accum, TitoliCasuali). % Riprova se trovi un duplicato

:- dynamic utente/1.
:- dynamic preferito/2.

"""
        )

def aggiungi_utente_in_prolog(nome_utente):
    """
    Aggiunge un nuovo utente a Prolog se non esiste già.
    """
    # Verifica se l'utente esiste già
    utenti = list(prolog.query(f"utente('{nome_utente}')"))
    if not utenti:
        # Se l'utente non esiste, lo aggiunge alla base di conoscenza
        prolog.assertz(f"utente('{nome_utente}')")
        print(f"Utente '{nome_utente}' creato e aggiunto al sistema.")
    else:
        print(f"Utente '{nome_utente}' esiste già nel sistema.")

def aggiungi_preferito(nome_utente, titolo):
    """
    Aggiunge un titolo alla lista dei preferiti dell'utente in Prolog se non esiste già.
    """
    # Escapa eventuali apostrofi presenti nel titolo con due apostrofi consecutivi
    titolo_escaped = titolo.replace("'", "\\'")

    # Verifica se il preferito esiste già
    preferiti = list(prolog.query(f"preferito('{nome_utente}', '{titolo_escaped}')"))
    if not preferiti:
        # Costruisci il fatto Prolog da asserire
        fatto_preferito = f"preferito('{nome_utente}', '{titolo_escaped}')"
        # Usa `assertz` per aggiungere il fatto
        prolog.assertz(fatto_preferito)
        print(f"{titolo} è stato aggiunto ai tuoi preferiti!")
    else:
        print(f"{titolo} è già presente nei tuoi preferiti.")

def escape_apostrofi(testo):
    """
    Escapa gli apostrofi nel testo.
    """
    return testo.replace("'", "\\'")

def scrivi_file_prolog():
    """
    Aggiunge i fatti utente e preferito al file Prolog esistente senza duplicati.
    """
    nome_file = "audible.pl"

    # Estrai i dati esistenti
    utenti = list(prolog.query("utente(X)"))
    preferiti = list(prolog.query("preferito(X, Y)"))

    # Leggi il contenuto del file esistente con codifica utf-8
    try:
        with open(nome_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []
    except UnicodeDecodeError:
        print("Errore di codifica del file. Assicurati che il file sia in formato UTF-8.")
        return

    existing_lines = set(line.strip() for line in lines)

    # Apri il file in modalità append con codifica utf-8
    with open(nome_file, "a", encoding="utf-8") as file:
        # Aggiungi i fatti utente
        for utente in utenti:
            fatto_utente = f"utente('{escape_apostrofi(utente['X'])}')."
            if fatto_utente not in existing_lines:
                file.write(fatto_utente + "\n")
                existing_lines.add(fatto_utente)

        # Aggiungi i fatti preferito
        for preferito in preferiti:
            fatto_preferito = f"preferito('{escape_apostrofi(preferito['X'])}', '{escape_apostrofi(preferito['Y'])}')."
            if fatto_preferito not in existing_lines:
                file.write(fatto_preferito + "\n")
                existing_lines.add(fatto_preferito)

    print("Dati aggiunti al file Prolog senza duplicati.")

def mostra_opzioni_utente():
    """
    Mostra un menu con le opzioni disponibili per l'utente.
    """
    print("\nScegli un'opzione:")
    print("1. Aggiungi titoli casuali ai tuoi preferiti")
    print("2. Trova audiolibri per autore")
    print("3. Trova audiolibri per categoria")
    print("4. Trova audiolibri per tag")
    print("5. Trova audiolibri per serie")
    print("6. Trova audiolibri per narratore")
    print("7. Trova audiolibri per durata minima")
    print("8. Trova audiolibri per rating minimo")
    print("9. Trova audiolibri per numero minimo di recensioni")
    print("10. Trova audiolibri per subcategoria")
    print("11. Trova audiolibri per tipo di podcast")
    print("12. Trova audiolibri per editore")
    print("13. Trova audiolibri simili ad un preferito")
    print("14. Rimuovi un preferito dalla lista")
    print("15. Esci")

    scelta = input("Inserisci il numero della tua scelta: ")
    return int(scelta)

def applica_regola(regola, parametro):
    """
    Applica una regola di Prolog con un parametro dato e restituisce i risultati.
    Prova prima il formato `regola(Titolo, Parametro)` e poi `regola(Parametro, Titolo)`.
    """
    # Determina se il parametro è un numero o una stringa e formatta di conseguenza
    if isinstance(parametro, (int, float)):
        # Se il parametro è un numero, non usare virgolette
        query1 = f"{regola}(Title, {parametro})"
        query2 = f"{regola}({parametro}, Title)"
    else:
        # Se il parametro è una stringa, usa virgolette
        parametro_escaped = parametro.replace("'", "''")
        query1 = f"{regola}(Title, '{parametro_escaped}')"
        query2 = f"{regola}('{parametro_escaped}', Title)"

    try:
        # Prova la prima query
        result = list(prolog.query(query1))
    except Exception as e:
        print(f"Errore durante l'esecuzione della query '{query1}': {e}")
        result = []

    # Se non ci sono risultati, prova la seconda query
    if not result:
        try:
            result = list(prolog.query(query2))
        except Exception as e:
            print(f"Errore durante l'esecuzione della query '{query2}': {e}")
            result = []

    # Restituisci i titoli trovati
    return [res["Title"] for res in result]

def gestisci_titoli_trovati(titoli, nome_utente):
    """
    Gestisce i titoli trovati e permette all'utente di aggiungere più titoli ai preferiti.
    """
    if not titoli:
        print("Nessun risultato trovato.")
        return

    print("\nTitoli trovati:")
    for i, titolo in enumerate(titoli, 1):
        print(f"{i}. {titolo}")

    # Richiedi all'utente di selezionare i titoli da aggiungere ai preferiti
    selezioni = input(
        "Inserisci i numeri dei titoli che vuoi aggiungere ai preferiti, separati da virgola (es. 1,2,3) oppure 0 per terminare: ")

    # Elaborare le selezioni
    numeri_selezionati = selezioni.split(',')
    numeri_selezionati = [int(num.strip()) for num in numeri_selezionati if num.strip().isdigit()]

    # Aggiungi i titoli selezionati ai preferiti
    for num in numeri_selezionati:
        numero = int(num)
        if 1 <= numero <= len(titoli):
            titolo_scelto = titoli[numero - 1]
            aggiungi_preferito(nome_utente, titolo_scelto)
        elif numero == 0:
            print("Operazione terminata.")
            break
        else:
            print(f"Numero '{numero}' non valido, ignorato.")

def trova_audiolibri_simili(nome_utente):
    """
    Trova audiolibri simili a uno di quelli nei preferiti dell'utente.
    """
    # Recupera i titoli preferiti dall'utente
    preferiti = list(prolog.query(f"preferito('{nome_utente}', Titolo)"))
    if not preferiti:
        print("Non hai ancora preferiti.")
        return

    print("\nI tuoi preferiti:")
    for i, pref in enumerate(preferiti, 1):
        print(f"{i}. {pref['Titolo']}")

    scelta = int(input("Inserisci il numero del titolo per trovare audiolibri simili: "))
    if 1 <= scelta <= len(preferiti):
        titolo_scelto = preferiti[scelta - 1]['Titolo']
        print(f"Sto cercando audiolibri simili a '{titolo_scelto}'...")
        simili = applica_regola("audiolibri_simili", titolo_scelto)
        gestisci_titoli_trovati(simili, nome_utente)
    else:
        print("Scelta non valida.")

def rimuovi_preferito(nome_utente, titolo):
    """
    Rimuove un titolo dalla lista dei preferiti dell'utente in Prolog e aggiorna il file Prolog.
    """
    # Escapa eventuali apostrofi presenti nel titolo
    titolo_escaped = titolo.replace("'", "''")

    # Verifica se il preferito esiste già
    preferiti = list(prolog.query(f"preferito('{nome_utente}', '{titolo_escaped}')"))
    if preferiti:
        # Rimuovi il fatto Prolog dalla base di conoscenza in memoria
        prolog.retract(f"preferito('{nome_utente}', '{titolo_escaped}')")
        print(f"{titolo} è stato rimosso dai tuoi preferiti.")

        # Aggiorna il file Prolog
        aggiorna_file_prolog(nome_utente, titolo_escaped)
    else:
        print(f"{titolo} non è presente nei tuoi preferiti.")

def aggiorna_file_prolog(nome_utente, titolo_escaped):
    """
    Aggiorna il file Prolog rimuovendo il fatto `preferito(nome_utente, titolo_escaped)`.
    """
    nome_file = "audible.pl"

    try:
        # Leggi tutte le righe del file
        with open(nome_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Scrivi nuovamente il file filtrando le righe che non corrispondono al fatto da rimuovere
        with open(nome_file, "w", encoding="utf-8") as file:
            for line in lines:
                # Costruisci la riga da confrontare
                riga_preferito = f"preferito('{nome_utente}', '{titolo_escaped}').\n"
                if line != riga_preferito:
                    file.write(line)

        print("File Prolog aggiornato correttamente.")
    except FileNotFoundError:
        print("File Prolog non trovato. Assicurati che il file esista.")
    except Exception as e:
        print(f"Si è verificato un errore durante l'aggiornamento del file Prolog: {e}")

def gestisci_utente():
    # Consulta il file Prolog generato
    prolog.consult("audible.pl")
    # Richiedi il nome dell'utente
    nome_utente = input("Inserisci il tuo nome utente: ")
    # Crea o verifica l'esistenza dell'utente
    aggiungi_utente_in_prolog(nome_utente)

    # Avvia il ciclo delle opzioni
    while True:
        # Mostra le opzioni disponibili per l'utente
        scelta = mostra_opzioni_utente()

        # Esegui l'azione in base alla scelta dell'utente
        if scelta == 1:
            # Aggiungi titoli casuali ai preferiti
            numero_casuale = int(input("Quanti audiolibri casuali vuoi aggiungere ai tuoi preferiti? "))

            query = f"titoli_casuali({numero_casuale}, TitoliCasuali)"
            # Esegui la query e ottieni un singolo risultato
            risultati = prolog.query(query)

            try:
                risultato = next(risultati)  # Ottiene il primo (e unico) risultato
                titoli_casuali = risultato['TitoliCasuali']
                print()
            finally:
                # Assicura la chiusura della query
                risultati.close()  # Chiude il generatore di risultati

            # Ora esegui l'aggiunta ai preferiti con la query chiusa
            for titolo in titoli_casuali:
                aggiungi_preferito(nome_utente, titolo)

            print(f"{numero_casuale} titoli casuali sono stati aggiunti ai tuoi preferiti.")

        elif scelta == 2:
            autore = input("Inserisci il nome dell'autore: ")
            titoli = applica_regola("written_by", autore)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 3:
            categoria = input("Inserisci la categoria: ")
            titoli = applica_regola("belongs_to_category", categoria)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 4:
            tag = input("Inserisci il tag: ")
            titoli = applica_regola("audiobook_by_tag", tag)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 5:
            serie = input("Inserisci la serie: ")
            titoli = applica_regola("belongs_to_series", serie)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 6:
            narratore = input("Inserisci il nome del narratore: ")
            titoli = applica_regola("narrated_by", narratore)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 7:
            min_durata = int(input("Inserisci la durata minima in minuti: "))
            titoli = applica_regola("has_minimum_duration", min_durata)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 8:
            min_rating = float(input("Inserisci il rating minimo (ad es. 4.0): "))
            titoli = applica_regola("has_minimum_rating", min_rating)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 9:
            min_review_count = int(input("Inserisci il numero minimo di recensioni: "))
            titoli = applica_regola("has_minimum_review_count", min_review_count)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 10:
            subcategoria = input("Inserisci la subcategoria: ")
            titoli = applica_regola("belongs_to_subcategory", subcategoria)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 11:
            tipo_podcast = input("Inserisci il tipo di podcast: ")
            titoli = applica_regola("has_podcast_type", tipo_podcast)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 12:
            editore = input("Inserisci il nome dell'editore: ")
            titoli = applica_regola("published_by", editore)
            gestisci_titoli_trovati(titoli, nome_utente)

        elif scelta == 13:
            trova_audiolibri_simili(nome_utente)

        elif scelta == 14:
            # Gestisci la rimozione dei preferiti
            preferiti = list(prolog.query(f"preferito('{nome_utente}', Titolo)"))
            if not preferiti:
                print("Non hai preferiti da rimuovere.")
            else:
                print("\nI tuoi preferiti:")
                for i, pref in enumerate(preferiti, 1):
                    print(f"{i}. {pref['Titolo']}")

                selezioni = input(
                    "Inserisci i numeri dei titoli che vuoi rimuovere dai preferiti, separati da virgola (es. 1,2,3) oppure 0 per terminare: ")

                numeri_selezionati = selezioni.split(',')
                numeri_selezionati = [num.strip() for num in numeri_selezionati if num.strip().isdigit()]

                for num in numeri_selezionati:
                    numero = int(num)
                    if 1 <= numero <= len(preferiti):
                        titolo_scelto = preferiti[numero - 1]['Titolo']
                        rimuovi_preferito(nome_utente, titolo_scelto)
                    elif numero == 0:
                        print("Operazione terminata.")
                        break
                    else:
                        print(f"Numero '{numero}' non valido, ignorato.")

        elif scelta == 15:
            # Salva i dati nel file Prolog
            scrivi_file_prolog()
            print("Uscita in corso...")
            break

        else:
            print("Scelta non valida. Riprova.")

def estrai_preferiti_da_prolog(utente, prolog_file):
    preferiti = []
    pattern = re.compile(r"preferito\('" + re.escape(utente) + r"',\s*'([^']+)'\)\.")

    with open(prolog_file, mode='r', encoding='utf-8') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                preferiti.append(match.group(1))

    return preferiti

def gestisci_preferiti(utente, input_file='audible_italiano_cleaned.csv', output_file='preferiti.csv'):
    preferiti = estrai_preferiti_da_prolog(utente, 'audible.pl')

    if not preferiti:
        print(f"Nessun titolo preferito dichiarato per l'utente {utente}.")
    else:
        temp_file = 'temp_audible.csv'  # File temporaneo per memorizzare il file senza i preferiti

        with open(input_file, mode='r', encoding='utf-8') as infile, \
            open(output_file, mode='w', encoding='utf-8', newline='') as outfile, \
            open(temp_file, mode='w', encoding='utf-8', newline='') as tempfile:

            reader = csv.reader(infile)
            writer_preferiti = csv.writer(outfile)
            writer_temp = csv.writer(tempfile)

            # Scrivi l'intestazione in entrambi i file di output
            header = next(reader)
            writer_preferiti.writerow(header)
            writer_temp.writerow(header)

            # Filtra e scrivi le righe corrispondenti
            for row in reader:
                title = row[0]  # Assumendo che il titolo sia nella prima colonna
                if title in preferiti:
                    writer_preferiti.writerow(row)  # Scrivi nel file preferiti.csv
                else:
                    writer_temp.writerow(row)  # Scrivi nel file temporaneo

        # Sostituisci il file originale con il file temporaneo
        os.remove(input_file)  # Elimina il file originale
        os.rename(temp_file, input_file)  # Rinomina il file temporaneo al nome originale

        print(f"I titoli preferiti per l'utente {utente} sono stati estratti e salvati in {output_file}, e le righe corrispondenti sono state rimosse da {input_file}.")

def trasferisci_preferiti(input_file='audible_italiano_cleaned.csv'):
    # Consulta il file Prolog generato
    prolog.consult("audible.pl")
    nome_utente = chiedi_nome_utente()

    if not nome_utente:
        return  # Esci se l'utente non esiste

    gestisci_preferiti(nome_utente, input_file)

def verifica_utente_esistente(nome_utente):
    utenti = list(prolog.query(f"utente('{nome_utente}')"))
    return bool(utenti)

def chiedi_nome_utente():
    nome_utente = input("Inserisci il tuo nome utente: ")
    if verifica_utente_esistente(nome_utente):
        print(f"Utente '{nome_utente}' trovato nel sistema.")
        return nome_utente
    else:
        print(f"Utente '{nome_utente}' non trovato. Assicurati di aver inserito il nome corretto.")
        return None

