# Progetto di Fondamenti di Intelligenza Artificiale A.A. 2023/2024
_Supervised Clustering with target variable: an application in Teleassistance_
L'obiettivo del progetto è profilare i pazienti tenendo conto del loro contributo all'aumento del servizio di teleassistenza. È importante identificare modelli e comportamenti comuni in base all'aumento delle teleassistenze dovute ai pazienti standard.

Progetto realizzato da:
- Mattia Muraro
- Edoardo Giacometto
- Andrea Scazzeri

## Come eseguire il codice
Per eseguire il codice è necessario eseguire i seguenti step:
1. Scaricare tutti i pacchetti richiesti aprendo il prompt nella cartella del progetto e eseguire il comando `pip install -r requirements.txt`
2. Eseguire il file `main.py`
   Per via della produzione di un grande numero di file, sono stati caricati su GitHub solo i file necessari all'osservazione dei risultati. Durante l'esecuzione del codice verranno prodotte nella folder del progetto altre sottocartelle necessarie all'esecuzione corretta di tutta la pipeline. Esse non vanno in alcun modo eliminate fintanto che la pipeline non sarà completata.
   Attenzione: la pipeline è pensata per poter essere interrotta e ripresa da dove era stata interrotta, semplicemente fermando l'esecuzione con il comando `ctrl+c`.

## Spiegazione del codice
La pipeline è articolata in 5 fasi:
1. La prima fase si occupa di effettuare il clustering su tutti i possibili sottoinsiemi di features, ottenendo un preliminare indice di bontà del clustering. Per fare questo si sfrutta come indice la purezza, per via della sua facilità di calcolo. All'interno di questa fase, ma anche delle fasi successive, il dataset viene preprocessato e viene eseguita la feature extraction.
2. La seconda fase si occupa di valutare i migliori risultati precedentemente ottenuti andando a calcolare la silhouette e la metrica finale. Per questi calcoli il tempo è decisamente maggiore, pertanto tali verifiche sono state effettuate solo sui sottinsiemi di features che forniscono i migliori valori di purezza.
3. La terza fase è la fase dell'hyperparameters tuning. In questa fase si utilizzano le migliori features per calcolare gli iperparametri del modello di clustering utilizzato migliori. Come nella fase 1, si utilizza la purezza per capire preliminarmente quali iperparametri forniscono risultati migliori.
4. La quarta fase valuta la silhouette la maetrica finale per i migliori risultati ottenuti durante l'hyperparameters tuning. Alla fine di questa fase, i due migliori sottinsiemi di features costituiscono la soluzione del problema.
5. L'ultima fase rappresenta i grafici in modo da avere una visualizzazione dei risultati ottenuti 
