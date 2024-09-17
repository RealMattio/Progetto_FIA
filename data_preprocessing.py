import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DataPreprocessing:
    def __init__(self, df:pd.DataFrame, manage_dummies:bool = False, dummies_columns:list = None):
        self.df = df
        self.data = df
        self.manage_dummies = manage_dummies
        self.dummies_columns = dummies_columns
    #La funzione clean_data deve eseguire diverse operazioni di pulizia dei dati, in particolare deve:
    #1) Riempire i dati mancanti nel dataset
    #2) Rimuovere i duplicati
    #3) Identificare ed eventualmente rimuovere possibili outliers
    #4) Effettuare lo smoothing del rumore dei dati
    def clean_data(self) -> pd.DataFrame:

        #self.df.fillna(self.df.mean(), inplace=True) sostituisco i valori mancanti con la media della colonna quindi con la media dei valori corrispondenti alla tipologia del dato mancante

        self.df.iloc[:,:-4].dropna(inplace=True)

        self.df.drop_duplicates(inplace=True) #rimuovo eventuali duplicati

        #z_score = np.abs(stats.zscore(self.df.select_dtypes(include=[np.number]))) seleziono colonne aventi come valori delle celle solo valori numerici, mentre con stats.zscore effettua il calcolo statistico per ogni valore estratto dalle colonne facendo la media 
                                                                                    #dei valori nella colonna corrispondente al valore sulla quale si sta effettuando il calcolo. ottengo un dataframe delle stesse dimenzioni ma con valori sostituiti dai corrispondenti z-score
        #self.df = self.df.mask(z_score>3) sostituisco gli outliers con NaN

        #for column in self.df.select_dtypes(include=[np.number]).columns: cosí come fatto prima seleziono colonne aventi solo contenuti numerici nelle celle e restituisce un dataframe 
            #self.df[column] = self.df[column].roling(Window=3, min_periods=1).mean()  per ogni colonna numerica applico una finestra mobile di lunghezza 3, che calcola la media. pongo min periods a 1 cosi che posso calcolare anche finestre di valori minori di tre
                                                                                    # utile per i primi valori della serie
        return self.df 

    #tale funzione ha il compito di normalizzare ed effettuare l'aggregazione dei dati del dataset
    #dove per aggregazione si intende trasformare le colonne di dati e orari nel dataset che possono assumere formati 
    #di stringhe o numerici in un formato di data e ora strutturato
    def transform_data(self) -> pd.DataFrame:

        #per la parte di aggregazione, alcune date all'interno del dataset potrebbero essere memorizzate come stringhe, per memorizzarle come oggetti date, si utilizza la funzione to_datetime()
        #l'argomento 'errors' garantisce che ciò che non può essere trasformato in un oggetto data non venga effettivamente trasformato
        #self.df['data_nascita'] = pd.to_datetime(self.df['data_nascita'], errors='coerce')
        #self.df['data_contatto'] = pd.to_datetime(self.df['data_contatto'], errors='coerce')
        #self.df['data_erogazione'] = pd.to_datetime(self.df['data_erogazione'], errors='coerce')
        self.df['data_nascita'] = pd.to_datetime(self.df['data_nascita'], utc=True)
        self.df['data_contatto'] = pd.to_datetime(self.df['data_contatto'], utc=True)
        self.df['data_erogazione'] = pd.to_datetime(self.df['data_erogazione'], utc=True)

        
        
        #Per comodità potrebbe essere utile introdurre la colonna età, ottenuta sottraendo la  data di nascita alla data e l'ora attuale
        #la funzione astype() converte il risultato in anni

        # Funzione per calcolare l'età
        self.df['eta'] = self.df['data_nascita'].apply(self.calculate_age)
        self.data['eta'] = self.data['data_nascita'].apply(self.calculate_age)
        
        # Creo la colonna 'fascia_età' basata sulla colonna 'eta'
        bins = [0, 11, 22, 45, 65, float('inf')]
        labels = ['0-11', '12-22', '23-45', '45-65', '66+']
        self.df['fascia_eta'] = pd.cut(self.df['eta'], bins=bins, labels=labels, right=True)
        

        #utilizziamo il modulo StandardScaler per normalizzare il dataset. l'obiettivo è quello di riscalare i dati, riportarli quindi alla stessa scala.
        #la funzione select_dtypes selezionerà tutte quante le colonne numeriche e le normalizzerà
        scaler = MinMaxScaler()
        #numerical_features = self.df.select_dtypes(include=['float64', 'int64']).columns
        #self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])
        self.df['eta'] = scaler.fit_transform(self.df[['eta']])
        #print(self.df['eta'])

        return self.df
    

    #tale funzione ha il compito di eliminare le colonne ridondanti del dataset. Il dataset è composto dalle colonne
    # id_prenotazione, id_paziente, data_nascita, sesso, regione_residenza, codice_regione_residenza, asl_residenza, codice_asl_residenza,
    # provincia_residenza, codice_provincia_residenza, comune_residenza, codice_comune_residenza, tipologia_servizio, descrizione_attivita, codice_descrizione_attivita,
    # data_contatto, regione_erogazione, codice_regione_erogazione, asl_erogazione, codice_asl_erogazione, provincia_erogazione, codice_provincia_erogazione,
    # struttura_erogazione, codice_struttura_erogazione, tipologia_struttura_erogazione, codice_tipologia_struttura_erogazione, id_professionista_sanitario,
    # tipologia_professionista_sanitario, codice_tipologia_professionista_sanitario, data_erogazione, ora_inizio_erogazione, ora_fine_erogazione, data_disdetta

    def reduce_data(self) -> pd.DataFrame:
        #Eliminazione delle colonne ridondanti: le colonne ridondanti sono i vari codici che identificano le regioni, le asl, le province, i comuni, le attività, le tipologie di professionisti sanitari e le strutture
        # viene eliminata la colonna id_paziente in quanto non è rilevante per il nostro scopo
        # viene eliminata la colonna descrizione_attivita in quanto è ridondante con la colonna codice_descrizione_attivita
        # viene eliminata la colonna tipologia_servizio in quanto si parla di Teleassistenza per ogni prestazione
        ### data = self.df[['id_paziente', 'codice_regione_residenza', 'codice_asl_residenza', 'codice_provincia_residenza', 'codice_comune_residenza', 'tipologia_servizio', 'descrizione_attivita', 'codice_regione_erogazione', 'codice_asl_erogazione', 'codice_provincia_erogazione', 'codice_struttura_erogazione', 'codice_tipologia_struttura_erogazione', 'codice_tipologia_professionista_sanitario']]
        
        self.df = self.df[['id_prenotazione', 'data_nascita', 'sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita', 'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario', 'tipologia_professionista_sanitario', 'data_erogazione', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_disdetta','eta', 'fascia_eta']]
        self.df['codice_descrizione_attivita'] = self.df['codice_descrizione_attivita'].astype('str')

        # le colonne rimanenti devono essere 'id_prenotazione', 'data_nascita', 'sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita',
        # 'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario',
        # 'tipologia_professionista_sanitario', 'data_erogazione', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_disdetta'


        #tra tutte le colonne rimanenti si trattano come variabili categoriche tutte trann i campi data e ora e gli id
        if self.manage_dummies:
            self.df = pd.get_dummies(self.df, columns=self.dummies_columns)
        return self.df

    #Funzione che richiama tutte le istanze delle funzioni precedentemente create
    def preprocessing_data(self) -> pd.DataFrame:
        self.clean_data()
        self.transform_data()
        self.reduce_data()
        return self.df
    
    def calculate_age(self, birthdate):
        today = datetime.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age



