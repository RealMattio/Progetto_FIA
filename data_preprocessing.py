import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    #La funzione clean_data deve eseguire diverse operazioni di pulizia dei dati, in particolare deve:
    #1) Riempire i dati mancanti nel dataset
    #2) Rimuovere i duplicati
    #3) Identificare ed eventualmente rimuovere possibili outliers
    #4) Effettuare lo smoothing del rumore dei dati
    def clean_data(self) -> pd.DataFrame:
        return self.df 

    #tale funzione ha il compito di normalizzare ed effettuare l'aggregazione dei dati del dataset
    #dove per aggregazione si intende trasformare le colonne di dati e orari nel dataset che possono assumere formati 
    #di stringhe o numerici in un formato di data e ora strutturato
    def transform_data(self) -> pd.DataFrame:

        #per la parte di aggregazione, alcune date all'interno del dataset potrebbero essere memorizzate come stringhe, per memorizzarle come oggetti date, si utilizza la funzione to_datetime()
        #l'argomento 'errors' garantisce che ciò che non può essere trasformato in un oggetto data non venga effettivamente trasformato
        self.df['data_nascita'] = pd.to_datetime(self.df['data_nascita'], errors='coerce')
        self.df['data_contatto'] = pd.to_datetime(self.df['data_contatto'], errors='coerce')
        self.df['data_erogazione'] = pd.to_datetime(self.df['data_erogazione'], errors='coerce')

        #Per comodità potrebbe essere utile introdurre la colonna età, ottenuta sottraendo la  data di nascita alla data e l'ora attuale
        #la funzione astype() converte il risultato in anni
        self.df['eta'] = (pd.Timestamp('now') - self.df['data_nascita']).astype('<m8[Y]')

        #utilizziamo il modulo StandardScaler per normalizzare il dataset. l'obiettivo è quello di riscalare i dati, riportarli quindi alla stessa scala.
        #la funzione select_dtypes selezionerà tutte quante le colonne numeriche e le normalizzerà
        scaler = StandardScaler()
        numerical_features = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])

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
        
        data = self.df[['id_prenotazione', 'data_nascita', 'sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita', 'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario', 'tipologia_professionista_sanitario', 'data_erogazione', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_disdetta']]

        # le colonne rimanenti devono essere 'id_prenotazione', 'data_nascita', 'sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita',
        # 'data_contatto', 'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 'id_professionista_sanitario',
        # 'tipologia_professionista_sanitario', 'data_erogazione', 'ora_inizio_erogazione', 'ora_fine_erogazione', 'data_disdetta'


        #tra tutte le colonne rimanenti si trattano come variabili categoriche tutte trann i campi data e ora e gli id
        data = pd.get_dummies(data, columns=['sesso', 'regione_residenza', 'asl_residenza', 'provincia_residenza', 'comune_residenza', 'codice_descrizione_attivita', 
                                               'regione_erogazione', 'asl_erogazione', 'provincia_erogazione', 'struttura_erogazione', 'tipologia_struttura_erogazione', 
                                             'tipologia_professionista_sanitario'])
        return data

    #Funzione che richiama tutte le istanze delle funzioni precedentemente create
    def preprocessing_data(self) -> pd.DataFrame:
        df = self.clean_data()
        df = self.transform_data()
        df = self.reduce_data()
        return self.df



