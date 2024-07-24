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
    

    #tale funzione ha il compito di eliminare le colonne ridondanti del dataset
    def reduce_data(self) -> pd.DataFrame:
        return self.df

    #Funzione che richiama tutte le istanze delle funzioni precedentemente create
    def preprocessing_data(self) -> pd.DataFrame:
        df = self.clean_data()
        df = self.transform_data()
        df = self.reduce_data()
        return self.df



