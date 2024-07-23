import pandas as pd
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



