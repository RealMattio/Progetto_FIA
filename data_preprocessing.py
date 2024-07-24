import pandas as pd
import numpy as np
from scipy import stats
class DataPreprocessing:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    #La funzione clean_data deve eseguire diverse operazioni di pulizia dei dati, in particolare deve:
    #1) Riempire i dati mancanti nel dataset
    #2) Rimuovere i duplicati
    #3) Identificare ed eventualmente rimuovere possibili outliers
    #4) Effettuare lo smoothing del rumore dei dati
    def clean_data(self) -> pd.DataFrame:

        #self.df.fillna(self.df.mean(), inplace=True) sostituisco i valori mancanti con la media della colonna quindi con la media dei valori corrispondenti alla tipologia del dato mancante

        self.df.dropna(inplace=True)

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



