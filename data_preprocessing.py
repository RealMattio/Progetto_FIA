import pandas as pd

#funzione per legge il dataset
def load_data(path ='data\challenge_campus_biomedico_2024_sample.csv'):
    return pd.read_csv(path)

#La funzione clean_data deve eseguire diverse operazioni di pulizia dei dati, in particolare deve:
#1) Riempire i dati mancanti nel dataset
#2) Rimuovere i duplicati
#3) Identificare ed eventualmente rimuovere possibili outliers
#4) Effettuare lo smoothing del rumore dei dati
def clean_data(df):
    return df

#tale funzione ha il compito di normalizzare ed effettuare l'aggregazione dei dati del dataset
#dove per aggregazione si intende trasformare le colonne di dati e orari nel dataset che possono assumere formati 
#di stringhe o numerici in un formato di data e ora strutturato
def transform_data(df):
    return df

#tale funzione ha il compito di eliminare le colonne ridondanti del dataset
def reduce_data(df):
    return df

#Funzione che richiama tutte le istanze delle funzioni precedentemente create
def preprocessing_data(df):
    df = load_data(path ='data\challenge_campus_biomedico_2024_sample.csv')
    df = clean_data(df)
    df = transform_data(df)
    df = reduce_data(df)
    return df



