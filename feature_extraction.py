import pandas as pd
from datetime import datetime as dt
class FeatureExtraction:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        print(self.df['data_erogazione'])
        # Estraggo l'anno e il quadrimestre dalla data di erogazione
        self.df['anno'] = self.df['data_erogazione'].dt.year

        # Estraggo il mese dalla colonna di date, producendo un numero intero tra 1 e 12.
        # riduco il valore del mese di 1. Questo perché i mesi sono numerati da 1 a 12, ma per il calcolo del quadrimestre è più conveniente lavorare con un indice che parte da 0.
        # divido il valore ottenuto per 4 così da dividere l'anno in quadrimestri
        #+ 1 converte l'indice zero-based ottenuto dalla divisione in un quadrimestre 1-based. 
        
        self.df['quadrimestre'] = ((self.df['data_erogazione'].dt.month - 1) // 4) + 1

        # Raggruppo i dati per anno e quadrimestre e conto il numero di teleassistenze
        self.df = self.df.groupby(['anno', 'quadrimestre']).size().reset_index(name='numero_teleassistenze')

        # Calcolo l'incremento tra i quadrimestri corrispondenti
        self.df['incremento'] = self.df.groupby(['quadrimestre'])['numero_teleassistenze'].diff().fillna(0)

        # Discretizzo la variabile target 'incremento_teleassistenze'
        bins = [-float('inf'), 0, 10, 50, float('inf')]
        labels = ['Costant', 'Low', 'Medium', 'High']
        self.df['incremento_teleassistenze'] = pd.cut(self.df['incremento'], bins=bins, labels=labels)

        return self.df