import pandas as pd
from datetime import datetime as dt


class FeatureExtraction:

    def __init__(self, df:pd.DataFrame):
        '''
        costruttore della classe FeatureExtraction
        :param df: dataframe contenente i dati
        '''
        self.df = df
        self.incrementi = None

    def assign_label(self):
        '''
        Assegna la label 'incremento_teleassistenze' ad ogni elemento del dataframe principale in base al fatto che l'istanza
        appartenga o meno a quel quadrimestre
        '''
        for index, row in self.incrementi.iterrows():
            self.df.loc[(self.df['anno'] == row['anno']) & (self.df['quadrimestre'] == row['quadrimestre']), 'incremento_teleassistenze'] = row['incremento_teleassistenze']

    def extract(self) -> pd.DataFrame:
        # Estraggo l'anno e il quadrimestre dalla data di erogazione
        self.df['anno'] = self.df['data_erogazione'].dt.year
        #print(self.df)

        # Estraggo il mese dalla colonna di date, producendo un numero intero tra 1 e 12.
        # riduco il valore del mese di 1. Questo perché i mesi sono numerati da 1 a 12, ma per il calcolo del quadrimestre è più conveniente lavorare con un indice che parte da 0.
        # divido il valore ottenuto per 4 così da dividere l'anno in quadrimestri
        #+ 1 converte l'indice zero-based ottenuto dalla divisione in un quadrimestre 1-based. 
        
        self.df['quadrimestre'] = ((self.df['data_erogazione'].dt.month - 1) // 3) + 1

        # Raggruppo i dati per anno e quadrimestre e conto il numero di teleassistenze
        self.incrementi = self.df.groupby(['anno', 'quadrimestre']).size().reset_index(name='numero_teleassistenze')

        # Calcolo l'incremento tra i quadrimestri corrispondenti
        self.incrementi['incremento'] = self.incrementi.groupby(['quadrimestre'])['numero_teleassistenze'].diff().fillna(0)

        self.incrementi['incrementi_periodici'] = self.incrementi['numero_teleassistenze'].diff().fillna(0)

        # Discretizzo la variabile target 'incremento_teleassistenze'
        bins = [-float('inf'), -10000, 0, 10000, float('inf')]
        #bins = [0, 2000, 5000, 10000, float('inf')]
        
        labels = ['Grande Decremento', 'Piccolo Decremento', 'Piccolo Incremento', 'Grande Incremento']
        #labels = ['COSTANT', 'LOW', 'MEDIUM', 'HIGH']
        #self.incrementi['incremento_teleassistenze'] = pd.cut(self.incrementi['incrementi_periodici'].abs(), bins=bins, labels=labels)
        self.incrementi['incremento_teleassistenze'] = pd.cut(self.incrementi['incremento'], bins=bins, labels=labels)
        #print(self.incrementi)
        
        # Assegno la label ad ogni elemento del dataframe principale
        self.assign_label()
        #print(self.df)
        return self.df