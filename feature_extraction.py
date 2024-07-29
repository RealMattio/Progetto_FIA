import pandas as pd
from datetime import datetime as dt
class FeatureExtraction:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        
        # Controllo iniziale dei dati
        #print("Tipo di dato della colonna 'data_erogazione':", self.df['data_erogazione'].dtype)
        #print("Valori NaT nella colonna 'data_erogazione':", self.df['data_erogazione'].isna().sum())

        # Conversione in datetime
        self.df['data_erogazione'] = pd.to_datetime(self.df['data_erogazione'], errors='coerce')

        # Verifica se la conversione è avvenuta correttamente
        if pd.api.types.is_datetime64_any_dtype(self.df['data_erogazione']):
            self.df['anno'] = self.df['data_erogazione'].dt.year
            self.df['quadrimestre'] = ((self.df['data_erogazione'].dt.month - 1) // 4) + 1
            print(self.df)
            #self.data = self.df
            # Raggruppo i dati per anno e quadrimestre e conto il numero di teleassistenze
            self.counts = self.df.groupby(['anno', 'quadrimestre']).size().reset_index(name='numero_teleassistenze')

            # Calcolo l'incremento tra i quadrimestri corrispondenti
            self.counts['incremento'] = self.counts.groupby(['anno'])['numero_teleassistenze'].diff().fillna(0)
            print(self.df)

            # Discretizzo la variabile target 'incremento_teleassistenze'
            bins = [-float('inf'), 0, 10, 50, float('inf')]
            labels = ['Costant', 'Low', 'Medium', 'High']
            self.counts['incremento_teleassistenze'] = pd.cut(self.counts['incremento'], bins=bins, labels=labels)

            self.df = self.df.merge(self.counts, on=['anno', 'quadrimestre'], how='left')
        else:
            print("Errore: la colonna 'data_erogazione' non è di tipo datetime.")

        return self.df