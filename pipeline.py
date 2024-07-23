# aggiungere tutti gli import necessari per eseguire opgni step della pipeline
import pandas as pd

# Description: Classe che si occupa di eseguire tutti i passaggi del pipeline
class Pipeline:

    # Costruttore della classe che si occuperà di eseguire tutti i passaggi del pipeline
    def __init__(self, path ='data\challenge_campus_biomedico_2024_sample.csv'):
        self.path = path


    #funzione per legge il dataset
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)
    

    # Metodo che esegue la pipeline
    def run(self):
        print(f"Running pipeline from data in {self.path}")
        print("Reading file")
        


        
