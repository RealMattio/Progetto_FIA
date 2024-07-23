# aggiungere tutti gli import necessari per eseguire opgni step della pipeline


# Description: Classe che si occupa di eseguire tutti i passaggi del pipeline
class Pipeline:

    # Costruttore della classe che si occuperà di eseguire tutti i passaggi del pipeline
    def __init__(self, path):
        self.path = path
    

    # Metodo che esegue la pipeline
    def run(self):
        print(f"Running pipeline from data in {self.path}")
        print("Reading file")
        


        
