import pandas as pd
class Clustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k
    
    def clustering(self) -> pd.DataFrame:
        return self.data