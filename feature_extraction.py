import pandas as pd
class FeatureExtraction:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def extract(self) -> pd.DataFrame:
        return self.df