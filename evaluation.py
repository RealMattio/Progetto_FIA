class ClusteringEvaluation:
    def __init__(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions
    
    def evaluate(self) -> dict:
        pass