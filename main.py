import pipeline

# Creazione di un'istanza della classe Pipeline
p = pipeline.Pipeline('data/challenge_campus_biomedico_2024.csv', clustering_type='kmodes', n_cluster=4)
p.run_hp_tuning()

print('Pipeline eseguita')