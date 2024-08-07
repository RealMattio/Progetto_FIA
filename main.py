import pipeline

# Creazione di un'istanza della classe Pipeline
p = pipeline.Pipeline('data/challenge_campus_biomedico_2024.csv', clustering_type='kmenas', n_clusters=4)
p.run()

print('Pipeline eseguita')