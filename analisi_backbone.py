import pandas as pd

df = pd.read_csv('lab1_backbone_evaluation.csv')
model_group = df.groupby('Model')
backbone_group = df.groupby('Backbone')

metrics = ['BA_Test', 'ACC_Test']  # le tue colonne

for m in metrics:
    print(f'Best {m}')
    print(df.loc[df[m].idxmax()], end='\n\n')