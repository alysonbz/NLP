import kagglehub
import shutil
import os
import pandas as pd

destination_folder = "C:/Users/Luciana/NLP/AV1"  # Substitua pelo caminho desejado
os.makedirs(destination_folder, exist_ok=True)


path = kagglehub.dataset_download("moesiof/portuguese-narrative-essays")
for file_name in os.listdir(path):
    source = os.path.join(path, file_name)
    destination = os.path.join(destination_folder, file_name)
    shutil.move(source, destination)
print("Dataset salvo em:", destination_folder)

dataset1 = pd.read_csv('test.csv')
dataset2 = pd.read_csv('train.csv')
dataset3 = pd.read_csv('validation.csv')

concatenated = pd.concat([dataset1, dataset2, dataset3], axis=0, ignore_index=True)
concatenated.to_csv('dataset_final.csv', index=False)