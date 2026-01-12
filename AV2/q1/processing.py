import kagglehub
from pathlib import Path
import pandas as pd

def ler_datasets():
    """Carregar os datasets diretamente do kaggle"""
    path = kagglehub.dataset_download("moesiof/portuguese-narrative-essays")
    path = Path(path)

    print("Path to dataset files:", path)

    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    validation = pd.read_csv(path / "validation.csv")

    return train, test, validation