import pandas as pd

def load_data(file_path):
    """
    Load the dataset from the provided CSV file path.
    """
    data = pd.read_csv(file_path)
    return data
