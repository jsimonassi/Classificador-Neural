import math

import pandas as pd
import zipfile


class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        """Extrai o arquivo zip e lê o arquivo csv"""
        try:
            with zipfile.ZipFile(self.file_path + ".zip", 'r') as zip_ref:
                zip_ref.extractall('./DataSet')
            df = pd.read_csv(self.file_path)

            # Remove linhas sem polaridade definida
            # droped_data = df.drop(df[(df.polarity != 1.0) & (df.polarity != 0.0)].index)
            return df
        except FileNotFoundError:
            raise Exception('File not found')
