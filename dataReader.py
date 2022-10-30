import pandas as pd
import zipfile

class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        try:
            with zipfile.ZipFile(self.file_path + ".zip", 'r') as zip_ref:
                zip_ref.extractall('./DataSet')
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise Exception('File not found')
