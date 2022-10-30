import pandas as pd


class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise Exception('File not found')
