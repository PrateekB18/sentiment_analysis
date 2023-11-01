import pandas as pd
import numpy as np
import re

class DataProcessor:
    def __init__(self, file_path = "data/processed/sentiment_dataset.csv"):
        self.file_path = file_path
        self.df = self.load_data()
        self.texts, self.sentiments = self.process_data()

    def load_data(self):
        df = pd.read_csv(self.file_path, index_col=0)
        df.dropna(inplace=True)
        return df

    @staticmethod
    def clean_text(text):
        if type(text) == np.float64:
            return ""
        else:
            text = str(text).lower()
            text = re.sub("'", "", text)  # to avoid removing contractions in English
            text = re.sub("@[A-Za-z0-9_]+", "", text)
            text = re.sub("#[A-Za-z0-9_]+", "", text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub('[()!?]', ' ', text)
            text = re.sub('\[.*?\]', ' ', text)
            text = re.sub("[^a-z0-9]", " ", text)
            return text

    def process_data(self):
        texts = self.df['Text'].apply(self.clean_text).values
        sentiments = self.df['Sentiment'].values
        return texts, sentiments
