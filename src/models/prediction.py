import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from .train_model import SentimentAnalysisModel
from .data_prep import DataProcessor
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentimentAnalyzer:
    def __init__(self, 
                    model_file='src/models/sentiment_analysis_model.h5', 
                    tokenizer_file="src/models/tokenizer.pickle"):

        self.model_file = model_file
        self.tokenizer_file = tokenizer_file
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        if os.path.exists(self.tokenizer_file):
            with open(self.tokenizer_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Saved tokenizer not found. Creating new tokenizer...")
            model = SentimentAnalysisModel()
            model.train_model(DataProcessor().texts, DataProcessor().sentiments)
            model.save_tokenizer()
            model.save_model()
            return model.tokenizer

    def load_model(self):
        if os.path.exists(self.model_file):
            return load_model(self.model_file)
        else:
            print("Saved model not found. Training new model...")
            model = SentimentAnalysisModel()
            model.train_model(DataProcessor().texts, DataProcessor().sentiments)
            model.save_tokenizer()
            model.save_model()
            return model.model

    def predict_sentiment(self, text):
        sequence = self.tokenizer.texts_to_sequences(text)
        padded_sequence = pad_sequences(sequence, maxlen=500, value=9999, padding='post')
        predictions = self.model.predict(padded_sequence)
        sentiment_classes = ['Neutral', 'Positive', 'Negative']
        predicted_sentiments = [sentiment_classes[np.argmax(prediction)] for prediction in predictions]
        #predicted_sentiment = sentiment_classes[np.argmax(prediction)]
        return predicted_sentiments


