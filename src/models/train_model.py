import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class SentimentAnalysisModel:
    def __init__(self, max_sequence_length=500, vocab_size=10000, embedding_dim=64):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        self.tokenizer = Tokenizer(num_words=vocab_size)

    def _build_model(self):
        model = Sequential([
                        Embedding(self.vocab_size, 
                                    self.embedding_dim, 
                                    input_length=self.max_sequence_length),
                        Dense(32, activation='relu'),
                        Flatten(),
                        Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
                    ])
        model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['categorical_accuracy'])
        return model

    def train_model(self, texts, sentiments):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, 
                                            maxlen=self.max_sequence_length, 
                                            value=self.vocab_size-1, 
                                            padding='post')

        encoded_sentiments = to_categorical(sentiments + 1, num_classes=3)

        X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_sentiments, test_size=0.15, random_state=42)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.15)

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    def save_model(self, file_path='src/models/sentiment_analysis_model.h5'):
        self.model.save(file_path)
        print(f'Model saved successfully as {file_path}')

    def save_tokenizer(self, tokenizer_file="src/models/tokenizer.pickle"):
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Tokenizer saved successfully as {tokenizer_file}')

