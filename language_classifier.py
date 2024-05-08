# ChatGPT was used to help create this file
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class LanguageClassifier:
    def __init__(self):
        # Import data
        self.train_data = pd.read_csv('data/train.csv')
        self.test_data = pd.read_csv('data/test.csv')
        # Clean data by changing abbreviations to full names
        self.abreviations_to_full_label_names = {
            "nl": "Dutch",
            "es": "Spanish",
            "it": "Italian",
            "ar": "Arabic",
            "ru": "Russian",
            "tr": "Turkish",
            "bg": "Bulgarian",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "fr": "French",
            "hi": "Hindi",
            "ja": "Japanese",
            "pl": "Polish",
            "pt": "Portuguese",
            "sw": "Swahili",
            "th": "Thai",
            "ur": "Urdu",
            "vi": "Vietnamese",
            "zh": "Chinese"
        }
        self.train_data["labels"] = self.train_data["labels"].map(self.abreviations_to_full_label_names)
        self.test_data["labels"] = self.test_data["labels"].map(self.abreviations_to_full_label_names)

        # Separate into test/train data
        self.X_train = self.train_data["text"]
        self.y_train = self.train_data["labels"]
        self.X_test = self.test_data["text"]
        self.y_test = self.test_data["labels"]

        # Vectorize data  - chatGPT was used to inform me about CountVectorizer
        self.vectorizer = CountVectorizer()
        self.X_train_counts = self.vectorizer.fit_transform(self.X_train)
        self.X_test_counts = self.vectorizer.transform(self.X_test)

        # Create model
        self.classifier = MultinomialNB()

        # Train model
        self.classifier = self.classifier.fit(self.X_train_counts, self.y_train)
