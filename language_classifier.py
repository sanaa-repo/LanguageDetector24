# Import statements
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load data from a CSV file
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

X_train = train_data["text"]
y_train = train_data["labels"]

X_test = test_data["text"]
y_test = test_data["labels"]

vectorizer = CountVectorizer() #Found out about count vectorizer through ChatGPT
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

classifier = MultinomialNB()

classifier.fit(X_train_counts, y_train)

y_pred = classifier.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)