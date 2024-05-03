import pygame
from pygame.locals import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load data from a CSV file
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

abreviations_to_full_label_names = {
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

# Replace abbreviated labels with full names
test_data["labels"] = test_data["labels"].map(abreviations_to_full_label_names)
train_data["labels"] = train_data["labels"].map(abreviations_to_full_label_names)

X_train = train_data["text"]
y_train = train_data["labels"]

X_test = test_data["text"]
y_test = test_data["labels"]

# Vectorize text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier = classifier.fit(X_train_counts, y_train)

# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Language Classifier")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (83, 153, 204)  # Light blue
GREEN = (171, 235, 198)  # Light green

# Define font
font = pygame.font.SysFont('calibri', 32)

# Function to predict language of input string
def predict_language(sample_text):
    input_text_counts = vectorizer.transform([sample_text])
    predicted_language = classifier.predict(input_text_counts)
    return predicted_language[0]

# Function to display text on the screen
def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

# Main loop
running = True
input_text = ""
result_text = ""
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_RETURN:
                # Predict language when Enter key is pressed
                result_text = predict_language(input_text)
                input_text = ""
            elif event.key == K_BACKSPACE:
                input_text = input_text[:-1]
            else:
                input_text += event.unicode
    
    # Draw background
    screen.fill(WHITE)

    # Draw title background
    pygame.draw.rect(screen, BLUE, (0, 0, WIDTH, 100))

    # Draw title
    draw_text("Language Classifier", font, WHITE, screen, 50, 25)

    # Draw input box
    pygame.draw.rect(screen, GREEN, (50, 200, 700, 50))
    pygame.draw.rect(screen, BLACK, (50, 200, 700, 50), 2)
    draw_text(input_text, font, BLACK, screen, 60, 210)

    # Draw result
    draw_text("Predicted language:", font, BLACK, screen, 50, 300)
    draw_text(result_text, font, BLUE, screen, 50, 350)
    
    # Draw prompt
    draw_text("Enter a phrase or sentence:", font, BLACK, screen, 50, 150)

    pygame.display.flip()

# Quit Pygame
pygame.quit()
