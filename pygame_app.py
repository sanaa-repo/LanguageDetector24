# ChatGPT was used to help create this file
import pygame
from pygame_gui import UIManager
from pygame.locals import *
from language_classifier import LanguageClassifier

class PygameApp:
    def __init__(self, language_classifier):
        self.language_classifier = language_classifier
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.manager = UIManager((self.WIDTH, self.HEIGHT), starting_language="ja")
        pygame.display.set_caption("Language Classifier")

        # Initialize color variables
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BLUE = (83, 153, 204)
        self.GREEN = (171, 235, 198)
        self.font = pygame.font.SysFont('calibri', 32)
        self.input_text = ""
        self.result_text = ""

    # Uses text input to predict language
    def predict_language(self, sample_text):
        input_text_counts = self.language_classifier.vectorizer.transform([sample_text])
        predicted_language = self.language_classifier.classifier.predict(input_text_counts)
        return predicted_language[0]

    # Displays text on screen
    def draw_text(self, text, color, x, y):
        textobj = self.font.render(text, True, color)
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        self.screen.blit(textobj, textrect)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        self.result_text = self.predict_language(self.input_text)
                        self.input_text = ""
                    elif event.key == K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        self.input_text += event.unicode
            # Draw graphics
            self.screen.fill(self.WHITE)
            pygame.draw.rect(self.screen, self.BLUE, (0, 0, self.WIDTH, 100))
            self.draw_text("Language Classifier", self.WHITE, 50, 25)
            pygame.draw.rect(self.screen, self.GREEN, (50, 200, 700, 50))
            pygame.draw.rect(self.screen, self.BLACK, (50, 200, 700, 50), 2)
            self.draw_text(self.input_text, self.BLACK, 60, 210)
            self.draw_text("Predicted language:", self.BLACK, 50, 300)
            self.draw_text(self.result_text, self.BLUE, 50, 350)
            self.draw_text("Enter a phrase or sentence:", self.BLACK, 50, 150)
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    classifier = LanguageClassifier()
    app = PygameApp(classifier)
    app.run()
