import json
import string 


def load_movies():
    with open("data/movies.json", "r") as file:
        movies = json.load(file)["movies"]
        return movies
    
def normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table)

def load_stop_words():
    stop_words = []
    with open("data/stopwords.txt", "r") as words_file:
        data = words_file.read().splitlines()
        stop_words = [normalize(word) for word in data]
    return stop_words

def tokenize_text(text):
    tokens = text.split()
    return tokens