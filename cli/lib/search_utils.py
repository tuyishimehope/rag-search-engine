import json
import string 

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

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

def tokenize_text(text: str) -> list[str]:
    tokens = normalize(text).split() 
    stop_words = load_stop_words()
    return [stemmer.stem(t) for t in tokens if t not in stop_words]