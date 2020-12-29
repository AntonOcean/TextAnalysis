import re

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class Model:
    filename = 'sentiment_model.pkl'

    def __init__(self):
        self.model = None

    def create_model(self):
        comments = pd.read_csv('labeled.csv')
        comments.head(10)

        text = np.array(comments.comment.values)
        target = comments.toxic.astype(int).values

        text = list(map(self.clean_text, text))

        model = Pipeline(
            [("vectorizer", TfidfVectorizer(tokenizer=self.clean_text, max_features=10000, ngram_range=(1, 2))),
             ("classifier", LinearSVC())]
        )

        model.fit(text, target)

        self.model = model

    @staticmethod
    def clean_text(string):
        """This function deletes all symbols except Cyrilic and Base Latin alphabet,
        stopwords, functional parts of speech. Returns string of words stem."""
        # Common cleaning
        string = string.lower()
        string = re.sub(r"http\S+", "", string)
        string = str.replace(string, 'Ё', 'е')
        string = str.replace(string, 'ё', 'е')
        prog = re.compile('[А-Яа-яA-Za-z]+')
        words = prog.findall(string.lower())

        stopwords = nltk.corpus.stopwords.words('russian')
        words = [w for w in words if w not in stopwords]

        functional_pos = {'CONJ', 'PRCL'}
        words = [w for w, pos in nltk.pos_tag(words, lang='rus') if pos not in functional_pos]

        stemmer = SnowballStemmer('russian')
        return ' '.join(list(map(stemmer.stem, words)))

    def save_model(self):
        joblib.dump(self.model, self.filename)

    @classmethod
    def load_model(cls):
        return joblib.load(cls.filename)


def main():
    model = Model()
    model.create_model()
    model.save_model()


if __name__ == '__main__':
    nltk.download('averaged_perceptron_tagger_ru')
    main()
