import os

import nltk
import streamlit as st

from model import Model


def main():
    st.title('Sentiment Analysis')

    st.write("""
    # What sentiment in your text?
    """)

    text = st.sidebar.text_area(
        'Your text',
    )

    clf = Model.load_model()

    def get_text(string):
        return Model.clean_text(string)

    x_test = get_text(text)

    if x_test:
        y_pred = clf.predict([x_test])

        decoder = {
            0: 'Positive',
            1: 'Negative'
        }

        st.write(f'Sentiment = {decoder[y_pred[0]]}')


if __name__ == '__main__':
    if os.getenv('APP', '') != 'heroku':
        nltk.download('averaged_perceptron_tagger_ru')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
    main()
