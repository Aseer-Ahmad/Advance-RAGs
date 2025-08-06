from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
import nltk

from collections import Counter
import numpy as np


def cosine(t1, t2) : 
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,   # Apply sublinear TF scaling
        analyzer='word'      # You could also experiment with 'char' or 'char_wb' for character-level features
    )

    tfidf = vectorizer.fit_transform([t1, t2])
    sim   = cosine_similarity(tfidf[0:1], tfidf[0:2])
    return sim[0][0]


def load_spacy_model():
    return spacy.load("en_core_web_sm")

def preprocess(text) : 
    nlp = load_spacy_model()
    lemmatized_words = []
    doc = nlp(text.lower())
    for token in doc :
        if token.is_stop or token.is_punct :
            continue
        lemmatized_words.append(token.lemma_)
    return lemmatized_words

def calculate_enhanced_similarity(text1, text2):

    words1 = preprocess(text1)
    words2 = preprocess(text2)

    freq1 = Counter(words1)
    freq2 = Counter(words2)

    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cosine_similarity