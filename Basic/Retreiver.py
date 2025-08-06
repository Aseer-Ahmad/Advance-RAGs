from sklearn.feature_extraction.text import TfidfVectorizer
from match import cosine

class Retreiver : 
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()

    def vectorize(self, df) : 
        self.tfidf_matrix =  self.vectorizer.fit_transform(df)

    def match_and_search(self, text) : 
        text_vector = self.vectorizer.transform([text])
        cosine_similarities = cosine(text_vector, self.tfidf_matrix)
        return cosine_similarities
    
    def retrieve(self, text):
        self.match_and_search(text)

        