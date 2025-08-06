from sklearn.feature_extraction.text import TfidfVectorizer

class Retreiver : 
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()

    def vectorize(self, df) : 
        self.tfidf_matrix =  self.vectorizer.fit_transform(df)

    def match_and_search(self) : 
        pass

    def query(self, text):
        pass

        