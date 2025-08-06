from sklearn.feature_extraction.text import TfidfVectorizer
from match import cosine

def search_by_keyword_match(query, db):
    best_score = 0
    best_record = None

    query_keywords = set(query.lower().split())

    for record in db:
        record_keywords = set(record.lower().split())

        common_keywords = query_keywords.intersection(record_keywords)
        current_score = len(common_keywords)

        if current_score > best_score:
            best_score = current_score
            best_record = record

    return best_score, best_record

def search_by_cosine_match(query, db):
    best_score = 0
    best_record = None
    for record in db:
        current_score = cosine(query, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

def vectorize(records) : 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix
