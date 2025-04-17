from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSearchEngine:
    def __init__(self, docs):
        self.vectorizer = TfidfVectorizer()
        self.docs = docs
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
