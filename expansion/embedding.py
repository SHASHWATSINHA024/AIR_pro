from gensim.models import Word2Vec

def train_embeddings(corpus_tokens):
    model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=2, workers=4)
    return model

def expand_query_embedding(query_tokens, model, topn=3):
    expanded = set(query_tokens)
    for word in query_tokens:
        if word in model.wv:
            expanded.update([w for w, _ in model.wv.most_similar(word, topn=topn)])
    return list(expanded)
