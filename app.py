from sklearn.datasets import fetch_20newsgroups
from utils.preprocess import preprocess
from search_engine.tfidf_search import TFIDFSearchEngine
from expansion.wordnet import expand_query_wordnet
import nltk
nltk.data.path.append('/Users/shashwatsinha/nltk_data')


# Load dataset
print("Loading data...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = data.data
target_names = data.target_names

# Preprocess
print("Preprocessing documents...")
preprocessed_docs = [' '.join(preprocess(doc)) for doc in documents]

# Search engine
search_engine = TFIDFSearchEngine(preprocessed_docs)

# User query
query = input("Enter your query: ")
tokens = preprocess(query)
expanded_tokens = expand_query_wordnet(tokens)
expanded_query = ' '.join(expanded_tokens)

# Search
results = search_engine.search(expanded_query, top_k=5)

# Display
print("\nTop 5 Search Results:\n")
for idx, score in results:
    print(f"Score: {score:.4f}\nDocument:\n{documents[idx][:300]}...\n{'-'*50}")
