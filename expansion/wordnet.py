from nltk.corpus import wordnet

def expand_query_wordnet(tokens):
    expanded = set(tokens)
    for word in tokens:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)
