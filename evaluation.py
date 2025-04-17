def precision_at_k(results, relevant, k):
    retrieved = [doc for doc, _ in results[:k]]
    true_positives = len(set(retrieved) & set(relevant))
    return true_positives / k if k else 0

def mean_average_precision(results, relevant):
    hits, score = 0, 0.0
    for i, (doc, _) in enumerate(results):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0
