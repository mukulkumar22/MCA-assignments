from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def relevance_feedback(vec_docs, vec_queries, sim, n=10):

    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim # change

    alpha = 0.8
    beta  = 0.2

    total_queries = vec_queries.shape[0]

    iterations = int(input("Iterations to perform? ____"))

    for m in range(iterations):
        for j in range(total_queries):

            top_results = np.argsort(-rf_sim[:,j])[:n]
            worst_results = np.argsort(rf_sim[:,j])[:n]

            for i in range(len(top_results)):
                relevant = top_results[i]
                non_relevant = worst_results[i]
                vec_queries[j] = vec_queries[j] + alpha*vec_docs[relevant] - beta*vec_docs[non_relevant]

    
        rf_sim = cosine_similarity(vec_docs,vec_queries)

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim

    alpha = 0.8
    beta = 0.2

    total_queries = vec_queries.shape[0]
    

    index_to_words = {}

    for word,index in tfidf_model.vocabulary_.items():
        index_to_words[index] = word

    iterations = int(input("Iterations to perform? ____"))
    terms_to_update = int(input("How many terms to update? ____"))


    for i in range(iterations):
        
        for j in range(total_queries):

            new_terms = []
            top_results = np.argsort(-rf_sim[:,j])[:n]
            worst_results = np.argsort(rf_sim[:,j])[:n]
            for k in range(len(top_results)):
                relevant = top_results[k]
                non_relevant = worst_results[k]
                vec_queries[j] = vec_queries[j] + alpha*vec_docs[relevant] - beta*vec_docs[non_relevant]
            for doc in top_results:
                top_terms = np.argsort(-vec_docs[doc,:])[:terms_to_update]
                top_terms = [index_to_words[term] for term in top_terms]
                new_terms.extend(top_terms)
                
                
            vec_queries[j] = vec_queries[j] + tfidf_model.transform(new_terms)[0]

    
        rf_sim = cosine_similarity(vec_docs,vec_queries)

    return rf_sim
            