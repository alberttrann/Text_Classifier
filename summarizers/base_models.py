"""
summarizers/base_models.py

This script contains the implementation for the baseline and classic
heuristic-based summarization models.

Models:
1. TextRank_Baseline
2. Original_Paper_Method
"""

import numpy as np
import networkx as nx
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Pre-processing (Shared by both models) ---
def _preprocess_text(text):
    """
    A robust pre-processing function using NLTK.
    Returns both original sentences and processed (tokenized, stemmed) sentences.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    try:
        original_sentences = sent_tokenize(text)
    except Exception:
        # Fallback for NLTK download issues
        original_sentences = text.split('. ')
    
    sentences_data = []
    for i, sentence in enumerate(original_sentences):
        # Basic filtering for sentences that are too short to be meaningful
        if len(sentence.split()) < 4:
            continue
        
        words = word_tokenize(sentence.lower())
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        
        if tokens: # Only add if there are tokens left after processing
            sentences_data.append({
                'id': i,
                'original': sentence,
                'tokens': tokens,
                'token_str': ' '.join(tokens)
            })
            
    return sentences_data

# ==============================================================================
# Model 1: TextRank Baseline
# ==============================================================================

def textrank_summarize(text, num_sentences=4):
    """
    Generates a summary using the classic TextRank algorithm.
    
    Returns:
        - summary (str): The final summary text.
        - details (dict): A dictionary containing intermediate data for visualization,
                          including sentences, similarity matrix, and scores.
    """
    # 1. Pre-process the text
    sentences_data = _preprocess_text(text)
    if not sentences_data:
        return "Input text is too short or could not be processed.", {}

    original_sentences = [s['original'] for s in sentences_data]
    
    # Handle cases where the text is shorter than the desired summary
    num_to_select = min(num_sentences, len(original_sentences))
    if len(original_sentences) <= num_to_select:
        return " ".join(original_sentences), {}

    # 2. Vectorize sentences using TF-IDF
    sentence_strings = [s['token_str'] for s in sentences_data]
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentence_strings)
    except ValueError:
        return "Could not create TF-IDF matrix (likely empty vocabulary after processing).", {}
    
    # 3. Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # 5. Use PageRank to score the sentences
    try:
        scores = nx.pagerank(nx_graph)
    except Exception:
        # Fallback for disconnected graphs
        scores = {i: 1.0 / len(original_sentences) for i in range(len(original_sentences))}

    # 6. Rank sentences and select the top N
    ranked_indices_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, score in ranked_indices_scores[:num_to_select]]
    
    # 7. Sort selected sentences back to their original order for coherence
    top_indices.sort()
    
    summary = " ".join([original_sentences[i] for i in top_indices])
    
    # 8. Package details for visualization
    details = {
        "processed_sentences": sentences_data,
        "similarity_matrix": similarity_matrix,
        "graph": nx_graph,
        "scores": scores,
        "ranked_sentences": ranked_indices_scores,
        "selected_indices": top_indices
    }
    
    return summary, details

# ==============================================================================
# Model 2: Original Paper's Method
# ==============================================================================

def original_paper_summarize(text, title, num_sentences=4, sim_threshold=0.3):
    """
    An implementation of the original research paper's method based on
    Triangle-Graph filtering and heuristic feature scoring.

    Returns:
        - summary (str): The final summary text.
        - details (dict): A dictionary containing intermediate data for visualization.
    """
    # 1. Pre-process
    sentences_data = _preprocess_text(text)
    if not sentences_data:
        return "Input text is too short or could not be processed.", {}
        
    original_sentences = [s['original'] for s in sentences_data]
    n = len(sentences_data)
    num_to_select = min(num_sentences, n)
    if n <= num_to_select:
        return " ".join(original_sentences), {}

    # 2. TF-IDF and Similarity Matrix
    sentence_strings = [s['token_str'] for s in sentences_data]
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(sentence_strings)
    except ValueError:
        return "Could not create TF-IDF matrix.", {}
    sim_matrix = cosine_similarity(tfidf_matrix)

    # 3. Graph and Triangle-based Filtering
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > sim_threshold:
                graph.add_edge(i, j, weight=sim_matrix[i, j])

    cliques = list(nx.find_cliques(graph))
    triangles = [clique for clique in cliques if len(clique) == 3]
    
    bitvector = np.zeros(n)
    is_fallback = False
    
    if triangles:
        nodes_in_triangles = set(node for tri in triangles for node in tri)
        for node_index in nodes_in_triangles:
            bitvector[node_index] = 1
    else:
        # Fallback if no triangles are found: use degree centrality
        is_fallback = True
        degrees = dict(graph.degree())
        if degrees:
            avg_degree = sum(degrees.values()) / n if n > 0 else 0
            for i in range(n):
                if degrees.get(i, 0) >= avg_degree:
                    bitvector[i] = 1
    
    # 4. Feature Extraction and Scoring
    stemmer = PorterStemmer()
    title_words = set([stemmer.stem(w.lower()) for w in word_tokenize(title) if w.isalnum()])
    
    scores = np.zeros(n)
    feature_details = []

    for i, s_data in enumerate(sentences_data):
        tf_score = len(set(s_data['tokens']) & title_words)
        sl_score = len(s_data['tokens'])
        sp_score = (n - (i+1)) / n # Corrected to be 0-indexed like
        nd_score = len(re.findall(r'\d+', s_data['original']))
        sts_score = np.sum(sim_matrix[i, :])
        
        feature_sum = tf_score + sl_score + sp_score + nd_score + sts_score
        
        # In fallback mode, we score all sentences. Otherwise, only those in triangles.
        if is_fallback or bitvector[i] == 1:
            scores[i] = feature_sum
        
        feature_details.append({
            "Sentence": s_data['original'][:50] + "...",
            "Title Words": tf_score, "Length": sl_score, "Position": f"{sp_score:.2f}",
            "Numbers": nd_score, "Similarity Sum": f"{sts_score:.2f}",
            "In Triangle": "Yes" if bitvector[i] == 1 else "No",
            "Final Score": f"{scores[i]:.2f}"
        })

    # 5. Summary Generation
    candidate_indices_scores = {i: scores[i] for i in range(n) if bitvector[i] == 1}
    
    # If not enough candidates in triangles, expand the pool to all sentences
    if len(candidate_indices_scores) < num_to_select:
        is_fallback = True # Log that we used a fallback
        candidate_indices_scores = {i: scores[i] for i in range(n)}
        
    ranked_indices = sorted(candidate_indices_scores, key=candidate_indices_scores.get, reverse=True)
    top_indices = sorted(ranked_indices[:num_to_select])
    
    summary = " ".join([original_sentences[i] for i in top_indices])
    
    # 6. Package details for visualization
    details = {
        "processed_sentences": sentences_data,
        "similarity_matrix": sim_matrix,
        "graph": graph,
        "triangles": triangles,
        "feature_scores": feature_details,
        "selected_indices": top_indices,
        "fallback_activated": is_fallback
    }
    
    return summary, details