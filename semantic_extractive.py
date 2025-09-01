import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# ==============================================================================
# PHASE 1: ADVANCED PRE-PROCESSING
# ==============================================================================
def preprocess(text):
    """
    Uses NLTK for robust sentence splitting, tokenization, stopword removal,
    and stemming. Returns a list of dictionaries, one for each sentence.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # 1. Use NLTK's robust sentence tokenizer
    original_sentences = sent_tokenize(text)
    
    processed_data = []
    for i, sentence in enumerate(original_sentences):
        # 2. Tokenize, lowercase, and filter for alphanumeric words
        words = word_tokenize(sentence.lower())
        # 3. Remove stopwords and stem
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        
        # We only process sentences with a meaningful number of tokens
        if len(tokens) > 3:
            processed_data.append({
                'id': i,
                'original': sentence,
                'tokens': tokens
            })
            
    return processed_data

# ==============================================================================
# PHASE 2: SEMANTIC REPRESENTATION & TOPIC MODELING
# ==============================================================================
def embed_sentences(sentences_data, model_name='intfloat/multilingual-e5-large-instruct'):
    """
    Generates a semantic embedding for each sentence using Sentence-BERT.
    """
    print(f"Loading Sentence-BERT model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = model.encode(original_sentences)
    
    for i, s_data in enumerate(sentences_data):
        s_data['embedding'] = embeddings[i]
        
    return sentences_data

def cluster_sentences(sentences_data):
    """
    Uses K-Means clustering on sentence embeddings to find topic clusters.
    """
    embeddings = np.array([s['embedding'] for s in sentences_data])
    
    # Heuristic to determine the number of clusters (topics)
    num_sentences = len(sentences_data)
    # At least 2 topics, but no more than 5, and roughly one topic per 4 sentences
    k = min(5, max(2, int(num_sentences / 4)))
    
    print(f"Clustering sentences into {k} topics...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = kmeans.labels_[i]
        
    return sentences_data, k

# ==============================================================================
# PHASE 3 & 4: SEMANTIC GRAPH & ADVANCED SENTENCE SCORING
# ==============================================================================
def score_sentences(sentences_data, similarity_threshold=0.3):
    """
    Scores each sentence based on a weighted combination of PageRank (from the
    semantic graph) and other improved features.
    """
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n = len(sentences_data)
    
    # 1. Build the semantic similarity matrix and graph
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    
    # Filter out weak edges to make PageRank more meaningful
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['weight'] < similarity_threshold])

    # 2. Calculate PageRank scores
    pagerank_scores = nx.pagerank(graph)

    # 3. Calculate other features
    for i, s_data in enumerate(sentences_data):
        # Feature: Sentence Position (heavily down-weighted)
        s_data['position_score'] = (n - i) / n
        
        # Feature: PageRank
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)

    # 4. Calculate Cluster Centrality feature
    clusters = {}
    for s_data in sentences_data:
        cid = s_data['cluster_id']
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(s_data['embedding'])
    
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
    
    for s_data in sentences_data:
        cid = s_data['cluster_id']
        centroid = centroids[cid]
        # Similarity to its own topic's center
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroid])[0][0]

    # 5. Calculate final weighted relevance score
    # These weights are hyperparameters that can be tuned.
    # We deliberately give 'position' a very low weight to fight lead bias.
    weights = {
        'pagerank_score': 0.5,
        'cluster_centrality_score': 0.4,
        'position_score': 0.1
    }
    
    for s_data in sentences_data:
        score = sum(s_data[key] * w for key, w in weights.items())
        s_data['relevance_score'] = score
        
    return sentences_data

# ==============================================================================
# PHASE 5: TOPIC-GUIDED SUMMARY GENERATION WITH MMR
# ==============================================================================
def generate_summary_with_mmr(sentences_data, num_clusters, compression_rate=0.2, mmr_lambda=0.6):
    """
    Generates a summary by picking representative sentences from each topic cluster
    using the MMR algorithm to ensure relevance and diversity.
    """
    total_sentences_needed = int(np.ceil(len(sentences_data) * compression_rate))
    
    # Group sentences by cluster
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data)

    # Determine how many sentences to pick from each cluster (proportional allocation)
    sentences_per_cluster = {cid: len(sents) for cid, sents in clusters.items()}
    total_in_clusters = sum(sentences_per_cluster.values())
    
    num_to_pick_from_cluster = {
        cid: max(1, int(round((count / total_in_clusters) * total_sentences_needed)))
        for cid, count in sentences_per_cluster.items()
    }

    # Ensure we get at least the required number of sentences
    while sum(num_to_pick_from_cluster.values()) < total_sentences_needed:
        # Add one to the largest cluster that hasn't been maxed out
        largest_cluster = max(sentences_per_cluster, key=sentences_per_cluster.get)
        if num_to_pick_from_cluster[largest_cluster] < sentences_per_cluster[largest_cluster]:
             num_to_pick_from_cluster[largest_cluster] += 1
        else: # this cluster is exhausted, so remove it from consideration for this loop
             sentences_per_cluster[largest_cluster] = -1


    final_summary_indices = []
    
    # Outer loop: Iterate through each topic cluster
    for cid, num_to_pick in num_to_pick_from_cluster.items():
        cluster_sentences = clusters[cid]
        cluster_embeddings = np.array([s['embedding'] for s in cluster_sentences])
        
        # Inner loop: Use MMR to select the best `num_to_pick` sentences from this cluster
        selected_in_cluster_indices = []
        
        for _ in range(num_to_pick):
            if not cluster_sentences: break

            candidate_scores = []
            summary_embeddings = np.array([sentences_data[i]['embedding'] for i in final_summary_indices])
            
            for sentence in cluster_sentences:
                relevance = sentence['relevance_score']
                
                # Calculate redundancy penalty
                if len(final_summary_indices) > 0:
                    similarities_to_summary = cosine_similarity([sentence['embedding']], summary_embeddings)[0]
                    redundancy = max(similarities_to_summary)
                else:
                    redundancy = 0
                
                mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy
                candidate_scores.append((mmr_score, sentence))

            # Select the best candidate and add it to our list
            best_score, best_sentence = max(candidate_scores, key=lambda x: x[0])
            final_summary_indices.append(best_sentence['id'])
            
            # Remove the selected sentence from the pool for this cluster
            cluster_sentences = [s for s in cluster_sentences if s['id'] != best_sentence['id']]
    
    # Sort the selected indices to maintain original document order
    final_summary_indices.sort()
    
    summary = " ".join([sentences_data[i]['original'] for i in final_summary_indices])
    return summary


# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- Hyperparameters ---
    COMPRESSION_RATE = 0.3  # Target summary length as a fraction of the original
    MMR_LAMBDA = 0.6        # Balance between relevance and diversity (0.5 is a good start)
    
    # --- Input Text (from user) ---
    TITLE = input("Enter the title of the text: ")
    print("Enter the text to summarize (press Ctrl+Z and Enter on Windows or Ctrl+D on Linux/macOS when done):")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    TEXT = "\n".join(lines)

    # --- Execute the Full Pipeline ---
    print("\n--- Starting Advanced Summarization Pipeline ---")
    
    # 1. Pre-process
    sentences_data = preprocess(TEXT)
    
    # 2. Embed & Cluster
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    
    # 3 & 4. Score Sentences
    sentences_data = score_sentences(sentences_data)
    
    # 5. Generate Final Summary
    print("\n--- Generating Final Summary ---")
    summary = generate_summary_with_mmr(sentences_data, num_clusters, COMPRESSION_RATE, MMR_LAMBDA)
    
    # --- Display Results ---
    print("\n=======================================")
    print("        ORIGINAL TEXT LENGTH           ")
    print(f"         {len(TEXT.split())} words")
    print("=======================================")
    
    print("\n=======================================")
    print("        FINAL GENERATED SUMMARY        ")
    print("=======================================")
    print(summary)
    print("\n=======================================")
    print("        SUMMARY STATS                  ")
    print(f"         {len(summary.split())} words")
    print(f"         Actual Compression: {len(summary.split()) / len(TEXT.split()):.2%}")
    print("=======================================")