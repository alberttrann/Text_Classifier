# This new version implements Priority #1 (Coherence Modeling) and Priority #2 (Improved Relevance Scoring).
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==============================================================================
# PHASE 1: ADVANCED PRE-PROCESSING (No changes from previous version)
# ==============================================================================
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    original_sentences = sent_tokenize(text)
    processed_data = []
    for i, sentence in enumerate(original_sentences):
        words = word_tokenize(sentence.lower())
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        if len(tokens) > 3:
            processed_data.append({'id': i, 'original': sentence, 'tokens': tokens})
    return processed_data

# ==============================================================================
# PHASE 2: SEMANTIC REPRESENTATION & TOPIC MODELING (No changes from previous version)
# ==============================================================================
def embed_sentences(sentences_data, model_name='intfloat/multilingual-e5-large-instruct'):
    print(f"Loading Sentence-BERT model '{model_name}'...")
    model = SentenceTransformer(model_name)
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = model.encode(original_sentences)
    for i, s_data in enumerate(sentences_data):
        s_data['embedding'] = embeddings[i]
    return sentences_data

def cluster_sentences(sentences_data):
    embeddings = np.array([s['embedding'] for s in sentences_data])
    num_sentences = len(sentences_data)
    k = min(5, max(2, int(num_sentences / 4)))
    print(f"Clustering sentences into {k} topics...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = kmeans.labels_[i]
    return sentences_data, k

# ==============================================================================
# PHASE 3 & 4: SEMANTIC GRAPH & ADVANCED SENTENCE SCORING (MODIFIED)
# ==============================================================================
## MODIFIED ##
def score_sentences(sentences_data, similarity_threshold=0.3):
    """
    MODIFIED: Scores each sentence with an added 'global_cohesion_score' to
    ensure sentences are relevant to the overall document topic.
    """
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n = len(sentences_data)
    
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['weight'] < similarity_threshold])

    pagerank_scores = nx.pagerank(graph)

    ## NEW ## - Priority #2: Improve Intra-Cluster Relevance Scoring
    # 1. Create a global document vector by averaging all sentence embeddings.
    document_vector = np.mean(embeddings, axis=0)

    for i, s_data in enumerate(sentences_data):
        s_data['position_score'] = (n - i) / n
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)

        ## NEW ##
        # 2. Add the new 'global_cohesion_score' feature for each sentence.
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]

    # Calculate Cluster Centrality feature
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data['embedding'])
    
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
    
    for s_data in sentences_data:
        cid = s_data['cluster_id']
        centroid = centroids[cid]
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroid])[0][0]

    ## MODIFIED ##
    # 3. Update the weights to include the new feature and re-balance.
    weights = {
        'pagerank_score': 0.4,
        'cluster_centrality_score': 0.3,
        'global_cohesion_score': 0.2, # Give the new feature a solid weight
        'position_score': 0.1         # Keep position weight very low
    }
    
    for s_data in sentences_data:
        score = sum(s_data[key] * w for key, w in weights.items())
        s_data['relevance_score'] = score
        
    return sentences_data

# ==============================================================================
# PHASE 5: TOPIC-GUIDED SUMMARY GENERATION WITH MMR (MODIFIED)
# ==============================================================================
## MODIFIED ##
def generate_summary_with_mmr(sentences_data, num_clusters, compression_rate=0.2, 
                              lambda_relevance=0.6, lambda_coherence=0.2):
    """
    MODIFIED: The MMR calculation now includes a 'coherence_bonus' to promote
    narrative flow between consecutively chosen sentences.
    """
    lambda_redundancy = 1 - lambda_relevance - lambda_coherence
    if lambda_redundancy < 0:
        raise ValueError("Lambda values must sum to 1 or less.")

    total_sentences_needed = int(np.ceil(len(sentences_data) * compression_rate))
    
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data)

    num_to_pick_from_cluster = {cid: max(1, int(round((len(sents) / len(sentences_data)) * total_sentences_needed))) for cid, sents in clusters.items()}
    while sum(num_to_pick_from_cluster.values()) < total_sentences_needed:
        largest_cluster = max(num_to_pick_from_cluster, key=lambda cid: len(clusters[cid]))
        if num_to_pick_from_cluster[largest_cluster] < len(clusters[largest_cluster]):
             num_to_pick_from_cluster[largest_cluster] += 1
        else:
             # This cluster is exhausted, find the next largest one
             temp_clusters = {cid: len(sents) for cid, sents in clusters.items() if num_to_pick_from_cluster[cid] < len(sents)}
             if not temp_clusters: break
             largest_cluster = max(temp_clusters, key=temp_clusters.get)
             num_to_pick_from_cluster[largest_cluster] += 1
             
    final_summary_indices = []
    
    # Outer loop: Iterate through clusters to ensure coverage
    for cid in range(num_clusters):
        num_to_pick = num_to_pick_from_cluster.get(cid, 0)
        cluster_sentences = clusters.get(cid, [])
        
        # Inner loop: Use modified MMR to select sentences from this cluster
        for _ in range(num_to_pick):
            if not cluster_sentences: break

            candidate_scores = []
            
            # Get embeddings of sentences already in our main summary
            summary_embeddings = np.array([s['embedding'] for s_id, s in enumerate(sentences_data) if s['id'] in final_summary_indices])
            
            ## NEW ## - Priority #1: Introduce Coherence Modeling
            # Get the embedding of the absolute last sentence selected
            last_selected_embedding = None
            if final_summary_indices:
                last_idx = final_summary_indices[-1]
                last_selected_embedding = sentences_data[last_idx]['embedding']

            for sentence in cluster_sentences:
                relevance = sentence['relevance_score']
                
                # Calculate redundancy against the entire current summary
                redundancy = 0
                if len(summary_embeddings) > 0:
                    similarities_to_summary = cosine_similarity([sentence['embedding']], summary_embeddings)[0]
                    redundancy = max(similarities_to_summary)
                
                ## NEW ##
                # Calculate coherence bonus with the last selected sentence
                coherence_bonus = 0
                if last_selected_embedding is not None:
                    coherence_bonus = cosine_similarity([sentence['embedding']], [last_selected_embedding])[0][0]
                
                ## MODIFIED ##
                # New final score formula incorporating all three aspects
                final_score = (
                    lambda_relevance * relevance - 
                    lambda_redundancy * redundancy + 
                    lambda_coherence * coherence_bonus
                )
                candidate_scores.append((final_score, sentence))

            if not candidate_scores: break
            
            best_score, best_sentence = max(candidate_scores, key=lambda x: x[0])
            final_summary_indices.append(best_sentence['id'])
            cluster_sentences = [s for s in cluster_sentences if s['id'] != best_sentence['id']]
    
    final_summary_indices.sort()
    summary = " ".join([sentences_data[i]['original'] for i in final_summary_indices])
    return summary

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- Hyperparameters ---
    COMPRESSION_RATE = 0.3
    
    ## NEW ## - Lambdas for the new coherence-aware MMR formula
    # These must sum to 1.0. Tune these to balance the priorities.
    # Higher relevance = more important sentences.
    # Higher coherence = better narrative flow.
    # (Redundancy is calculated implicitly).
    LAMBDA_RELEVANCE = 0.6  # Focus primarily on relevance
    LAMBDA_COHERENCE = 0.2  # Add a moderate bonus for good flow
    
    # --- Input Text (The "Electroreception" example, which had coherence issues) ---
    TITLE = input("Enter the title for the text: ")
    print("Enter or paste the text to summarize. Press CTRL+Z and then Enter to finish.")
    TEXT = ""
    while True:
        try:
            line = input()
            TEXT += line + "\n"
        except EOFError:
            break

    # --- Execute the Full Pipeline ---
    print(f"--- Starting Coherence-Aware Summarization (Relevance={LAMBDA_RELEVANCE}, Coherence={LAMBDA_COHERENCE}) ---")
    
    sentences_data = preprocess(TEXT)
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data)
    
    summary = generate_summary_with_mmr(
        sentences_data, 
        num_clusters, 
        COMPRESSION_RATE, 
        lambda_relevance=LAMBDA_RELEVANCE, 
        lambda_coherence=LAMBDA_COHERENCE
    )
    
    # --- Display Results ---
    print("\n=======================================")
    print("        FINAL GENERATED SUMMARY        ")
    print("=======================================")
    print(summary)
    print("\n=======================================")
    print("        SUMMARY STATS                  ")
    print(f"         {len(summary.split())} words")
    print(f"         Actual Compression: {len(summary.split()) / len(TEXT.split()):.2%}")
    print("=======================================")