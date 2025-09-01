import numpy as np
import networkx as nx
import spacy # NEW: For information density scoring
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
spacy.load('en_core_web_sm')

# ==============================================================================
# PHASE 1 & 2: PRE-PROCESSING, EMBEDDING, CLUSTERING (No changes)
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
    MODIFIED: Now includes information density and structural position bonuses.
    """
    ## NEW ## - Load spaCy model for Named Entity Recognition
    nlp = spacy.load("en_core_web_sm")

    embeddings = np.array([s['embedding'] for s in sentences_data])
    n = len(sentences_data)
    
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['weight'] < similarity_threshold])
    pagerank_scores = nx.pagerank(graph)
    document_vector = np.mean(embeddings, axis=0)

    # Calculate Cluster Centrality
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
    
    for i, s_data in enumerate(sentences_data):
        # Base scores
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        centroid = centroids[cid]
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroid])[0][0]

        ## NEW ## - Priority #2: Information Content Scoring
        # Use the number of Named Entities (people, places, orgs) as a proxy for info density.
        doc = nlp(s_data['original'])
        s_data['info_density_score'] = len(doc.ents)

        ## NEW ## - Priority #3: Positional Bonuses for Structure
        # Give a bonus to the first and last sentences.
        position_bonus = 0
        if i == 0:
            position_bonus = 0.15 # Introduction bonus
        elif i == n - 1:
            position_bonus = 0.10 # Conclusion bonus
        s_data['position_score'] = position_bonus # We no longer use the linear decay

    # Normalize info density before weighting
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores) > 0:
        norm_info_scores = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data):
            s_data['info_density_score'] = norm_info_scores[i]

    ## MODIFIED ##
    # Re-balanced weights with our new, more intelligent features
    weights = {
        'pagerank_score': 0.35,
        'cluster_centrality_score': 0.25,
        'global_cohesion_score': 0.20,
        'info_density_score': 0.10, # Give a small but meaningful weight to info density
        'position_score': 0.10      # Structural bonuses
    }
    
    for s_data in sentences_data:
        score = sum(s_data[key] * w for key, w in weights.items())
        s_data['relevance_score'] = score
        
    return sentences_data

# ==============================================================================
# PHASE 5: TOPIC-GUIDED SUMMARY GENERATION WITH MMR (MODIFIED)
# ==============================================================================
## MODIFIED ##
def generate_summary_with_mmr(sentences_data, num_clusters, 
                              lambda_relevance=0.6, lambda_coherence=0.2,
                              dynamic_compression_std_dev=1.0):
    """
    MODIFIED: Now includes a robust sentence lookup to prevent IndexError,
    while still using dynamic compression and hybrid equitable allocation.
    """
    lambda_redundancy = 1 - lambda_relevance - lambda_coherence
    if lambda_redundancy < 0:
        raise ValueError("Lambda values must sum to 1 or less.")

    ## Create a robust lookup dictionary.
    # This maps the original sentence ID to its data dictionary, preventing crashes.
    sentence_lookup = {s['id']: s for s in sentences_data}

    # Dynamic Compression Rate
    all_scores = [s['relevance_score'] for s in sentences_data]
    if not all_scores:
        return "", 0, 0 # Handle empty input case
    
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    importance_threshold = mean_score + (dynamic_compression_std_dev * std_dev)
    
    total_sentences_needed = len([s for s in all_scores if s >= importance_threshold])
    total_sentences_needed = min(len(sentences_data), max(2, total_sentences_needed)) # Ensure we don't try to select more than available
    print(f"Dynamic compression rate determined. Target summary length: {total_sentences_needed} sentences.")

    # Group sentences by cluster ID
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        # Use the safe lookup to get the full data for the sentence
        clusters[s_data['cluster_id']].append(sentence_lookup[s_data['id']])

    # Hybrid Equitable Allocation
    num_to_pick_from_cluster = {}
    available_slots = total_sentences_needed
    
    # First pass: equitable base
    for cid in range(num_clusters):
        if clusters.get(cid) and available_slots > 0:
            num_to_pick_from_cluster[cid] = 1
            available_slots -= 1
    
    # Second pass: proportional remainder
    if available_slots > 0:
        cluster_importance = {cid: sum(s['relevance_score'] for s in sents) for cid, sents in clusters.items()}
        total_importance = sum(cluster_importance.values())
        
        # Sort clusters by importance to distribute remaining slots
        sorted_clusters = sorted(cluster_importance.items(), key=lambda item: item[1], reverse=True)
        
        idx = 0
        while available_slots > 0 and idx < len(sorted_clusters):
            cid, _ = sorted_clusters[idx % len(sorted_clusters)]
            # Ensure we don't try to pick more sentences than a cluster has
            if num_to_pick_from_cluster.get(cid, 0) < len(clusters.get(cid, [])):
                num_to_pick_from_cluster[cid] += 1
                available_slots -= 1
            idx += 1

    final_summary_indices = []
    
    # Outer loop: Iterate through clusters
    for cid in range(num_clusters):
        num_to_pick = num_to_pick_from_cluster.get(cid, 0)
        # We need a mutable list of candidates for this cluster
        candidate_sentences = list(clusters.get(cid, []))
        
        # Inner loop: Use MMR
        for _ in range(num_to_pick):
            if not candidate_sentences: break
            
            mmr_candidate_scores = []
            
            summary_embeddings = np.array([sentence_lookup[s_id]['embedding'] for s_id in final_summary_indices]) if final_summary_indices else np.array([])
            last_selected_embedding = sentence_lookup[final_summary_indices[-1]]['embedding'] if final_summary_indices else None

            for sentence in candidate_sentences:
                relevance = sentence['relevance_score']
                
                redundancy = max(cosine_similarity([sentence['embedding']], summary_embeddings)[0]) if summary_embeddings.size > 0 else 0
                coherence_bonus = cosine_similarity([sentence['embedding']], [last_selected_embedding])[0][0] if last_selected_embedding is not None else 0
                
                final_score = (lambda_relevance * relevance - lambda_redundancy * redundancy + lambda_coherence * coherence_bonus)
                mmr_candidate_scores.append((final_score, sentence))

            if not mmr_candidate_scores: break
            
            best_score, best_sentence = max(mmr_candidate_scores, key=lambda x: x[0])
            final_summary_indices.append(best_sentence['id'])
            # Remove the selected sentence from the candidate pool for this cluster
            candidate_sentences = [s for s in candidate_sentences if s['id'] != best_sentence['id']]
    
    final_summary_indices.sort()
    
    ## MODIFIED ## - Use the safe lookup dictionary for the final join
    summary = " ".join([sentence_lookup[i]['original'] for i in final_summary_indices])
    return summary, len(final_summary_indices), len(sentences_data)

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # --- Hyperparameters ---
    LAMBDA_RELEVANCE = 0.5
    LAMBDA_COHERENCE = 0.3
    DYNAMIC_COMPRESSION_STD_DEV = 0.7 # Lower value = more sentences, Higher value = more selective
    
    # --- Input Text (The "Time Travel" example, which had unbalanced topics) ---
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
    print(f"--- Starting Final Advanced Summarization ---")
    
    sentences_data = preprocess(TEXT)
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data)
    
    summary, num_sents_summary, num_sents_original = generate_summary_with_mmr(
        sentences_data, 
        num_clusters, 
        lambda_relevance=LAMBDA_RELEVANCE, 
        lambda_coherence=LAMBDA_COHERENCE,
        dynamic_compression_std_dev=DYNAMIC_COMPRESSION_STD_DEV
    )
    
    # --- Display Results ---
    print("\n=======================================")
    print("        FINAL GENERATED SUMMARY        ")
    print("=======================================")
    print(summary)
    print("\n=======================================")
    print("        SUMMARY STATS                  ")
    print(f"         {len(summary.split())} words")
    print(f"         {num_sents_summary} sentences selected from {num_sents_original} original sentences")
    print(f"         Actual Compression: {len(summary.split()) / len(TEXT.split()):.2%}")
    print("=======================================")