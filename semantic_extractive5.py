import numpy as np
import networkx as nx
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import math

# ==============================================================================
# PHASES 1-4: PRE-PROCESSING, EMBEDDING, CLUSTERING, SCORING (No changes from previous version)
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    original_sentences = sent_tokenize(text)
    processed_data = []
    unfiltered_to_filtered_map = {}
    filtered_idx = 0
    for i, sentence in enumerate(original_sentences):
        words = word_tokenize(sentence.lower())
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        if len(tokens) > 3:
            processed_data.append({'id': i, 'original': sentence, 'tokens': tokens})
            unfiltered_to_filtered_map[i] = filtered_idx
            filtered_idx += 1
    return processed_data, unfiltered_to_filtered_map

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

def score_sentences(sentences_data, unfiltered_map, similarity_threshold=0.3):
    print("Loading spaCy model for NER...")
    nlp = spacy.load("en_core_web_sm")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_filtered = len(sentences_data)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else n_filtered
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['weight'] < similarity_threshold])
    pagerank_scores = nx.pagerank(graph)
    document_vector = np.mean(embeddings, axis=0)
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
    for i, s_data in enumerate(sentences_data):
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        centroid = centroids[cid]
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroid])[0][0]
        doc = nlp(s_data['original'])
        s_data['info_density_score'] = len(doc.ents)
        structural_bonus = 0
        original_id = s_data['id']
        if original_id == 0: structural_bonus = 0.15
        elif original_id == n_original - 1: structural_bonus = 0.10
        s_data['structural_bonus'] = structural_bonus
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores) > 0:
        norm_info_scores = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data):
            s_data['info_density_score'] = norm_info_scores[i]
    weights = {'pagerank_score': 0.30, 'cluster_centrality_score': 0.25, 'global_cohesion_score': 0.15, 'info_density_score': 0.15, 'structural_bonus': 0.15}
    for s_data in sentences_data:
        score = sum(s_data.get(key, 0) * w for key, w in weights.items())
        s_data['relevance_score'] = score
    return sentences_data
# ==============================================================================

## MODIFIED ## - Now returns a list of sentence IDs
def generate_summary_with_mmr(sentences_data, num_clusters, 
                              lambda_relevance, lambda_coherence,
                              dynamic_compression_std_dev):
    sentence_lookup = {s['id']: s for s in sentences_data}
    all_scores = [s['relevance_score'] for s in sentences_data]
    if not all_scores: return [], 0, 0
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    importance_threshold = mean_score + (dynamic_compression_std_dev * std_dev)
    total_sentences_needed = len([s for s in all_scores if s >= importance_threshold])
    total_sentences_needed = min(len(sentences_data), max(2, total_sentences_needed))
    
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        clusters[s_data['cluster_id']].append(s_data)

    num_to_pick_from_cluster = {}
    available_slots = total_sentences_needed
    for cid in range(num_clusters):
        if clusters.get(cid) and available_slots > 0:
            num_to_pick_from_cluster[cid] = 1
            available_slots -= 1
    if available_slots > 0:
        cluster_importance = {cid: max(s['relevance_score'] for s in sents) if sents else 0 for cid, sents in clusters.items()}
        sorted_clusters = sorted(cluster_importance.items(), key=lambda item: item[1], reverse=True)
        idx = 0
        while available_slots > 0 and idx < len(sorted_clusters):
            cid, _ = sorted_clusters[idx % len(sorted_clusters)]
            if num_to_pick_from_cluster.get(cid, 0) < len(clusters.get(cid, [])):
                num_to_pick_from_cluster[cid] += 1
                available_slots -= 1
            idx += 1

    final_summary_indices = []
    lambda_redundancy = 1 - lambda_relevance - lambda_coherence
    for cid in sorted(num_to_pick_from_cluster.keys()):
        num_to_pick = num_to_pick_from_cluster.get(cid, 0)
        candidate_sentences = list(clusters.get(cid, []))
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
            candidate_sentences = [s for s in candidate_sentences if s['id'] != best_sentence['id']]
            
    final_summary_indices.sort()
    ## NEW RETURN ##
    return final_summary_indices

# ==============================================================================
# NEW: STAGE 1 - CANDIDATE GENERATION
# ==============================================================================
def generate_candidate_summaries(sentences_data, num_clusters):
    """
    Generates multiple candidate summaries by running the main algorithm with
    different "personalities" (hyperparameter variations).
    """
    print("\n--- Generating Candidate Summaries with different 'personalities' ---")
    
    personalities = {
        "balanced":        {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 0.7},
        "prefers_relevance": {'lambda_relevance': 0.7, 'lambda_coherence': 0.1, 'dynamic_compression_std_dev': 0.7},
        "prefers_coherence": {'lambda_relevance': 0.4, 'lambda_coherence': 0.5, 'dynamic_compression_std_dev': 0.7},
        "more_selective":  {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 1.0},
        "more_verbose":    {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 0.4},
    }
    
    candidate_indices_set = set()
    for name, params in personalities.items():
        print(f"Generating candidate: '{name}'...")
        indices = generate_summary_with_mmr(
            sentences_data, 
            num_clusters, 
            **params
        )
        # Use a tuple to make the list hashable for the set
        candidate_indices_set.add(tuple(indices))
        
    return [list(indices) for indices in candidate_indices_set]

# ==============================================================================
# NEW: STAGE 2 - CANDIDATE RE-RANKING
# ==============================================================================
def rerank_candidates(candidates, sentences_data, num_clusters, n_original):
    """
    Scores each candidate summary based on global properties (structure, balance, coherence)
    and returns the single best one.
    """
    print("\n--- Re-ranking Candidate Summaries ---")
    sentence_lookup = {s['id']: s for s in sentences_data}
    
    # Weights for the re-ranking scores
    rerank_weights = {
        'structure': 0.4,
        'balance': 0.3,
        'coherence': 0.3
    }
    
    scored_candidates = []
    
    for i, candidate_ids in enumerate(candidates):
        if not candidate_ids: continue

        # 1. Structural Integrity Score
        has_first = 1 if 0 in candidate_ids else 0
        has_last = 1 if (n_original - 1) in candidate_ids else 0
        structure_score = (has_first + has_last) / 2.0
        
        # 2. Topic Balance Score (Normalized Entropy)
        cluster_ids = [sentence_lookup[sid]['cluster_id'] for sid in candidate_ids]
        distribution = np.bincount(cluster_ids, minlength=num_clusters)
        probs = distribution / sum(distribution)
        # Use log base num_clusters for normalization, add epsilon for stability
        entropy = -sum(p * math.log(p, num_clusters) for p in probs if p > 0)
        balance_score = entropy

        # 3. Coherence Score (Average similarity of adjacent sentences)
        coherence_scores = []
        for j in range(len(candidate_ids) - 1):
            emb1 = sentence_lookup[candidate_ids[j]]['embedding']
            emb2 = sentence_lookup[candidate_ids[j+1]]['embedding']
            coherence_scores.append(cosine_similarity([emb1], [emb2])[0][0])
        coherence_score = np.mean(coherence_scores) if coherence_scores else 0
        
        # Final Holistic Score
        final_score = (
            rerank_weights['structure'] * structure_score +
            rerank_weights['balance'] * balance_score +
            rerank_weights['coherence'] * coherence_score
        )
        
        scored_candidates.append({
            'id': i,
            'indices': candidate_ids,
            'score': final_score,
            'structure': structure_score,
            'balance': balance_score,
            'coherence': coherence_score
        })

    # Print the scoring for transparency
    print("Candidate Scores:")
    for cand in sorted(scored_candidates, key=lambda x: x['score'], reverse=True):
        print(f"  Candidate {cand['id']}: Final Score={cand['score']:.3f} "
              f"(Structure={cand['structure']:.2f}, Balance={cand['balance']:.2f}, Coherence={cand['coherence']:.2f})")
        
    # Return the indices of the best candidate
    best_candidate = max(scored_candidates, key=lambda x: x['score'])
    return best_candidate['indices']

# ==============================================================================
# NEW: TOP-LEVEL ORCHESTRATOR
# ==============================================================================
def generate_and_rerank_summary(text, title):
    """
    Orchestrates the entire advanced pipeline:
    1. Pre-process and Score sentences
    2. Generate multiple candidate summaries
    3. Re-rank candidates to find the best one
    4. Format and return the final summary string
    """
    sentences_data, unfiltered_map = preprocess(text)
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data, unfiltered_map)
    
    candidate_indices = generate_candidate_summaries(sentences_data, num_clusters)
    
    best_indices = rerank_candidates(candidate_indices, sentences_data, num_clusters, len(unfiltered_map)+1)
    
    # Format final summary
    sentence_lookup = {s['id']: s for s in sentences_data}
    summary_text = " ".join([sentence_lookup[i]['original'] for i in best_indices])
    
    return summary_text, len(best_indices), len(unfiltered_map)+1

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    TITLE = input("Enter the title for the text: ")
    print("Enter or paste the text to summarize. Press CTRL+Z and then Enter to finish.")
    TEXT = ""
    while True:
        try:
            line = input()
            TEXT += line + "\n"
        except EOFError:
            break

    summary, num_sents_summary, num_sents_original = generate_and_rerank_summary(TEXT, TITLE)
    
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