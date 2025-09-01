import numpy as np
import networkx as nx
import spacy
import hdbscan # NEW: For dynamic topic discovery
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import math

# ==============================================================================
# PHASES 1 & 2: PRE-PROCESSING, EMBEDDING, CLUSTERING (MODIFIED)
# ==============================================================================
def preprocess(text):
    # ... (No changes from the bug-fixed version)
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
    # ... (No changes)
    print(f"Loading Sentence-BERT model '{model_name}'...")
    model = SentenceTransformer(model_name)
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = model.encode(original_sentences)
    for i, s_data in enumerate(sentences_data):
        s_data['embedding'] = embeddings[i]
    return sentences_data

## MODIFIED ##
def cluster_sentences(sentences_data, min_cluster_size=2):
    """
    MODIFIED: Uses HDBSCAN to dynamically find the optimal number of clusters
    and identify outlier sentences (noise).
    """
    print("Clustering with HDBSCAN to find natural topic clusters...")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    
    # HDBSCAN does not require 'k'. It finds clusters based on density.
    # min_cluster_size is the most important parameter to tune.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              metric='euclidean',
                              gen_min_span_tree=False)
    
    clusterer.fit(embeddings)
    
    # The number of clusters found is max label + 1. Label -1 is for noise.
    num_clusters = clusterer.labels_.max() + 1
    num_noise = np.sum(clusterer.labels_ == -1)
    
    print(f"HDBSCAN found {num_clusters} distinct topics and {num_noise} outlier sentences.")
    
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = clusterer.labels_[i]
        
    return sentences_data, num_clusters

# ==============================================================================
# PHASE 3 & 4: SCORING (No changes)
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
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data if s['cluster_id'] != -1)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1:
            clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}
    for i, s_data in enumerate(sentences_data):
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        if cid != -1:
            centroid = centroids[cid]
            s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroid])[0][0]
        else:
            s_data['cluster_centrality_score'] = 0
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
# PHASE 5: GENERATE-AND-RE-RANK (MODIFIED)
# ==============================================================================
## MODIFIED ##
def generate_candidate_summaries(sentences_data, num_clusters, n_original):
    """
    MODIFIED: Now includes a special 'structure_focused' candidate to ensure
    the re-ranker has a well-framed option to choose from.
    """
    print("\n--- Generating Candidate Summaries with different 'personalities' ---")
    
    personalities = {
        "balanced":        {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 0.7},
        "prefers_relevance": {'lambda_relevance': 0.7, 'lambda_coherence': 0.1, 'dynamic_compression_std_dev': 0.8},
        "prefers_coherence": {'lambda_relevance': 0.4, 'lambda_coherence': 0.5, 'dynamic_compression_std_dev': 0.6},
        "more_selective":  {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 1.0},
        "more_verbose":    {'lambda_relevance': 0.5, 'lambda_coherence': 0.3, 'dynamic_compression_std_dev': 0.4},
    }
    
    candidates = []
    for name, params in personalities.items():
        print(f"Generating candidate: '{name}'...")
        indices = generate_summary_with_mmr(sentences_data, num_clusters, **params)
        candidates.append(indices)
        
    ## NEW ## - Priority #1: Structurally-Biased Candidate Generation
    print("Generating candidate: 'structure_focused'...")
    # Determine a reasonable length for this candidate (e.g., average of others)
    avg_len = int(np.mean([len(c) for c in candidates if c])) if candidates else 4
    
    # Start with the first and last sentences if they exist
    structural_candidate = []
    sentence_lookup = {s['id']: s for s in sentences_data}
    if 0 in sentence_lookup: structural_candidate.append(0)
    if (n_original - 1) in sentence_lookup: structural_candidate.append(n_original - 1)
    
    # Fill the rest of the slots with the highest relevance sentences from the middle
    remaining_slots = avg_len - len(structural_candidate)
    if remaining_slots > 0:
        middle_sentences = [s for s in sentences_data if s['id'] not in [0, n_original - 1]]
        middle_sentences.sort(key=lambda s: s['relevance_score'], reverse=True)
        for s in middle_sentences[:remaining_slots]:
            structural_candidate.append(s['id'])
            
    structural_candidate.sort()
    candidates.append(structural_candidate)
    
    # Remove duplicates
    candidate_indices_set = {tuple(c) for c in candidates if c}
    return [list(indices) for indices in candidate_indices_set]

def generate_summary_with_mmr(sentences_data, num_clusters, 
                              lambda_relevance, lambda_coherence,
                              dynamic_compression_std_dev):
    # ... (The exact same bug-fixed logic as before) ...
    sentence_lookup = {s['id']: s for s in sentences_data}
    all_scores = [s['relevance_score'] for s in sentences_data]
    if not all_scores: return []
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    importance_threshold = mean_score + (dynamic_compression_std_dev * std_dev)
    total_sentences_needed = len([s for s in all_scores if s >= importance_threshold])
    total_sentences_needed = min(len(sentences_data), max(2, total_sentences_needed))
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1:
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
    return final_summary_indices

def rerank_candidates(candidates, sentences_data, num_clusters, n_original):
    # ... (No changes)
    print("\n--- Re-ranking Candidate Summaries ---")
    sentence_lookup = {s['id']: s for s in sentences_data}
    rerank_weights = {'structure': 0.4, 'balance': 0.3, 'coherence': 0.3}
    scored_candidates = []
    for i, candidate_ids in enumerate(candidates):
        if not candidate_ids: continue
        has_first = 1 if 0 in candidate_ids else 0
        has_last = 1 if (n_original - 1) in candidate_ids else 0
        structure_score = (has_first + has_last) / 2.0
        cluster_ids = [sentence_lookup[sid]['cluster_id'] for sid in candidate_ids if sid in sentence_lookup and sentence_lookup[sid]['cluster_id'] != -1]
        if not cluster_ids:
             balance_score = 0
        else:
             distribution = np.bincount(cluster_ids, minlength=num_clusters)
             probs = distribution / sum(distribution)
             entropy = -sum(p * math.log(p, num_clusters) for p in probs if p > 0)
             balance_score = entropy
        coherence_scores = []
        for j in range(len(candidate_ids) - 1):
            if candidate_ids[j] in sentence_lookup and candidate_ids[j+1] in sentence_lookup:
                emb1 = sentence_lookup[candidate_ids[j]]['embedding']
                emb2 = sentence_lookup[candidate_ids[j+1]]['embedding']
                coherence_scores.append(cosine_similarity([emb1], [emb2])[0][0])
        coherence_score = np.mean(coherence_scores) if coherence_scores else 0
        final_score = (rerank_weights['structure'] * structure_score + rerank_weights['balance'] * balance_score + rerank_weights['coherence'] * coherence_score)
        scored_candidates.append({'id': i, 'indices': candidate_ids, 'score': final_score, 'structure': structure_score, 'balance': balance_score, 'coherence': coherence_score})
    print("Candidate Scores:")
    for cand in sorted(scored_candidates, key=lambda x: x['score'], reverse=True):
        print(f"  Candidate {cand['id']}: Final Score={cand['score']:.3f} (Structure={cand['structure']:.2f}, Balance={cand['balance']:.2f}, Coherence={cand['coherence']:.2f})")
    best_candidate = max(scored_candidates, key=lambda x: x['score'])
    return best_candidate['indices']

def generate_and_rerank_summary(text, title):
    # ... (No changes)
    sentences_data, unfiltered_map = preprocess(text)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data, unfiltered_map)
    candidate_indices = generate_candidate_summaries(sentences_data, num_clusters, n_original)
    best_indices = rerank_candidates(candidate_indices, sentences_data, num_clusters, n_original)
    sentence_lookup = {s['id']: s for s in sentences_data}
    summary_text = " ".join([sentence_lookup[i]['original'] for i in best_indices])
    return summary_text, len(best_indices), n_original

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