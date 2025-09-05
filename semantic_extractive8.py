import numpy as np
import networkx as nx
import spacy
import hdbscan
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# ==============================================================================
# PHASES 1 & 2: PRE-PROCESSING, EMBEDDING, CLUSTERING (No changes)
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

def cluster_sentences(sentences_data, min_cluster_size=2):
    print("Clustering with HDBSCAN to find natural topic clusters...")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=False)
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max() + 1
    num_noise = np.sum(clusterer.labels_ == -1)
    print(f"HDBSCAN found {num_clusters} distinct topics and {num_noise} outlier sentences.")
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = clusterer.labels_[i]
    return sentences_data, num_clusters
# ==============================================================================

def extract_keywords_tfidf(sentences_data, top_n=15):
    all_tokens = [' '.join(s['tokens']) for s in sentences_data]
    if not all_tokens: return set()
    vectorizer = TfidfVectorizer(max_features=top_n)
    vectorizer.fit_transform(all_tokens)
    return set(vectorizer.get_feature_names_out())

## MODIFIED ##
def score_sentences(sentences_data, unfiltered_map, custom_weights=None):
    """
    MODIFIED: Now accepts custom weights to allow for specialized scoring
    by different candidate 'personalities'.
    """
    print("Loading spaCy model for NER...")
    nlp = spacy.load("en_core_web_sm")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_filtered = len(sentences_data)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else n_filtered

    keywords = extract_keywords_tfidf(sentences_data)
    print(f"Extracted top keywords: {list(keywords)[:10]}...")
    
    # ... (Graph, PageRank, Centroid calculations are the same) ...
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    pagerank_scores = nx.pagerank(graph)
    document_vector = np.mean(embeddings, axis=0)
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data if s['cluster_id'] != -1)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1: clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items() if embs}

    for i, s_data in enumerate(sentences_data):
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroids[cid]])[0][0] if cid != -1 and cid in centroids else 0
        doc = nlp(s_data['original'])
        s_data['info_density_score'] = len(doc.ents)
        structural_bonus = 0
        original_id = s_data['id']
        if original_id == 0: structural_bonus = 1.0 # Use 1.0 for clearer impact
        elif original_id == n_original - 1: structural_bonus = 1.0
        s_data['structural_bonus'] = structural_bonus
        key_concept_count = len(set(s_data['tokens']) & keywords)
        s_data['key_concept_score'] = key_concept_count / len(keywords) if keywords else 0

    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores) > 0:
        norm_info_scores = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info_scores[i]

    ## MODIFIED ## - Use custom weights if provided, otherwise use a balanced default
    if custom_weights:
        weights = custom_weights
    else:
        weights = {
            'pagerank_score': 0.25, 'cluster_centrality_score': 0.20,
            'key_concept_score': 0.20, 'global_cohesion_score': 0.10,
            'info_density_score': 0.10, 'structural_bonus': 0.15
        }
    
    for s_data in sentences_data:
        score = sum(s_data.get(key, 0) * w for key, w in weights.items())
        s_data['relevance_score'] = score
        
    return sentences_data

# ==============================================================================
# PHASE 5: RE-ARCHITECTED SUMMARY GENERATION
# ==============================================================================
## NEW ## - A flexible, powerful sentence selection engine
def _select_sentences_for_candidate(sentences_data, num_clusters, allocation_config, mmr_config):
    """
    A generic engine to select sentences based on allocation and MMR strategies.
    """
    sentence_lookup = {s['id']: s for s in sentences_data}
    
    # 1. Allocation Logic
    num_to_pick_from_cluster = {}
    
    # Map detail level to a base number of sentences per cluster
    detail_level = allocation_config.get('detail_level', 'Balanced')
    base_sents_per_cluster = {'Concise': 1, 'Balanced': 2, 'Detailed': 3}[detail_level]
    
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1: clusters[s_data['cluster_id']].append(s_data)

    # Calculate total slots based on the base number
    total_sentences_needed = sum(min(base_sents_per_cluster, len(sents)) for sents in clusters.values())
    
    # For now, we use a simple equitable allocation based on the detail level.
    # More complex logic (like importance-weighting) can be added here.
    for cid in range(num_clusters):
        num_to_pick_from_cluster[cid] = min(base_sents_per_cluster, len(clusters.get(cid, [])))

    # 2. MMR Selection Logic
    final_summary_indices = []
    lambda_relevance = mmr_config.get('lambda_relevance', 0.5)
    lambda_coherence = mmr_config.get('lambda_coherence', 0.3)
    lambda_redundancy = 1 - lambda_relevance - lambda_coherence

    for cid in sorted(num_to_pick_from_cluster.keys()):
        num_to_pick = num_to_pick_from_cluster.get(cid, 0)
        candidate_sentences = list(clusters.get(cid, []))
        
        for _ in range(num_to_pick):
            if not candidate_sentences: break
            # ... (MMR loop is the same) ...
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

## MODIFIED ##
def generate_candidate_summaries(sentences_data, num_clusters, n_original, unfiltered_map, detail_level):
    """
    MODIFIED: Generates truly distinct candidates using specialized strategies.
    """
    print("\n--- Generating Candidate Summaries with specialized 'personalities' ---")
    candidates = []
    
    # Personality 1: Coverage-Focused (our main, balanced strategy)
    print("Generating candidate: 'coverage_focused'...")
    coverage_indices = _select_sentences_for_candidate(
        sentences_data, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.5, 'lambda_coherence': 0.3}
    )
    candidates.append(coverage_indices)

    # Personality 2: Coherence-Focused
    print("Generating candidate: 'coherence_focused'...")
    coherence_indices = _select_sentences_for_candidate(
        sentences_data, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.3, 'lambda_coherence': 0.6} # High coherence weight
    )
    candidates.append(coherence_indices)

    # Personality 3: Density-Focused
    print("Generating candidate: 'density_focused'...")
    density_weights = { # New weights that prioritize facts
        'info_density_score': 0.50, 'key_concept_score': 0.30,
        'pagerank_score': 0.10, 'cluster_centrality_score': 0.10,
        'global_cohesion_score': 0.0, 'structural_bonus': 0.0
    }
    # Create a temporary copy with new scores
    density_scored_sents = score_sentences(sentences_data[:], unfiltered_map, custom_weights=density_weights)
    density_indices = _select_sentences_for_candidate(
        density_scored_sents, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.7, 'lambda_coherence': 0.1} # High relevance
    )
    candidates.append(density_indices)

    # Personality 4: Structure-Focused
    print("Generating candidate: 'structure_focused'...")
    struct_candidate = []
    sentence_lookup = {s['id']: s for s in sentences_data}
    if 0 in sentence_lookup: struct_candidate.append(0)
    if (n_original - 1) in sentence_lookup: struct_candidate.append(n_original - 1)
    
    # Fill the middle with the sentences from the balanced 'coverage' candidate
    fill_len = len(coverage_indices)
    remaining_slots = fill_len - len(struct_candidate)
    if remaining_slots > 0:
        middle_sentences = [idx for idx in coverage_indices if idx not in struct_candidate]
        struct_candidate.extend(middle_sentences[:remaining_slots])
            
    struct_candidate.sort()
    candidates.append(struct_candidate)
    
    candidate_indices_set = {tuple(c) for c in candidates if c}
    return [list(indices) for indices in candidate_indices_set]

def rerank_candidates(candidates, sentences_data, num_clusters, n_original):
    # ... (No changes from previous version) ...
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
        if not cluster_ids or num_clusters <= 1:
            balance_score = 1.0
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
    if not scored_candidates: return []
    best_candidate = max(scored_candidates, key=lambda x: x['score'])
    return best_candidate['indices']

def generate_and_rerank_summary(text, title, detail_level_str):
    sentences_data, unfiltered_map = preprocess(text)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    if not sentences_data: return "Could not process text.", 0, 0
    
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    
    # The default scoring is now just a baseline, as personalities can override it
    sentences_data = score_sentences(sentences_data, unfiltered_map)
    
    candidate_indices = generate_candidate_summaries(sentences_data, num_clusters, n_original, unfiltered_map, detail_level_str)
    
    best_indices = rerank_candidates(candidate_indices, sentences_data, num_clusters, n_original)
    
    sentence_lookup = {s['id']: s for s in sentences_data}
    summary_text = " ".join([sentence_lookup[i]['original'] for i in best_indices])
    
    return summary_text, len(best_indices), n_original

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    ## NEW ## - Priority #3: User-friendly "Detail Level" Slider
    DETAIL_LEVEL = input("Enter the detail level for the summary (Options: 'Concise', 'Balanced', 'Detailed'): ")

    TITLE = input("Enter the title for the text: ")
    print("Enter or paste the text to summarize. Press CTRL+Z and then Enter to finish.")
    TEXT = ""
    while True:
        try:
            line = input()
            TEXT += line + "\n"
        except EOFError:
            break

    summary, num_sents_summary, num_sents_original = generate_and_rerank_summary(TEXT, TITLE, DETAIL_LEVEL)
    
    print(f"\n--- Final Summary for '{TITLE}' at '{DETAIL_LEVEL}' detail level ---")
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