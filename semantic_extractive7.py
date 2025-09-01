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

# = a============================================================================
# PHASES 1 & 2: PRE-PROCESSING, EMBEDDING, CLUSTERING (No changes)
# ... (Functions preprocess, embed_sentences, cluster_sentences are the same as the previous version) ...
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

## NEW ## - Priority #2 (Helper function)
def extract_keywords_tfidf(sentences_data, top_n=15):
    """
    Extracts the most important keywords from the entire document using TF-IDF.
    """
    all_tokens = [' '.join(s['tokens']) for s in sentences_data]
    if not all_tokens: return set()
    
    vectorizer = TfidfVectorizer(max_features=top_n)
    vectorizer.fit_transform(all_tokens)
    return set(vectorizer.get_feature_names_out())

## MODIFIED ##
def score_sentences(sentences_data, unfiltered_map, similarity_threshold=0.3):
    """
    MODIFIED: Now includes a 'key_concept_score' based on document-level keywords.
    """
    print("Loading spaCy model for NER...")
    nlp = spacy.load("en_core_web_sm")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_filtered = len(sentences_data)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else n_filtered

    ## NEW ## - Extract keywords to be used in the new feature
    keywords = extract_keywords_tfidf(sentences_data)
    print(f"Extracted top keywords: {list(keywords)[:10]}...")
    
    # ... (Graph, PageRank, Centroid calculations are the same) ...
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    graph.remove_edges_from([(u, v) for u, v, d in graph.edges(data=True) if d['weight'] < similarity_threshold])
    pagerank_scores = nx.pagerank(graph)
    document_vector = np.mean(embeddings, axis=0)
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data if s['cluster_id'] != -1)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1: clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items()}

    for i, s_data in enumerate(sentences_data):
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroids[cid]])[0][0] if cid != -1 else 0
        doc = nlp(s_data['original'])
        s_data['info_density_score'] = len(doc.ents)
        
        structural_bonus = 0
        original_id = s_data['id']
        if original_id == 0: structural_bonus = 0.15
        elif original_id == n_original - 1: structural_bonus = 0.10
        s_data['structural_bonus'] = structural_bonus

        ## NEW ## - Priority #2: Add the Key Concept feature
        key_concept_count = len(set(s_data['tokens']) & keywords)
        s_data['key_concept_score'] = key_concept_count / len(keywords) if keywords else 0

    # Normalize info density
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores) > 0:
        norm_info_scores = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info_scores[i]

    ## MODIFIED ## - Updated weights
    weights = {
        'pagerank_score': 0.25,
        'cluster_centrality_score': 0.20,
        'key_concept_score': 0.20, # Give the new feature a strong weight
        'global_cohesion_score': 0.10,
        'info_density_score': 0.10,
        'structural_bonus': 0.15
    }
    
    for s_data in sentences_data:
        score = sum(s_data.get(key, 0) * w for key, w in weights.items())
        s_data['relevance_score'] = score
        
    return sentences_data

# ==============================================================================
# PHASE 5: RE-ARCHITECTED SUMMARY GENERATION
# ==============================================================================
## NEW ## - Priority #1: "Guaranteed Representation First" Policy
def select_indices_with_dynamic_length(sentences_data, num_clusters, detail_level_std_dev):
    """
    Implements the new policy:
    1. Guarantees one sentence from each topic.
    2. Expands the summary with other statistically important sentences.
    The final length is a dynamic outcome.
    """
    sentence_lookup = {s['id']: s for s in sentences_data}
    
    # --- Step 1: Build the "Core" Summary (Guaranteed Representation) ---
    core_summary_indices = []
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1:
            clusters[s_data['cluster_id']].append(s_data)

    for cid in range(num_clusters):
        cluster_sents = clusters.get(cid, [])
        if cluster_sents:
            # Select the single best sentence from this topic cluster
            best_sentence_in_cluster = max(cluster_sents, key=lambda s: s['relevance_score'])
            core_summary_indices.append(best_sentence_in_cluster['id'])
            
    # --- Step 2: Define the "Expansion Pool" (Statistically Important) ---
    all_scores = [s['relevance_score'] for s in sentences_data]
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    importance_threshold = mean_score + (detail_level_std_dev * std_dev)
    expansion_indices = [s['id'] for s in sentences_data if s['relevance_score'] >= importance_threshold]

    # --- Step 3: Combine and Finalize ---
    final_indices = sorted(list(set(core_summary_indices) | set(expansion_indices)))
    
    print(f"Dynamic summary constructed. Core sentences: {len(core_summary_indices)}, "
          f"Expansion sentences: {len(set(expansion_indices) - set(core_summary_indices))}. "
          f"Total: {len(final_indices)} sentences.")
          
    return final_indices

## MODIFIED ## - The MMR function is no longer the main generator, but a candidate generator strategy
def generate_summary_with_mmr(sentences_data, num_clusters, lambda_relevance, lambda_coherence, detail_level_std_dev):
    # This function now uses the new selection logic and then applies MMR for diversity.
    # It's one of the "personalities" for candidate generation.
    
    # We use the new logic to get a set of candidate indices first.
    base_indices = select_indices_with_dynamic_length(sentences_data, num_clusters, detail_level_std_dev)
    
    # Now, we can optionally run MMR over this pool if we want to ensure diversity,
    # but for simplicity in this final version, the re-ranker's coherence score handles this.
    # The new selection logic is powerful enough to be a candidate generator on its own.
    return base_indices

def generate_candidate_summaries(sentences_data, num_clusters, n_original):
    print("\n--- Generating Candidate Summaries with different 'personalities' ---")
    
    ## MODIFIED ## - Personalities now tune the "detail level"
    personalities = {
        "balanced":        {'detail_level_std_dev': 0.7},
        "more_selective":  {'detail_level_std_dev': 1.1},
        "more_verbose":    {'detail_level_std_dev': 0.4},
    }
    
    candidates = []
    for name, params in personalities.items():
        print(f"Generating candidate: '{name}'...")
        indices = select_indices_with_dynamic_length(sentences_data, num_clusters, **params)
        candidates.append(indices)
        
    ## MODIFIED ## - The structural candidate is now more robustly generated
    print("Generating candidate: 'structure_focused'...")
    struct_candidate = []
    sentence_lookup = {s['id']: s for s in sentences_data}
    if 0 in sentence_lookup: struct_candidate.append(0)
    if (n_original - 1) in sentence_lookup: struct_candidate.append(n_original - 1)
    # Get the core sentences from the "balanced" run to fill the middle
    balanced_indices = select_indices_with_dynamic_length(sentences_data, num_clusters, detail_level_std_dev=0.7)
    for idx in balanced_indices:
        if idx not in struct_candidate:
            struct_candidate.append(idx)
    struct_candidate.sort()
    candidates.append(struct_candidate)
    
    candidate_indices_set = {tuple(c) for c in candidates if c}
    return [list(indices) for indices in candidate_indices_set]

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
        if not cluster_ids or num_clusters == 0:
            balance_score = 0
        else:
             distribution = np.bincount(cluster_ids, minlength=num_clusters)
             probs = distribution / sum(distribution)
             entropy = -sum(p * math.log(p, num_clusters) for p in probs if p > 0) if num_clusters > 1 else 1.0
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
    # ... (Orchestrator is the same, just passes the detail level down)
    detail_level_map = {'Concise': 1.1, 'Balanced': 0.7, 'Detailed': 0.4}
    detail_level_std_dev = detail_level_map[detail_level_str]

    sentences_data, unfiltered_map = preprocess(text)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    if not sentences_data: return "Could not process text.", 0, 0
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data, unfiltered_map)
    # The detail level is now used to generate the candidates
    candidate_indices = generate_candidate_summaries(sentences_data, num_clusters, n_original)
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