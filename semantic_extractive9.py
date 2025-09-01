import numpy as np
import networkx as nx
import spacy
import hdbscan
import requests # NEW: For communicating with LM Studio
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# ==============================================================================
# PHASES 1-4: PRE-PROCESSING, EMBEDDING, CLUSTERING, SCORING (No changes)
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

def extract_keywords_tfidf(sentences_data, top_n=15):
    all_tokens = [' '.join(s['tokens']) for s in sentences_data]
    if not all_tokens: return set()
    vectorizer = TfidfVectorizer(max_features=top_n)
    vectorizer.fit_transform(all_tokens)
    return set(vectorizer.get_feature_names_out())

def score_sentences(sentences_data, unfiltered_map, custom_weights=None):
    print("Loading spaCy model for NER...")
    nlp = spacy.load("en_core_web_sm")
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_filtered = len(sentences_data)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else n_filtered
    keywords = extract_keywords_tfidf(sentences_data)
    print(f"Extracted top keywords: {list(keywords)[:10]}...")
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
        if original_id == 0: structural_bonus = 1.0
        elif original_id == n_original - 1: structural_bonus = 1.0
        s_data['structural_bonus'] = structural_bonus
        key_concept_count = len(set(s_data['tokens']) & keywords)
        s_data['key_concept_score'] = key_concept_count / len(keywords) if keywords else 0
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores) > 0:
        norm_info_scores = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info_scores[i]
    if custom_weights:
        weights = custom_weights
    else:
        weights = {'pagerank_score': 0.25, 'cluster_centrality_score': 0.20, 'key_concept_score': 0.20, 'global_cohesion_score': 0.10, 'info_density_score': 0.10, 'structural_bonus': 0.15}
    for s_data in sentences_data:
        score = sum(s_data.get(key, 0) * w for key, w in weights.items())
        s_data['relevance_score'] = score
    return sentences_data
# ==============================================================================

# ==============================================================================
# PHASE 5: RE-ARCHITECTED SUMMARY GENERATION
# ... (Functions _select_sentences_for_candidate, generate_candidate_summaries, rerank_candidates are the same) ...
def _select_sentences_for_candidate(sentences_data, num_clusters, allocation_config, mmr_config):
    sentence_lookup = {s['id']: s for s in sentences_data}
    detail_level = allocation_config.get('detail_level', 'Balanced')
    base_sents_per_cluster = {'Concise': 1, 'Balanced': 2, 'Detailed': 3}[detail_level]
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1: clusters[s_data['cluster_id']].append(s_data)
    total_sentences_needed = sum(min(base_sents_per_cluster, len(sents)) for sents in clusters.values())
    num_to_pick_from_cluster = {cid: min(base_sents_per_cluster, len(clusters.get(cid, []))) for cid in range(num_clusters)}
    final_summary_indices = []
    lambda_relevance = mmr_config.get('lambda_relevance', 0.5)
    lambda_coherence = mmr_config.get('lambda_coherence', 0.3)
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

def generate_candidate_summaries(sentences_data, num_clusters, n_original, detail_level, unfiltered_map):
    print("\n--- Generating Candidate Summaries with specialized 'personalities' ---")
    candidates = []
    personalities = {"coverage_focused": {'lambda_relevance': 0.5, 'lambda_coherence': 0.3}, "coherence_focused": {'lambda_relevance': 0.3, 'lambda_coherence': 0.6}}
    for name, mmr_params in personalities.items():
        print(f"Generating candidate: '{name}'...")
        indices = _select_sentences_for_candidate(sentences_data, num_clusters, {'detail_level': detail_level}, mmr_params)
        candidates.append(indices)
    print("Generating candidate: 'density_focused'...")
    density_weights = {'info_density_score': 0.50, 'key_concept_score': 0.30, 'pagerank_score': 0.10, 'cluster_centrality_score': 0.10, 'global_cohesion_score': 0.0, 'structural_bonus': 0.0}
    density_scored_sents = score_sentences(sentences_data[:], unfiltered_map, custom_weights=density_weights)
    density_indices = _select_sentences_for_candidate(density_scored_sents, num_clusters, {'detail_level': detail_level}, {'lambda_relevance': 0.7, 'lambda_coherence': 0.1})
    candidates.append(density_indices)
    print("Generating candidate: 'structure_focused'...")
    struct_candidate = []
    sentence_lookup = {s['id']: s for s in sentences_data}
    if 0 in sentence_lookup: struct_candidate.append(0)
    if (n_original - 1) in sentence_lookup: struct_candidate.append(n_original - 1)
    coverage_indices = _select_sentences_for_candidate(sentences_data, num_clusters, {'detail_level': detail_level}, {'lambda_relevance': 0.5, 'lambda_coherence': 0.3})
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
# ==============================================================================

## NEW ## - PHASE 6: ABSTRACTIVE POLISHING WITH LLM
def polish_summary_with_llm(extractive_summary_text, title):
    """
    Sends the extractive summary to a local LLM via LM Studio for polishing.
    """
    print("\n--- Polishing summary with local LLM via LM Studio ---")
    
    # The API endpoint for LM Studio is compatible with OpenAI's API
    url = "http://localhost:1234/v1/chat/completions"
    
    # The prompt is the most important part of this function
    prompt = (
        "You are an expert editor and professional writer. Your task is to rewrite the following set of "
        "disconnected facts into a single, cohesive, and fluent summary paragraph. "
        "The summary should be about the topic: '{title}'.\n\n"
        "RULES:\n"
        "- You MUST preserve all the key information and named entities from the original facts.\n"
        "- You MUST NOT add any new facts, opinions, or information that is not present below.\n"
        "- Your goal is to improve the narrative flow, resolve pronouns, and add natural transition words.\n\n"
        "Here are the key facts to synthesize:\n"
        "--- START OF FACTS ---\n"
        "{facts}\n"
        "--- END OF FACTS ---\n\n"
        "Now, provide the polished, single-paragraph summary:"
    ).format(title=title, facts=extractive_summary_text)
    
    headers = {"Content-Type": "application/json"}
    
    # The payload for the chat completions endpoint
    payload = {
        "model": "loaded-model", # This is a placeholder, LM Studio uses the model you've loaded
        "messages": [
            {"role": "system", "content": "You are a helpful editing assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3, # Lower temperature for more deterministic, factual output
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Extract the content from the response
        response_json = response.json()
        polished_summary = response_json['choices'][0]['message']['content']
        return polished_summary.strip()
        
    except requests.exceptions.RequestException as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Could not connect to LM Studio server.      !!!")
        print("!!! Please ensure LM Studio is running and the server  !!!")
        print("!!! has been started on the 'Local Server' tab.        !!!")
        print(f"!!! Details: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return f"[ERROR: Could not connect to LLM. Returning extractive summary as fallback.]\n\n{extractive_summary_text}"

# ==============================================================================
# FINAL ORCHESTRATOR
# ==============================================================================
def run_full_hybrid_pipeline(text, title, detail_level_str):
    
    # --- Extractive Stage ---
    sentences_data, unfiltered_map = preprocess(text)
    n_original = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    if not sentences_data: return "Could not process text.", 0, 0
    sentences_data = embed_sentences(sentences_data)
    sentences_data, num_clusters = cluster_sentences(sentences_data)
    sentences_data = score_sentences(sentences_data, unfiltered_map)
    candidate_indices = generate_candidate_summaries(sentences_data, num_clusters, n_original, detail_level_str, unfiltered_map)
    best_indices = rerank_candidates(candidate_indices, sentences_data, num_clusters, n_original)
    
    sentence_lookup = {s['id']: s for s in sentences_data}
    extractive_summary = " ".join([sentence_lookup[i]['original'] for i in best_indices])
    
    # --- Abstractive Stage ---
    final_polished_summary = polish_summary_with_llm(extractive_summary, title)
    
    return extractive_summary, final_polished_summary, len(best_indices), n_original

# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    # --- Get User Input ---
    detail_level_options = ['Concise', 'Balanced', 'Detailed']
    DETAIL_LEVEL = ""
    while DETAIL_LEVEL.capitalize() not in detail_level_options:
        DETAIL_LEVEL = input("Enter the detail level for the summary (Options: 'Concise', 'Balanced', 'Detailed'): ")
    DETAIL_LEVEL = DETAIL_LEVEL.capitalize()

    TITLE = input("Enter the title for the text: ")
    print("Enter or paste the text to summarize. Press CTRL+Z and then Enter (Windows) or CTRL+D (Linux/macOS) to finish.")
    TEXT = ""
    import sys
    TEXT = sys.stdin.read()
    
    # --- Execute the Full Pipeline ---
    extractive_sum, polished_sum, num_sents, n_orig = run_full_hybrid_pipeline(TEXT, TITLE, DETAIL_LEVEL)
    
    # --- Display Results ---
    print("\n\n\n=======================================================")
    print("           Extractive Summary (Fact Sheet)           ")
    print("=======================================================")
    print(extractive_sum)
    print("\n-------------------------------------------------------")
    print(f"STATS: {len(extractive_sum.split())} words, {num_sents} sentences selected from {n_orig}")
    print("-------------------------------------------------------")

    print("\n\n=======================================================")
    print("        FINAL Polished Summary (from LLM)        ")
    print("=======================================================")
    print(polished_sum)
    print("\n-------------------------------------------------------")
    print(f"STATS: {len(polished_sum.split())} words")
    print("-------------------------------------------------------")