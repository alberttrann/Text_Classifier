"""
summarizers/advanced_extractive.py (Final Version with XAI Logging)

This script contains the implementation for our advanced unsupervised and
hybrid summarization models, featuring the full Generate-and-Re-rank
architecture and detailed logging for UI transparency.
"""

import numpy as np
import networkx as nx
import spacy
import hdbscan
import requests
import json
import math
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

SBERT_MODEL = None
SPACY_NLP = None
NLI_PIPELINE = None

def _load_models():
    """Initializes global models if they haven't been loaded."""
    global SBERT_MODEL, SPACY_NLP, NLI_PIPELINE
    if SBERT_MODEL is None:
        print("Loading Sentence-BERT model (intfloat/multilingual-e5-large-instruct)...")
        SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    if SPACY_NLP is None:
        print("Loading spaCy model (en_core_web_lg)...")
        try:
            SPACY_NLP = spacy.load("en_core_web_lg")
        except OSError:
            print("ERROR: spaCy model 'en_core_web_lg' not found.")
    if NLI_PIPELINE is None:
        print("Loading NLI model (DeBERTa-v3-base-mnli)...")
        NLI_PIPELINE = pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=-1)

# ==============================================================================
# --- Helper Functions for the Pipeline ---
# ==============================================================================

def _embed(sentences_data, sbert_model):
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = sbert_model.encode(original_sentences, show_progress_bar=False)
    for i, s_data in enumerate(sentences_data):
        s_data['embedding'] = embeddings[i]
    return sentences_data
def _cluster(sentences_data, min_cluster_size=2):
    embeddings = np.array([s['embedding'] for s in sentences_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=False, allow_single_cluster=True)
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max() + 1
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = clusterer.labels_[i]
    return sentences_data, num_clusters

def _score_sentences(sentences_data, n_original, custom_weights=None):
    """Scores sentences based on a weighted combination of features."""
    embeddings = np.array([s['embedding'] for s in sentences_data])
    sim_matrix = cosine_similarity(embeddings)
    graph = nx.from_numpy_array(sim_matrix)
    pagerank_scores = nx.pagerank(graph)
    document_vector = np.mean(embeddings, axis=0)
    
    cluster_ids = set(s['cluster_id'] for s in sentences_data if s['cluster_id'] != -1)
    clusters = {cid: [] for cid in cluster_ids}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1:
            clusters[s_data['cluster_id']].append(s_data['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items() if embs}

    for i, s_data in enumerate(sentences_data):
        s_data['pagerank_score'] = pagerank_scores.get(i, 0)
        s_data['global_cohesion_score'] = cosine_similarity([s_data['embedding']], [document_vector])[0][0]
        cid = s_data['cluster_id']
        s_data['cluster_centrality_score'] = cosine_similarity([s_data['embedding']], [centroids[cid]])[0][0] if cid != -1 and cid in centroids else 0
        doc = SPACY_NLP(s_data['original'])
        s_data['info_density_score'] = len(doc.ents)
        s_data['structural_bonus'] = 1.0 if s_data['id'] == 0 or s_data['id'] == n_original - 1 else 0.0

    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores or [0]) > 0:
        norm_info = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info[i]
            
    if custom_weights is None:
        custom_weights = {'pagerank_score': 0.4, 'global_cohesion_score': 0.3, 'cluster_centrality_score': 0.2, 'info_density_score': 0.1}
    
    for s_data in sentences_data:
        s_data['relevance_score'] = sum(s_data.get(k, 0) * w for k, w in custom_weights.items())
    return sentences_data

def _select_sentences_for_candidate(sentences_data, num_clusters, allocation_config, mmr_config):
    sentence_lookup = {s['id']: s for s in sentences_data}
    detail_level = allocation_config.get('detail_level', 'Balanced')
    base_sents_per_cluster = {'Balanced': 2, 'Detailed': 3}[detail_level]
    
    clusters = {i: [] for i in range(num_clusters)}
    for s_data in sentences_data:
        if s_data['cluster_id'] != -1:
            clusters[s_data['cluster_id']].append(s_data)
            
    num_to_pick_from_cluster = {cid: min(base_sents_per_cluster, len(sents)) for cid, sents in clusters.items()}

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

def _generate_candidate_summaries(sentences_data, num_clusters, n_original, detail_level):
    """Generates multiple, distinct candidates using specialized strategies."""
    print("\n--- Generating Candidate Summaries with specialized 'personalities' ---")
    candidates = []
    
    # Personality 1: Coverage-Focused (balanced strategy)
    print("Generating candidate: 'coverage_focused'...")
    sentences_data = _score_sentences(sentences_data, n_original)
    coverage_indices = _select_sentences_for_candidate(
        sentences_data, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.5, 'lambda_coherence': 0.3}
    )
    candidates.append({"name": "Coverage-Focused", "indices": coverage_indices})

    # Personality 2: Coherence-Focused
    print("Generating candidate: 'coherence_focused'...")
    coherence_indices = _select_sentences_for_candidate(
        sentences_data, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.3, 'lambda_coherence': 0.6}
    )
    candidates.append({"name": "Coherence-Focused", "indices": coherence_indices})

    # Personality 3: Density-Focused
    print("Generating candidate: 'density_focused'...")
    density_weights = {'info_density_score': 0.6, 'pagerank_score': 0.4}
    density_scored_sents = _score_sentences(sentences_data[:], n_original, custom_weights=density_weights)
    density_indices = _select_sentences_for_candidate(
        density_scored_sents, num_clusters,
        allocation_config={'detail_level': detail_level},
        mmr_config={'lambda_relevance': 0.8, 'lambda_coherence': 0.1}
    )
    candidates.append({"name": "Density-Focused", "indices": density_indices})

    # Personality 4: Structure-Focused
    print("Generating candidate: 'structure_focused'...")
    struct_candidate = []
    sentence_lookup = {s['id']: s for s in sentences_data}
    if 0 in sentence_lookup: struct_candidate.append(0)
    if (n_original - 1) in sentence_lookup: struct_candidate.append(n_original - 1)
    
    fill_len = len(coverage_indices)
    remaining_slots = fill_len - len(struct_candidate)
    if remaining_slots > 0:
        middle_sentences = [idx for idx in coverage_indices if idx not in struct_candidate]
        struct_candidate.extend(middle_sentences[:remaining_slots])
    struct_candidate.sort()
    candidates.append({"name": "Structure-Focused", "indices": struct_candidate})
    
    unique_candidates = []
    seen_indices = set()
    for cand in candidates:
        indices_tuple = tuple(cand["indices"])
        if indices_tuple not in seen_indices:
            unique_candidates.append(cand)
            seen_indices.add(indices_tuple)
    return unique_candidates

def _rerank_candidates_with_details(candidates, sentences_data, num_clusters, n_original):
    """Scores each candidate summary based on global properties and returns the best one."""
    print("\n--- Re-ranking Candidate Summaries ---")
    sentence_lookup = {s['id']: s for s in sentences_data}
    rerank_weights = {'structure': 0.4, 'balance': 0.3, 'coherence': 0.3}
    
    scored_candidates = []
    
    for i, candidate in enumerate(candidates):
        candidate_ids = candidate["indices"]
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
        
        scored_candidates.append({
            'id': i, 'name': candidate.get('name', f"Candidate {i}"),
            'indices': candidate_ids, 'score': final_score,
            'structure': structure_score, 'balance': balance_score,
            'coherence': coherence_score
        })

    if not scored_candidates: return [], []
    
    best_candidate = max(scored_candidates, key=lambda x: x['score'])
    return best_candidate['indices'], scored_candidates

def _polish_with_llm(extractive_summary, title):
    """Simple polishing without fact-checking, returns prompt."""
    print("--- Polishing summary with local LLM via LM Studio ---")
    prompt = (
        "You are an expert editor. Your task is to rewrite the following set of "
        "disconnected key sentences into a single, cohesive, and fluent summary paragraph. "
        "The topic is '{title}'.\n\n"
        "RULES:\n"
        "1.  Preserve all key information and named entities.\n"
        "2.  Do not add any new facts, opinions, or information.\n"
        "3.  Improve the narrative flow, resolve pronouns, and add natural transition words.\n\n"
        "--- KEY SENTENCES TO SYNTHESIZE ---\n"
        "{facts}\n"
        "--- END OF KEY SENTENCES ---\n\n"
        "Polished, single-paragraph summary:"
    ).format(title=title, facts=extractive_summary)
    
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "loaded-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        polished_text = response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"!!! WARNING: LLM polishing failed. Returning extractive summary. Error: {e}")
        polished_text = extractive_summary

    details = {"initial_summary": polished_text, "prompt_sent_to_llm": prompt}
    return polished_text, details

def _polish_with_fact_checking(extractive_summary, original_article_text, title):
    """
    The most advanced polishing function. It performs three key steps:
    1.  Uses a high-constraint prompt to generate an initial polished summary.
    2.  Iterates through each sentence of the polished summary and uses a powerful
        NLI model to "fact-check" it against the original article.
    3.  If a sentence fails the fact-check, it asks the LLM to rewrite it
        until it is factually consistent, or discards it for safety.

    Returns:
        - final_summary (str): The verified and potentially corrected summary.
        - details (dict): A rich log of the entire fact-checking process for UI visualization.
    """
    print("\n--- Starting Advanced Polishing with Fact-Checking Loop ---")
    
    initial_prompt = (
        "You are a meticulous and fact-focused editor. Your task is to rewrite the following set of "
        "disconnected key sentences into a single, cohesive, and fluent summary paragraph. "
        "The topic is '{title}'.\n\n"
        "**CRITICAL RULES:**\n"
        "1.  **Absolute Faithfulness:** You MUST NOT add any new facts, statistics, opinions, or information that is not explicitly present in the provided key sentences. Your primary goal is to be factually grounded to the source facts.\n"
        "2.  **Synthesize, Don't Invent:** You should merge the ideas, improve the flow with transition words, and resolve pronouns. Do not invent details to connect ideas.\n"
        "3.  **Preserve Key Entities:** Ensure all named entities (people, places, organizations, specific numbers) from the key sentences are present in your final output.\n\n"
        "--- KEY SENTENCES TO SYNTHESIZE ---\n"
        "{facts}\n"
        "--- END OF KEY SENTENCES ---\n\n"
        "Now, provide the polished, single-paragraph summary:"
    ).format(title=title, facts=extractive_summary)


    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "loaded-model", "messages": [{"role": "user", "content": initial_prompt}], "temperature": 0.2}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        initial_polished_summary = response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        error_log = [{"status": "ERROR", "original": f"Initial LLM call failed: {e}"}]
        details = { "initial_summary": extractive_summary, "fact_checking_log": error_log, "prompt_sent_to_llm": initial_prompt }
        return extractive_summary, details

    print("Initial polished summary generated. Now entering NLI fact-checking loop...")
    try:
        polished_sents = sent_tokenize(initial_polished_summary)
        article_sents = sent_tokenize(original_article_text)
        if not polished_sents or not article_sents:
            return initial_polished_summary, {"initial_summary": initial_polished_summary, "fact_checking_log": [], "prompt_sent_to_llm": initial_prompt}
    except Exception:
        return initial_polished_summary, {"initial_summary": initial_polished_summary, "fact_checking_log": [], "prompt_sent_to_llm": initial_prompt}

    article_embeddings = SBERT_MODEL.encode(article_sents, convert_to_tensor=True, show_progress_bar=False)
    final_verified_sents = []
    fact_checking_log = []

    for sent in polished_sents:
        sent_embedding = SBERT_MODEL.encode(sent, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.pytorch_cos_sim(sent_embedding, article_embeddings)[0]
        premise_idx = torch.argmax(similarities).item()
        premise = article_sents[premise_idx]
        nli_results = NLI_PIPELINE(f"{premise}</s></s>{sent}")
        top_label = max(nli_results, key=lambda x: x['score'])
        
        if top_label['label'] == 'entailment' and top_label['score'] > 0.8:
            final_verified_sents.append(sent)
            fact_checking_log.append({"status": "PASSED", "original": sent, "premise": premise})
        else:
            rewrite_prompt = ( "You are a fact-correction assistant... REWRITTEN AND FACTUALLY ACCURATE SENTENCE ---" ).format(source=premise, sentence=sent)
            try:
                rewrite_payload = {"model": "loaded-model", "messages": [{"role": "user", "content": rewrite_prompt}], "temperature": 0.0}
                rewrite_response = requests.post(url, headers=headers, data=json.dumps(rewrite_payload), timeout=90)
                rewrite_response.raise_for_status()
                corrected_sent = rewrite_response.json()['choices'][0]['message']['content'].strip()
                final_verified_sents.append(corrected_sent)
                fact_checking_log.append({"status": "REWRITTEN", "original": sent, "corrected": corrected_sent, "premise": premise})
            except requests.exceptions.RequestException:
                fact_checking_log.append({"status": "DISCARDED", "original": sent, "premise": premise})

    final_summary = " ".join(final_verified_sents)
    details = {
        "initial_summary": initial_polished_summary,
        "fact_checking_log": fact_checking_log,
        "prompt_sent_to_llm": initial_prompt
    }
    return final_summary, details

# ==============================================================================
# --- MAIN ORCHESTRATOR ---
# ==============================================================================
def advanced_summarize(text, title, detail_level='Balanced', enable_hybrid=False, enable_fact_checking=False):
    """The main orchestrator for the advanced models."""
    _load_models()
    if SPACY_NLP is None: return "spaCy model is not loaded. Cannot proceed.", {}

    try:
        original_sentences = sent_tokenize(text)
        n_original = len(original_sentences)
    except Exception:
        return "Could not process text.", {}
    
    sentences_data = []
    filtered_idx_counter = 0
    for i, s in enumerate(original_sentences):
        sentences_data.append({'id': i, 'original': s, 'filtered_idx': -1})

    sentences_to_process = [s for s in sentences_data if len(s['original'].split()) > 3]
    for s in sentences_to_process:
        s['filtered_idx'] = filtered_idx_counter
        filtered_idx_counter += 1

    if not sentences_to_process: return "Input text has no sufficiently long sentences.", {}

    sentences_to_process = _embed(sentences_to_process, SBERT_MODEL)
    sentences_to_process, num_clusters = _cluster(sentences_to_process)
    
    candidate_indices = _generate_candidate_summaries(sentences_to_process, num_clusters, n_original, detail_level)
    best_indices, rerank_details = _rerank_candidates_with_details(candidate_indices, sentences_to_process, num_clusters, n_original)
    
    sentence_lookup = {s['id']: s for s in sentences_data}
    extractive_summary = " ".join([sentence_lookup[i]['original'] for i in best_indices])
    
    polished_summary = None
    polish_details = {}
    if enable_hybrid:
        if enable_fact_checking:
            polished_summary, polish_details = _polish_with_fact_checking(extractive_summary, text, title)
        else:
            polished_summary, polish_details = _polish_with_llm(extractive_summary, title)

    details = {
        "processed_sentences": sentences_data,
        "rerank_details": rerank_details,
        "selected_indices": best_indices,
        "extractive_summary": extractive_summary,
        "polish_details": polish_details
    }
    
    final_summary = polished_summary if enable_hybrid else extractive_summary
    return final_summary, details