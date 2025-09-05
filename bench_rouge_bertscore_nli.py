import os
import re
import math
import json
import time
import requests
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import networkx as nx
import spacy
import hdbscan
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
from tqdm import tqdm

# --- Global Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# --- MODEL DEFINITIONS ---
# ==============================================================================

# --- Global Models (loaded once) ---
print("Loading global models (SBERT, spaCy, NLI)... This may take a moment.")
SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=DEVICE)
SPACY_NLP = spacy.load("en_core_web_sm")
NLI_PIPELINE = hf_pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=0 if torch.cuda.is_available() else -1)
print("Global models loaded.")

# --- Model 1: Original Paper Method ---
def model_1_summarize(text, title, num_sentences, sim_threshold=0.3):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    try: original_sentences = sent_tokenize(text)
    except Exception: return ""
    sentences_data = []
    for i, sentence in enumerate(original_sentences):
        words = word_tokenize(sentence.lower())
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        if len(tokens) > 3:
            sentences_data.append({'id': i, 'original': sentence, 'tokens': tokens})
    if not sentences_data: return ""
    sentence_strings = [' '.join(s['tokens']) for s in sentences_data]
    vectorizer = TfidfVectorizer()
    try: tfidf_matrix = vectorizer.fit_transform(sentence_strings)
    except ValueError: return ""
    sim_matrix = cosine_similarity(tfidf_matrix)
    n = sim_matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > sim_threshold:
                graph.add_edge(i, j)
    cliques = list(nx.find_cliques(graph))
    triangles = [clique for clique in cliques if len(clique) == 3]
    bitvector = np.zeros(n)
    if triangles:
        nodes_in_triangles = set(node for tri in triangles for node in tri)
        for node_index in nodes_in_triangles:
            bitvector[node_index] = 1
    title_words = set([stemmer.stem(w.lower()) for w in word_tokenize(title) if w.isalnum()])
    scores = np.zeros(n)
    is_fallback = (np.sum(bitvector) == 0)
    for i, s_data in enumerate(sentences_data):
        feature_sum = (len(set(s_data['tokens']) & title_words) +
                       len(s_data['tokens']) +
                       ((n - i) / n) +
                       len(re.findall(r'\d+', s_data['original'])) +
                       np.sum(sim_matrix[i, :]))
        if is_fallback: scores[i] = feature_sum
        else: scores[i] = bitvector[i] * feature_sum
    ## MODIFIED ## - Use the explicitly passed num_sentences
    num_sentences_to_select = min(n, max(1, num_sentences))
    if is_fallback:
        candidate_indices = np.argsort(scores)[::-1][:num_sentences_to_select]
    else:
        triangle_indices = np.where(bitvector == 1)[0]
        if len(triangle_indices) < num_sentences_to_select or len(triangle_indices) == 0:
             candidate_indices = np.argsort(scores)[::-1][:num_sentences_to_select]
        else:
             triangle_scores = scores[triangle_indices]
             top_triangle_indices_local = np.argsort(triangle_scores)[::-1][:num_sentences_to_select]
             candidate_indices = triangle_indices[top_triangle_indices_local]
    selected_indices = sorted(list(candidate_indices))
    summary = " ".join([sentences_data[i]['original'] for i in selected_indices if i < len(sentences_data)])
    return summary

# --- Advanced Models Helper Functions ---
def _preprocess(text):
    # ... (This function is correct and unchanged)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    try: original_sentences = sent_tokenize(text)
    except Exception: return [], {}
    processed_data, unfiltered_map = [], {}
    idx = 0
    for i, sentence in enumerate(original_sentences):
        words = word_tokenize(sentence.lower())
        tokens = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
        if len(tokens) > 3:
            processed_data.append({'id': i, 'original': sentence, 'tokens': tokens})
            unfiltered_map[i] = idx
            idx += 1
    return processed_data, unfiltered_map
def _embed(sentences_data, sbert_model):
    # ... (This function is correct and unchanged)
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = sbert_model.encode(original_sentences, show_progress_bar=False)
    for i, s_data in enumerate(sentences_data):
        s_data['embedding'] = embeddings[i]
    return sentences_data
def _cluster(sentences_data, min_cluster_size=2):
    # ... (This function is correct and unchanged)
    embeddings = np.array([s['embedding'] for s in sentences_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=False, allow_single_cluster=True)
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max() + 1
    for i, s_data in enumerate(sentences_data):
        s_data['cluster_id'] = clusterer.labels_[i]
    return sentences_data, num_clusters
def _score(sentences_data, unfiltered_map):
    # ... (This function is correct and unchanged)
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_orig = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    graph = nx.from_numpy_array(cosine_similarity(embeddings))
    pagerank_scores = nx.pagerank(graph)
    doc_vector = np.mean(embeddings, axis=0)
    for s in sentences_data:
        s['pagerank_score'] = pagerank_scores.get(s['id'], 0)
        s['global_cohesion_score'] = cosine_similarity([s['embedding']], [doc_vector])[0][0]
        s['info_density_score'] = len(SPACY_NLP(s['original']).ents)
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores or [0]) > 0:
        norm_info = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info[i]
    weights = {'pagerank_score': 0.4, 'global_cohesion_score': 0.3, 'info_density_score': 0.3}
    for s in sentences_data:
        s['relevance_score'] = sum(s.get(k, 0) * w for k, w in weights.items())
    return sentences_data

# --- Model 2 & 3: Advanced Pipeline ---
## MODIFIED ## - Corrected function signature
def advanced_pipeline(precomputed_cache, title, target_num_sentences):
    def _select(sentences_data, target_len):
        sorted_sents = sorted(sentences_data, key=lambda s: s['relevance_score'], reverse=True)
        top_indices = [s['id'] for s in sorted_sents[:target_len]]
        top_indices.sort()
        return top_indices
    def _polish(extractive_text, title):
        prompt = f"You are an expert editor. Rewrite the following disconnected facts about '{title}' into a single, cohesive, and fluent summary paragraph. Preserve all key information and named entities. Do not add new information.\n\nFACTS:\n{extractive_text}\n\nPOLISHED SUMMARY:"
        url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "loaded-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException:
            return f"[LLM polish failed] {extractive_text}"
            
    sentences_data = precomputed_cache['sentences_data']
    best_indices = _select(sentences_data, target_num_sentences)
    sentence_lookup = {s['id']: s for s in sentences_data}
    extractive_summary = " ".join([sentence_lookup.get(i, {}).get('original', '') for i in best_indices])
    polished_summary = _polish(extractive_summary, title)
    return extractive_summary, polished_summary

# --- Model 4: LLM-Only ---
## MODIFIED ## - Corrected function signature
def model_4_llm_only(text, title, target_num_sentences):
    if not text or not isinstance(text, str): return ""
    detail_instruction = f"The summary MUST contain exactly {target_num_sentences} sentences."
    prompt = f"You are an expert summarizer. Create a factually accurate, abstractive summary of the following text about '{title}'.\n\nINSTRUCTIONS:\n1. {detail_instruction}\n2. The summary MUST be based ONLY on the provided text.\n3. The summary must be a single, cohesive paragraph.\n\nTEXT:\n{text}\n\nSUMMARY:"
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "loaded-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException:
        return "[LLM request failed]"

# --- Baseline Model: TextRank ---
def textrank_baseline(text, num_sentences):
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences: return " ".join(sentences)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
        summary_sentences.sort(key=lambda s: sentences.index(s))
        return " ".join(summary_sentences)
    except Exception: return ""

# --- NLI Scorer ---
def nli_score(article_text, summary_text):
    if not article_text or not summary_text: return 0.0, 0.0
    try:
        article_sentences = sent_tokenize(article_text)
        summary_sentences = sent_tokenize(summary_text)
        if not article_sentences or not summary_sentences: return 0.0, 0.0
    except Exception: return 0.0, 0.0
    article_embeddings = SBERT_MODEL.encode(article_sentences, convert_to_tensor=True, show_progress_bar=False)
    entailment_scores, contradiction_scores = [], []
    for summary_sent in summary_sentences:
        summary_sent_embedding = SBERT_MODEL.encode(summary_sent, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.pytorch_cos_sim(summary_sent_embedding, article_embeddings)[0]
        premise_idx = torch.argmax(similarities).item()
        premise_sent = article_sentences[premise_idx]
        nli_results = NLI_PIPELINE(f"{premise_sent}</s></s>{summary_sent}")
        entailment_score, contradiction_score = 0.0, 0.0
        for result in nli_results:
            if result['label'] == 'entailment': entailment_score = result['score']
            elif result['label'] == 'contradiction': contradiction_score = result['score']
        entailment_scores.append(entailment_score)
        contradiction_scores.append(contradiction_score)
    return np.mean(entailment_scores) if entailment_scores else 0.0, np.max(contradiction_scores) if contradiction_scores else 0.0

# --- Dataset Loader ---
def load_bbc_dataset(base_path):
    print(f"Loading dataset from: {base_path}")
    all_data = []
    articles_path = Path(base_path) / "News Articles"
    summaries_path = Path(base_path) / "Summaries"
    for category_path in articles_path.iterdir():
        if category_path.is_dir():
            category = category_path.name
            for article_file in category_path.glob("*.txt"):
                try:
                    with open(article_file, 'r', encoding='utf-8', errors='ignore') as f: article_content = f.read()
                    summary_file = summaries_path / category / article_file.name
                    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f: summary_content = f.read()
                    all_data.append({"filename": article_file.name, "category": category, "article": article_content, "reference_summary": summary_content})
                except Exception as e:
                    print(f"Skipping file {article_file.name} due to error: {e}")
    return pd.DataFrame(all_data)

# ==============================================================================
# --- MAIN BENCHMARKING SCRIPT ---
# ==============================================================================
def run_grand_benchmark():
    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
        
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nRunning grand quantitative benchmark on {len(test_df)} articles.")

    models_to_benchmark = {
        "TextRank_Baseline": {"type": "textrank"},
        "Original_Paper_Method": {"type": "original"},
        "Advanced_Extractive": {"type": "advanced_extractive"},
        "Hybrid": {"type": "hybrid"},
        "LLM_Only": {"type": "llm_only"},
    }

    generated_summaries = {model_name: [] for model_name in models_to_benchmark}
    reference_summaries = list(test_df['reference_summary'])
    articles = list(test_df['article'])

    pbar_generate = tqdm(range(len(test_df)), desc="Generating Summaries")
    for i in pbar_generate:
        article = articles[i]
        reference = reference_summaries[i]
        title = sent_tokenize(article)[0] if article and sent_tokenize(article) else "Article"
        
        try: target_len = len(sent_tokenize(reference))
        except: target_len = 3
        
        s_data, u_map = _preprocess(article)
        if not s_data:
            for model_name in models_to_benchmark: generated_summaries[model_name].append("")
            continue
        s_data = _embed(s_data, SBERT_MODEL)
        s_data, n_clusters = _cluster(s_data)
        s_data = _score(s_data, u_map)
        precomputed_cache = {'sentences_data': s_data, 'num_clusters': n_clusters}

        generated_summaries["TextRank_Baseline"].append(textrank_baseline(article, num_sentences=target_len))
        generated_summaries["Original_Paper_Method"].append(model_1_summarize(article, title, num_sentences=target_len))
        ext, pol = advanced_pipeline(precomputed_cache, title, target_num_sentences=target_len)
        generated_summaries["Advanced_Extractive"].append(ext)
        generated_summaries["Hybrid"].append(pol)
        generated_summaries["LLM_Only"].append(model_4_llm_only(article, title, target_num_sentences=target_len))

    # --- Batch Metric Calculation ---
    final_report = {}
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    pbar_score = tqdm(models_to_benchmark.items(), desc="Calculating Scores")
    for model_name, config in pbar_score:
        pbar_score.set_description(f"Scoring {model_name}")
        
        candidates = generated_summaries[model_name]
        
        rouge1, rouge2, rougeL = [], [], []
        for i in range(len(candidates)):
            if candidates[i] and reference_summaries[i]:
                scores = rouge_eval.score(reference_summaries[i], candidates[i])
                rouge1.append(scores['rouge1'].fmeasure)
                rouge2.append(scores['rouge2'].fmeasure)
                rougeL.append(scores['rougeL'].fmeasure)
        
        # BERTScore requires filtering empty strings
        filtered_candidates = [c for c in candidates if c]
        filtered_references = [r for c, r in zip(candidates, reference_summaries) if c]
        if filtered_candidates:
            P, R, F1 = bert_scorer(filtered_candidates, filtered_references, lang="en", verbose=False, batch_size=16)
            bert_f1_mean = F1.mean().item()
        else:
            bert_f1_mean = 0.0

        nli_entailments, nli_contradictions = [], []
        for i in range(len(candidates)):
             entail, contra = nli_score(articles[i], candidates[i])
             nli_entailments.append(entail)
             nli_contradictions.append(contra)

        final_report[model_name] = {
            "ROUGE-1": np.mean(rouge1) if rouge1 else 0.0, "ROUGE-2": np.mean(rouge2) if rouge2 else 0.0, "ROUGE-L": np.mean(rougeL) if rougeL else 0.0,
            "BERTScore-F1": bert_f1_mean,
            "NLI-Entailment": np.mean(nli_entailments) if nli_entailments else 0.0,
            "NLI-Contradiction": np.mean(nli_contradictions) if nli_contradictions else 0.0
        }

    report_df = pd.DataFrame(final_report).T
    report_df = report_df.sort_values(by="BERTScore-F1", ascending=False)
    
    print("\n\n" + "="*95)
    print(" " * 25 + "GRAND UNIFIED QUANTITATIVE BENCHMARK REPORT" + " " * 25)
    print("="*95)
    print(report_df.to_string(formatters={
        'ROUGE-1': '{:,.4f}'.format, 'ROUGE-2': '{:,.4f}'.format, 'ROUGE-L': '{:,.4f}'.format,
        'BERTScore-F1': '{:,.4f}'.format,
        'NLI-Entailment': '{:,.4f}'.format, 'NLI-Contradiction': '{:,.4f}'.format,
    }))
    print("="*95)
    print("\nBenchmark complete.")

if __name__ == "__main__":
    run_grand_benchmark()