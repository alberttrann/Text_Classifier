import os
import re
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

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# --- MODEL DEFINITIONS & HELPERS ---
# ==============================================================================

print("Loading global models (SBERT, spaCy, NLI)... This may take a moment.")
SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=DEVICE)
SPACY_NLP = spacy.load("en_core_web_sm")
NLI_PIPELINE = hf_pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=0 if torch.cuda.is_available() else -1)
print("Global models loaded.")

def _preprocess(text):
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
def _score(sentences_data, unfiltered_map):
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

# --- The "Faithfulness-Guaranteed" Hybrid Model ---
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

def verified_polish_with_llm(extractive_summary, original_article, title, max_retries=2):
    current_summary = extractive_summary
    for i in range(max_retries + 1):
        if i == 0:
            prompt_type = "Initial Polish"
            prompt = (
                "You are an expert editor with a strict focus on factual accuracy. Your task is to rewrite the "
                "following set of disconnected facts into a single, cohesive, and fluent summary paragraph about '{title}'.\n\n"
                "**CRITICAL RULES:**\n"
                "1.  Your primary goal is FACTUAL ACCURACY. You MUST NOT add any information, entities, or nuances that are not explicitly present in the facts provided.\n"
                "2.  You MUST NOT distort the meaning of the original facts.\n"
                "3.  Your secondary goal is to improve narrative flow, resolve pronouns, and add natural transitions.\n\n"
                "--- START OF FACTS ---\n{facts}\n--- END OF FACTS ---\n\n"
                "Polished, single-paragraph summary:"
            ).format(title=title, facts=current_summary)
        else:
            prompt_type = f"Refinement Attempt {i}"
            prompt = (
                "Your previous summary was not fully faithful to the original source text. "
                "Please try again. Rewrite the following summary to be more strictly and accurately "
                "grounded in the provided 'Original Facts'. Do not add or change any information.\n\n"
                "--- ORIGINAL FACTS ---\n{facts}\n\n"
                "--- YOUR PREVIOUS (FLAWED) SUMMARY ---\n{summary}\n\n"
                "--- REVISED, MORE FACTUAL SUMMARY ---"
            ).format(facts=extractive_summary, summary=current_summary)
        url = "http://localhost:1234/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "loaded-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            polished_candidate = response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"LLM polish failed on attempt {i}: {e}")
            return extractive_summary
        entailment, contradiction = nli_score(original_article, polished_candidate)
        print(f"  - {prompt_type}: NLI Entailment={entailment:.4f}, Contradiction={contradiction:.4f}")
        if entailment >= 0.90 and contradiction <= 0.1:
            print("  - Summary PASSED factual consistency check.")
            return polished_candidate
        current_summary = polished_candidate
    print("  - FAILED to produce a faithful summary after all retries. Falling back to extractive version.")
    return extractive_summary

def faithfulness_guaranteed_hybrid_summarizer(text, title, target_num_sentences):
    s_data, u_map = _preprocess(text)
    if not s_data: return ""
    s_data = _embed(s_data, SBERT_MODEL)
    s_data, n_clusters = _cluster(s_data)
    s_data = _score(s_data, u_map)
    precomputed_cache = {'sentences_data': s_data, 'num_clusters': n_clusters}
    
    # Selection logic from the advanced pipeline
    sorted_sents = sorted(s_data, key=lambda s: s['relevance_score'], reverse=True)
    top_indices = [s['id'] for s in sorted_sents[:target_num_sentences]]
    top_indices.sort()
    sentence_lookup = {s['id']: s for s in s_data}
    extractive_summary = " ".join([sentence_lookup.get(i, {}).get('original', '') for i in top_indices])
    
    # The new, verified polishing stage
    final_summary = verified_polish_with_llm(extractive_summary, text, title)
    return final_summary

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
def run_final_hybrid_benchmark():
    # --- 1. Load Data ---
    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
        
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nRunning final benchmark on {len(test_df)} articles.")

    # --- 2. Generation and Scoring Loop ---
    candidates = []
    references = list(test_df['reference_summary'])
    articles = list(test_df['article'])
    
    pbar_generate = tqdm(range(len(test_df)), desc="Generating Verified Hybrid Summaries")
    for i in pbar_generate:
        article = articles[i]
        reference = references[i]
        title = sent_tokenize(article)[0] if article and sent_tokenize(article) else "Article"
        
        try:
            target_len = len(sent_tokenize(reference))
        except:
            target_len = 3
        
        summary = faithfulness_guaranteed_hybrid_summarizer(article, title, target_num_sentences=target_len)
        candidates.append(summary)

    # --- 3. Batch Metric Calculation ---
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results, nli_results = [], []

    print("\nCalculating ROUGE and NLI scores...")
    for i in tqdm(range(len(candidates)), desc="Scoring"):
        if candidates[i] and references[i]:
            rouge_scores = rouge_eval.score(references[i], candidates[i])
            rouge_results.append(rouge_scores)
            
            nli_entail, nli_contra = nli_score(articles[i], candidates[i])
            nli_results.append({'entailment': nli_entail, 'contradiction': nli_contra})

    print("Calculating BERTScore (this may take a while)...")
    _, _, bert_f1_scores = bert_scorer(candidates, references, lang="en", verbose=True, batch_size=16)
    
    # --- 4. Print Final Report Card ---
    rouge1 = np.mean([r['rouge1'].fmeasure for r in rouge_results])
    rouge2 = np.mean([r['rouge2'].fmeasure for r in rouge_results])
    rougeL = np.mean([r['rougeL'].fmeasure for r in rouge_results])
    bertscore_f1 = bert_f1_scores.mean().item()
    nli_entailment = np.mean([n['entailment'] for n in nli_results])
    nli_contradiction = np.mean([n['contradiction'] for n in nli_results])

    print("\n\n" + "="*80)
    print("      FINAL EVALUATION REPORT: Faithfulness-Guaranteed Hybrid Model      ")
    print("="*80)
    print("\n--- Quantitative Metrics (Average Scores on Full Test Set) ---")
    print(f"Avg ROUGE-1 F1:      {rouge1:.4f}")
    print(f"Avg ROUGE-2 F1:      {rouge2:.4f}")
    print(f"Avg ROUGE-L F1:      {rougeL:.4f}")
    print("-" * 40)
    print(f"Avg BERTScore F1:    {bertscore_f1:.4f}")
    print("-" * 40)
    print(f"Avg NLI Entailment:  {nli_entailment:.4f} (Higher is better)")
    print(f"Avg NLI Contradiction: {nli_contradiction:.4f} (Lower is better)")
    print("\n" + "="*80)
    print("Benchmark complete.")

if __name__ == "__main__":
    run_final_hybrid_benchmark()