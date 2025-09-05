import os
import re
import json
import time
import requests
import warnings
from pathlib import Path
import numpy as np
import openai

import pandas as pd
import networkx as nx
import spacy
import hdbscan
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


CLOUD_API_KEY = "" # Your key here
CLOUD_MODEL_NAME = "gpt-oss-120b"
CLOUD_BASE_URL = "https://mkp-api.fptcloud.com/v1" 
CLOUD_API_URL = f"{CLOUD_BASE_URL}/chat/completions" 

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Create the OpenAI client ONLY for the cloud model
try:
    ## Pass the base URL directly as a string.
    import openai
    cloud_client = openai.OpenAI(api_key=CLOUD_API_KEY, base_url=CLOUD_BASE_URL)
    print("Cloud LLM client for judging configured successfully.")
except Exception as e:
    print(f"Failed to configure OpenAI client for judging: {e}")
    cloud_client = None
print("Loading global models (SBERT, spaCy)... This may take a moment.")
SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
SPACY_NLP = spacy.load("en_core_web_sm")
print("Global models loaded.")

# ==============================================================================
# --- FULL MODEL AND HELPER DEFINITIONS ---
# ==============================================================================

# --- Helper Functions for Advanced Pipeline ---
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
    for i, s_data in enumerate(sentences_data): s_data['embedding'] = embeddings[i]
    return sentences_data
def _cluster(sentences_data, min_cluster_size=2):
    embeddings = np.array([s['embedding'] for s in sentences_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=False, allow_single_cluster=True)
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max() + 1
    for i, s_data in enumerate(sentences_data): s_data['cluster_id'] = clusterer.labels_[i]
    return sentences_data, num_clusters
def _extract_keywords(sentences_data, top_n=15):
     all_tokens = [' '.join(s['tokens']) for s in sentences_data]
     if not all_tokens: return set()
     v = TfidfVectorizer(max_features=top_n);
     try:
         v.fit_transform(all_tokens)
         return set(v.get_feature_names_out())
     except ValueError:
         return set()
def _score(sentences_data, unfiltered_map):
    embeddings = np.array([s['embedding'] for s in sentences_data])
    n_orig = max(unfiltered_map.keys()) + 1 if unfiltered_map else len(sentences_data)
    keywords = _extract_keywords(sentences_data)
    graph = nx.from_numpy_array(cosine_similarity(embeddings))
    pagerank_scores = nx.pagerank(graph)
    doc_vector = np.mean(embeddings, axis=0)
    clusters = {cid: [] for cid in set(s['cluster_id'] for s in sentences_data if s['cluster_id'] != -1)}
    for s in sentences_data:
        if s['cluster_id'] != -1: clusters[s['cluster_id']].append(s['embedding'])
    centroids = {cid: np.mean(embs, axis=0) for cid, embs in clusters.items() if embs}
    for s in sentences_data:
        s['pagerank_score'] = pagerank_scores.get(s['id'], 0)
        s['global_cohesion_score'] = cosine_similarity([s['embedding']], [doc_vector])[0][0]
        cid = s['cluster_id']
        s['cluster_centrality_score'] = cosine_similarity([s['embedding']], [centroids[cid]])[0][0] if cid != -1 and cid in centroids else 0
        s['info_density_score'] = len(SPACY_NLP(s['original']).ents)
        s['structural_bonus'] = 1.0 if s['id'] == 0 or s['id'] == n_orig - 1 else 0.0
        s['key_concept_score'] = len(set(s['tokens']) & keywords) / len(keywords) if keywords else 0
    info_scores = [s['info_density_score'] for s in sentences_data]
    if max(info_scores or [0]) > 0:
        norm_info = [s / max(info_scores) for s in info_scores]
        for i, s_data in enumerate(sentences_data): s_data['info_density_score'] = norm_info[i]
    weights = {'pagerank_score': 0.25, 'cluster_centrality_score': 0.20, 'key_concept_score': 0.20, 'global_cohesion_score': 0.10, 'info_density_score': 0.10, 'structural_bonus': 0.15}
    for s in sentences_data:
        s['relevance_score'] = sum(s.get(k, 0) * w for k, w in weights.items())
    return sentences_data

# --- Model 1: Original Paper Method ---
def model_1_summarize(text, title, compression_ratio=0.3, sim_threshold=0.3):
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
    num_sentences_to_select = max(1, int(round(n * compression_ratio)))
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
    
# --- Model 2 & 3: Advanced Pipeline ---
def advanced_pipeline(precomputed_cache, title, detail_level_str):
    def _select(sentences_data, num_clusters, detail_level):
        base_sents_per_cluster = {'Concise': 1, 'Balanced': 2, 'Detailed': 3}[detail_level]
        clusters = {i: [] for i in range(num_clusters)}
        for s in sentences_data:
            if s['cluster_id'] != -1: clusters[s['cluster_id']].append(s)
        final_indices = []
        for cid in range(num_clusters):
            cluster_sents = sorted(clusters.get(cid, []), key=lambda s: s['relevance_score'], reverse=True)
            to_pick = min(base_sents_per_cluster, len(cluster_sents))
            final_indices.extend([s['id'] for s in cluster_sents[:to_pick]])
        final_indices.sort()
        return final_indices
    def _polish(extractive_text, title):
        prompt = f"You are an expert editor. Rewrite the following disconnected facts about '{title}' into a single, cohesive, and fluent summary paragraph. Preserve all key information and named entities. Do not add new information.\n\nFACTS:\n{extractive_text}\n\nPOLISHED SUMMARY:"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
        try:
            response = requests.post(LM_STUDIO_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"[LM Studio polish failed: {e}] {extractive_text}"

    sentences_data = precomputed_cache['sentences_data']
    num_clusters = precomputed_cache['num_clusters']
    best_indices = _select(sentences_data, num_clusters, detail_level_str)
    sentence_lookup = {s['id']: s for s in sentences_data}
    extractive_summary = " ".join([sentence_lookup.get(i, {}).get('original', '') for i in best_indices])
    polished_summary = _polish(extractive_summary, title)
    return extractive_summary, polished_summary

# --- Model 4: LLM-Only ---
def model_4_llm_only(text, title, detail_level_str):
    if not text or not isinstance(text, str): return ""
    detail_map = {'Concise': "1-2 sentences", 'Balanced': "3-4 sentences", 'Detailed': "5-7 sentences"}
    detail_instruction = f"The summary should be {detail_map[detail_level_str]}."
    prompt = f"You are an expert summarizer. Create a factually accurate, abstractive summary of the following text about '{title}'.\n\nINSTRUCTIONS:\n1. {detail_instruction}\n2. The summary MUST be based ONLY on the provided text.\n3. The summary must be a single, cohesive paragraph.\n\nTEXT:\n{text}\n\nSUMMARY:"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[LM Studio request failed: {e}]"
    

# --- Baseline Model: TextRank ---
def textrank_baseline(text, num_sentences=4):
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

# --- Dataset and Judge Functions ---
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

def llm_as_judge(original_article, candidate_summary):
    system_prompt = (
        "You are an expert, impartial evaluator for text summarization systems. "
        "Your sole purpose is to provide a rigorous, objective assessment of a generated summary "
        "based *only* on the provided original article. Your output MUST be a single, valid JSON object and nothing else."
    )
    user_prompt_template = (
        "{preamble}"
        "Please evaluate the 'Candidate Summary' on four criteria: Relevance, Faithfulness, Coherence, and Conciseness, "
        "providing an integer score from 1 (very poor) to 5 (excellent) for each. "
        "Base your evaluation ONLY on the provided 'Original Article'.\n\n"
        "--- ORIGINAL ARTICLE ---\n{original_article}\n\n"
        "--- CANDIDATE SUMMARY ---\n{candidate_summary}\n\n"
        "--- EVALUATION (Return ONLY the raw JSON object with keys: relevance_score, faithfulness_score, coherence_score, conciseness_score, brief_justification) ---"
    ).format(original_article=original_article[:6000], candidate_summary=candidate_summary, preamble="{preamble}")
    
    default_response = {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "brief_justification": "LLM call failed.", "error": "Unknown"}
    
    last_error = ""
    for attempt in range(2): # Retry logic
        try:
            preamble = ""
            if attempt > 0:
                preamble = f"PREVIOUS ATTEMPT FAILED. REASON: {last_error}. Please correct your output and provide ONLY the JSON object.\n\n"
            
            final_user_prompt = user_prompt_template.format(preamble=preamble)

            ## Using the robust 'requests' library for the cloud call
            headers = {"Authorization": f"Bearer {CLOUD_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": CLOUD_MODEL_NAME, 
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user_prompt}
                ], 
                "temperature": 0.1
            }

            response = requests.post(CLOUD_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status() # This will raise an error for 4xx or 5xx statuses
            
            json_response_text = response.json()['choices'][0]['message']['content']
            
            match = re.search(r'\{.*\}', json_response_text, re.DOTALL)
            if not match:
                last_error = f"Response did not contain a JSON object. Response: {json_response_text[:200]}"
                continue

            json_str = match.group(0)
            data = json.loads(json_str)
            
            score_keys = ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']
            if not all(key in data for key in score_keys):
                 last_error = f"Response missing required keys. Keys found: {list(data.keys())}"
                 continue

            final_evaluation = {}
            for key in score_keys:
                try: final_evaluation[key] = int(data.get(key, 0))
                except (ValueError, TypeError): final_evaluation[key] = 0
            
            final_evaluation["brief_justification"] = str(data.get("brief_justification", "N/A"))
            return final_evaluation

        except Exception as e:
            last_error = str(e)
            print(f"\n[DEBUG] An exception occurred during Cloud LLM Judge API call (Attempt {attempt+1}/2): {last_error}")
            time.sleep(3)
            
    default_response["error"] = last_error
    return default_response


# --- Main Evaluation Loop ---
def run_qualitative_deep_dive():
    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print("ERROR: Dataset not found.")
        return
    df = load_bbc_dataset(DATASET_PATH)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    sample_df = test_df.sample(n=50, random_state=42)
    print(f"Running qualitative deep-dive on a sample of {len(sample_df)} articles.")

    models_to_compare = {
        "TextRank_Baseline": {"type": "textrank"},
        "Original_Paper_Method": {"type": "original"},
        "Advanced_Extractive_Balanced": {"type": "advanced_extractive", "detail": "Balanced"},
        "Hybrid_Balanced": {"type": "hybrid", "detail": "Balanced"},
        "LLM_Only_Balanced": {"type": "llm_only", "detail": "Balanced"},
    }

    all_evaluations = []
    pbar = tqdm(sample_df.itertuples(), total=len(sample_df), desc="Qualitative Evaluation")
    for row in pbar:
        s_data, u_map = _preprocess(row.article)
        if not s_data: continue
        s_data = _embed(s_data, SBERT_MODEL)
        s_data, n_clusters = _cluster(s_data)
        s_data = _score(s_data, u_map)
        precomputed_cache = {'sentences_data': s_data, 'num_clusters': n_clusters}
        
        for model_name, config in models_to_compare.items():
            pbar.set_description(f"Evaluating {model_name} on {row.filename}")
            title = sent_tokenize(row.article)[0] if row.article and sent_tokenize(row.article) else "Article"
            
            generated_summary = ""
            if config["type"] == "textrank":
                generated_summary = textrank_baseline(row.article, num_sentences=4)
            elif config["type"] == "original":
                generated_summary = model_1_summarize(row.article, title, compression_ratio=0.3)
            elif config["type"] == "advanced_extractive":
                generated_summary, _ = advanced_pipeline(precomputed_cache, title, config["detail"])
            elif config["type"] == "hybrid":
                 _, generated_summary = advanced_pipeline(precomputed_cache, title, config["detail"])
            elif config["type"] == "llm_only":
                generated_summary = model_4_llm_only(row.article, title, config["detail"])

            if generated_summary:
                evaluation = llm_as_judge(row.article, generated_summary)
                evaluation['model'] = model_name
                evaluation['filename'] = row.filename
                all_evaluations.append(evaluation)
            time.sleep(1)

    eval_df = pd.DataFrame(all_evaluations)
    print("\n\n" + "="*80)
    print("                 QUALITATIVE DEEP-DIVE REPORT (Cloud LLM as Judge)               ")
    print("="*80)
    
    if not eval_df.empty and 'relevance_score' in eval_df.columns:
        avg_scores = eval_df.groupby('model')[['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']].mean()
        model_order = ["TextRank_Baseline", "Original_Paper_Method", "Advanced_Extractive_Balanced", "Hybrid_Balanced", "LLM_Only_Balanced"]
        avg_scores = avg_scores.reindex(model_order).dropna(how='all')
        print("\n--- Average Scores Across All Samples (1=Poor, 5=Excellent) ---")
        print(avg_scores.to_string(float_format="{:.2f}".format))
    else:
        print("\nCould not generate average scores.")

    print("\n" + "="*80)
    print("Deep-dive complete.")

if __name__ == "__main__":
    run_qualitative_deep_dive()