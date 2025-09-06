import os
import re
import json
import time
import requests
import warnings
from pathlib import Path
import openai
import torch
import networkx as nx

import numpy as np
import pandas as pd
import spacy
import hdbscan
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import pipeline as hf_pipeline
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLOUD_API_KEY = ""
CLOUD_MODEL_NAME = "gpt-oss-120b"
CLOUD_BASE_URL = "https://mkp-api.fptcloud.com"

try:
    cloud_client = openai.OpenAI(api_key=CLOUD_API_KEY, base_url=CLOUD_BASE_URL)
    print("Cloud LLM client for judging configured successfully.")
except Exception as e:
    print(f"Failed to configure OpenAI client for judging: {e}")
    cloud_client = None

print("Loading global models (SBERT, spaCy, NLI)... This may take a moment.")
try:
    SBERT_MODEL = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=DEVICE)
    print("SBERT model loaded successfully.")
except Exception as e:
    print(f"Failed to load SBERT model: {e}")
    SBERT_MODEL = None

try:
    SPACY_NLP = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except OSError as e:
    print(f"Failed to load spaCy model: {e}")
    print("Please install with: python -m spacy download en_core_web_sm")
    SPACY_NLP = None

try:
    NLI_PIPELINE = hf_pipeline("text-classification", model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device=0 if torch.cuda.is_available() else -1)
    print("NLI pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load NLI pipeline: {e}")
    NLI_PIPELINE = None

print("Global models loading complete.")

# ==============================================================================
# --- FULL MODEL AND HELPER DEFINITIONS ---
# (These are the functions that constitute final hybrid model)
# ==============================================================================

def _preprocess(text):
    if not SPACY_NLP:
        return [], {}
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    try: 
        original_sentences = sent_tokenize(text)
    except Exception: 
        return [], {}
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
    if not sbert_model:
        return sentences_data
    original_sentences = [s['original'] for s in sentences_data]
    embeddings = sbert_model.encode(original_sentences, show_progress_bar=False)
    for i, s_data in enumerate(sentences_data): 
        s_data['embedding'] = embeddings[i]
    return sentences_data

def _cluster(sentences_data, min_cluster_size=2):
    if not sentences_data:
        return sentences_data, 0
    embeddings = np.array([s['embedding'] for s in sentences_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=False, allow_single_cluster=True)
    clusterer.fit(embeddings)
    num_clusters = clusterer.labels_.max() + 1
    for i, s_data in enumerate(sentences_data): 
        s_data['cluster_id'] = clusterer.labels_[i]
    return sentences_data, num_clusters

def _score(sentences_data, unfiltered_map):
    if not sentences_data or not SPACY_NLP:
        return sentences_data
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
        for i, s_data in enumerate(sentences_data): 
            s_data['info_density_score'] = norm_info[i]
    weights = {'pagerank_score': 0.4, 'global_cohesion_score': 0.3, 'info_density_score': 0.3}
    for s in sentences_data:
        s['relevance_score'] = sum(s.get(k, 0) * w for k, w in weights.items())
    return sentences_data

def nli_score(article_text, summary_text):
    if not article_text or not summary_text or not SBERT_MODEL or not NLI_PIPELINE: 
        return 0.0, 0.0
    try:
        article_sentences = sent_tokenize(article_text)
        summary_sentences = sent_tokenize(summary_text)
        if not article_sentences or not summary_sentences: 
            return 0.0, 0.0
    except Exception: 
        return 0.0, 0.0
    article_embeddings = SBERT_MODEL.encode(article_sentences, convert_to_tensor=True, show_progress_bar=False)
    entailment_scores, contradiction_scores = [], []
    for summary_sent in summary_sentences:
        summary_sent_embedding = SBERT_MODEL.encode(summary_sent, convert_to_tensor=True, show_progress_bar=False)
        similarities = util.pytorch_cos_sim(summary_sent_embedding, article_embeddings)[0]
        premise_idx = torch.argmax(similarities).item()
        premise_sent = article_sentences[premise_idx]
        try:
            nli_results = NLI_PIPELINE(f"{premise_sent}</s></s>{summary_sent}")
            entailment_score, contradiction_score = 0.0, 0.0
            for result in nli_results:
                if result['label'] == 'entailment': 
                    entailment_score = result['score']
                elif result['label'] == 'contradiction': 
                    contradiction_score = result['score']
            entailment_scores.append(entailment_score)
            contradiction_scores.append(contradiction_score)
        except Exception as e:
            print(f"NLI pipeline error: {e}")
            entailment_scores.append(0.0)
            contradiction_scores.append(0.0)
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
        payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            polished_candidate = response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            print(f"Local LLM polish failed on attempt {i}: {e}")
            return extractive_summary
        entailment, contradiction = nli_score(original_article, polished_candidate)
        print(f"  - {prompt_type}: NLI Entailment={entailment:.4f}, Contradiction={contradiction:.4f}")
        if entailment >= 0.90 and contradiction <= 0.1:
            print("  - Summary PASSED factual consistency check.")
            return polished_candidate
        current_summary = polished_candidate
    print("  - FAILED to produce a faithful summary. Falling back to extractive version.")
    return extractive_summary

def faithfulness_guaranteed_hybrid_summarizer(text, title, target_num_sentences):
    if not all([SBERT_MODEL, SPACY_NLP]):
        print("Error: Required models not loaded. Falling back to simple extraction.")
        sentences = sent_tokenize(text)
        return " ".join(sentences[:target_num_sentences])
    
    s_data, u_map = _preprocess(text)
    if not s_data: 
        return ""
    s_data = _embed(s_data, SBERT_MODEL)
    s_data, n_clusters = _cluster(s_data)
    s_data = _score(s_data, u_map)
    
    sorted_sents = sorted(s_data, key=lambda s: s['relevance_score'], reverse=True)
    top_indices = [s['id'] for s in sorted_sents[:target_num_sentences]]
    top_indices.sort()
    sentence_lookup = {s['id']: s for s in s_data}
    extractive_summary = " ".join([sentence_lookup.get(i, {}).get('original', '') for i in top_indices])
    
    # final_summary = verified_polish_with_llm(extractive_summary, text, title)
    final_summary = extractive_summary  # Use extractive summary directly
    return final_summary

# ==============================================================================
# --- LLM-as-a-Judge and Dataset Functions ---
# ==============================================================================
def llm_as_judge(original_article, candidate_summary):
    """
    Uses a cloud LLM to score a candidate summary based ONLY on the original article.
    """
    if not cloud_client:
        return {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "error": "Cloud client not configured."}

    prompt = (
        "You are an expert evaluator for text summarization systems. Your task is to provide a rigorous, objective "
        "assessment of a generated 'Candidate Summary' based *only* on the provided 'Original Article'. "
        "Do not use any external knowledge.\n\n"
        "Please evaluate the 'Candidate Summary' on four criteria, providing an integer score from 1 (very poor) "
        "to 5 (excellent) for each. You MUST provide your final answer in JSON format, with keys: relevance_score, "
        "faithfulness_score, coherence_score, conciseness_score, and a brief_justification string.\n\n"
        "--- ORIGINAL ARTICLE ---\n{original_article}\n\n"
        "--- CANDIDATE SUMMARY ---\n{candidate_summary}\n\n"
        "--- EVALUATION (Return ONLY the raw JSON object) ---"
    ).format(original_article=original_article[:6000], candidate_summary=candidate_summary)
    
    default_response = {"relevance_score": 0, "faithfulness_score": 0, "coherence_score": 0, "conciseness_score": 0, "brief_justification": "LLM call failed.", "error": "Unknown"}
    
    try:
        response = cloud_client.chat.completions.create(
            model=CLOUD_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        
        json_response_text = response.choices[0].message.content.strip()
        print(f"Raw LLM response: {json_response_text[:200]}...")  
        
        try:
            data = json.loads(json_response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', json_response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                print(f"No valid JSON found in response: {json_response_text}")
                return default_response
        
        final_evaluation = {}
        score_keys = ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']
        for key in score_keys:
            try:
                score = data.get(key, 0)
                # Handle various formats
                if isinstance(score, str):
                    score = int(re.search(r'\d+', score).group()) if re.search(r'\d+', score) else 0
                final_evaluation[key] = int(score)
            except (ValueError, TypeError, AttributeError):
                final_evaluation[key] = 0
        final_evaluation["brief_justification"] = str(data.get("brief_justification", "N/A"))
        
        print(f"Parsed scores: {final_evaluation}")  
        return final_evaluation

    except Exception as e:
        print(f"Cloud LLM Judge Error: {e}")
        default_response["error"] = str(e)
        return default_response

def load_bbc_dataset(base_path):
    print(f"Loading dataset from: {base_path}")
    all_data = []
    articles_path = Path(base_path) / "News Articles"
    summaries_path = Path(base_path) / "Summaries"
    
    if not articles_path.exists() or not summaries_path.exists():
        print(f"Error: Dataset paths do not exist. Expected structure:")
        print(f"  {base_path}/News Articles/")
        print(f"  {base_path}/Summaries/")
        return pd.DataFrame()
    
    for category_path in articles_path.iterdir():
        if category_path.is_dir():
            category = category_path.name
            for article_file in category_path.glob("*.txt"):
                try:
                    with open(article_file, 'r', encoding='utf-8', errors='ignore') as f: 
                        article_content = f.read()
                    summary_file = summaries_path / category / article_file.name
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f: 
                            summary_content = f.read()
                        all_data.append({
                            "filename": article_file.name, 
                            "category": category, 
                            "article": article_content, 
                            "reference_summary": summary_content
                        })
                except Exception as e:
                    print(f"Skipping file {article_file.name} due to error: {e}")
    
    print(f"Loaded {len(all_data)} articles from dataset")
    return pd.DataFrame(all_data)

# ==============================================================================
# --- Main Evaluation Loop ---
# ==============================================================================
def run_hybrid_qualitative_benchmark():
    if not all([SBERT_MODEL, SPACY_NLP]):
        print("ERROR: Required models not loaded. Cannot proceed with evaluation.")
        return
        
    DATASET_PATH = "./BBC News Summary"
    if not Path(DATASET_PATH).exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    df = load_bbc_dataset(DATASET_PATH)
    if df.empty:
        print("ERROR: No data loaded from dataset")
        return
    
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    sample_size = min(10, len(test_df))  # Start with smaller sample for testing
    sample_df = test_df.sample(n=50, random_state=42)
    print(f"Running qualitative deep-dive on a sample of {len(sample_df)} articles.")

    all_evaluations = []
    successful_evaluations = 0
    
    pbar = tqdm(sample_df.itertuples(), total=len(sample_df), desc="Qualitative Evaluation of Hybrid Model")
    for idx, row in enumerate(pbar):
        print(f"\n--- Processing article {idx+1}/{len(sample_df)}: {row.filename} ---")
        
        # Fair Length Control
        try:
            target_len = len(sent_tokenize(row.reference_summary))
        except:
            target_len = 3
        
        print(f"Target summary length: {target_len} sentences")
        
        title = sent_tokenize(row.article)[0] if row.article and sent_tokenize(row.article) else "Article"
        
        generated_summary = faithfulness_guaranteed_hybrid_summarizer(row.article, title, target_num_sentences=target_len)
        
        if generated_summary:
            print(f"Generated summary: {generated_summary[:200]}...")
            
            # Get the LLM-as-a-Judge evaluation (reference-free)
            evaluation = llm_as_judge(row.article, generated_summary)
            
            # Check if evaluation was successful (non-zero scores)
            if any(evaluation.get(key, 0) > 0 for key in ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']):
                successful_evaluations += 1
            
            all_evaluations.append(evaluation)
        else:
            print("Failed to generate summary")
            all_evaluations.append({
                "relevance_score": 0, 
                "faithfulness_score": 0, 
                "coherence_score": 0, 
                "conciseness_score": 0,
                "brief_justification": "Summary generation failed",
                "error": "No summary generated"
            })
        
        time.sleep(2) 

    # --- Print Final Report ---
    eval_df = pd.DataFrame(all_evaluations)
    print("\n\n" + "="*80)
    print("      QUALITATIVE DEEP-DIVE REPORT: Faithfulness-Guaranteed Hybrid Model      ")
    print("="*80)
    
    print(f"Total samples processed: {len(all_evaluations)}")
    print(f"Successful evaluations: {successful_evaluations}")
    
    if not eval_df.empty and 'relevance_score' in eval_df.columns:
        avg_scores = eval_df[['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']].mean()
        print("\n--- Average Scores Across All Samples (1=Poor, 5=Excellent) ---")
        print(avg_scores.to_frame(name='Average Score').to_string(float_format="{:.2f}".format))
        
        # Show non-zero scores only
        non_zero_mask = (eval_df[['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']] > 0).any(axis=1)
        if non_zero_mask.sum() > 0:
            avg_scores_non_zero = eval_df[non_zero_mask][['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']].mean()
            print(f"\n--- Average Scores for Successful Evaluations ({non_zero_mask.sum()} samples) ---")
            print(avg_scores_non_zero.to_frame(name='Average Score').to_string(float_format="{:.2f}".format))
    else:
        print("\nCould not generate average scores. This might happen if all LLM-as-a-Judge calls failed.")
    
    # Show some example evaluations
    if successful_evaluations > 0:
        print("\n--- Sample Evaluations ---")
        for i, eval_result in enumerate(all_evaluations[:3]):
            if any(eval_result.get(key, 0) > 0 for key in ['relevance_score', 'faithfulness_score', 'coherence_score', 'conciseness_score']):
                print(f"Sample {i+1}: {eval_result}")

    print("\n" + "="*80)
    print("Deep-dive complete.")

if __name__ == "__main__":
    if not all([SBERT_MODEL, SPACY_NLP, cloud_client]):
        print("\nAborting benchmark due to initialization errors.")
        print("Required components:")
        print(f"  - SBERT Model: {'✓' if SBERT_MODEL else '✗'}")
        print(f"  - spaCy NLP: {'✓' if SPACY_NLP else '✗'}")
        print(f"  - Cloud Client: {'✓' if cloud_client else '✗'}")
    else:
        run_hybrid_qualitative_benchmark()